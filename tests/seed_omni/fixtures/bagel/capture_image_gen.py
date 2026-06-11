"""Capture official BAGEL base image-generation one-step velocity fixtures.

The generated fixture is an oracle artifact for SeedOmni V2 parity work. It may
contain official ``packed_*`` field names, but those names must stay on the
capture/test side and must not become V2 runtime module inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights


pytestmark = pytest.mark.skip(reason="BAGEL official capture helper; run explicitly to generate parity fixtures.")

DEFAULT_OUTPUT = Path("outputs/bagel_v2/parity/image_generation_one_step_velocity.pt")
DEFAULT_PROMPT = "A small ceramic teapot on a wooden table, soft morning light."


def _log(message: str) -> None:
    print(f"[bagel-image-gen-fixture] {message}", flush=True)


def _compute_default_rope_parameters(
    config: Any, device: torch.device | None = None, **kwargs: Any
) -> tuple[torch.Tensor, float]:
    del kwargs
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def _import_official(official_repo: Path) -> dict[str, Any]:
    _log(f"importing official BAGEL from {official_repo}")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    sys.path.insert(0, str(official_repo))
    from data.data_utils import add_special_tokens
    from modeling.autoencoder import AutoEncoderParams
    from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel.qwen2_navit import NaiveCache
    from modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    return {
        "add_special_tokens": add_special_tokens,
        "AutoEncoderParams": AutoEncoderParams,
        "Bagel": Bagel,
        "BagelConfig": BagelConfig,
        "NaiveCache": NaiveCache,
        "Qwen2Config": Qwen2Config,
        "Qwen2ForCausalLM": Qwen2ForCausalLM,
        "Qwen2Tokenizer": Qwen2Tokenizer,
    }


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast("cuda", enabled=True, dtype=dtype)


def _move_tensors(data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}


def _cpu_tensors(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in data.items()}


def _cache_to_tensors(cache: Any) -> dict[str, Any]:
    keys: list[torch.Tensor | None] = []
    values: list[torch.Tensor | None] = []
    for idx in range(cache.num_layers):
        key = cache.key_cache[idx]
        value = cache.value_cache[idx]
        keys.append(None if key is None else key.detach().cpu())
        values.append(None if value is None else value.detach().cpu())
    return {
        "num_layers": cache.num_layers,
        "key": keys,
        "value": values,
    }


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _vae_config(official: dict[str, Any]) -> Any:
    return official["AutoEncoderParams"](
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )


def _load_state(path: Path, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    prefixes = (
        "language_model.",
        "vae2llm.",
        "llm2vae.",
        "time_embedder.",
        "latent_pos_embed.",
    )
    _log(f"loading text+flow weights from {path} to {device} as {dtype}")
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefixes):
                state_dict[key] = f.get_tensor(key).to(device=device, dtype=dtype)
    _log(f"loaded {len(state_dict)} text+flow tensors")
    return state_dict


def _build_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, int], torch.device, torch.dtype]:
    official = _import_official(args.official_repo)
    device = torch.device(args.device)
    torch_dtype = _resolve_dtype(args.dtype)

    _log("reading official llm_config.json")
    llm_config = official["Qwen2Config"].from_json_file(str(args.model_root / "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = llm_config.bos_token_id

    _log(f"loading tokenizer from {args.model_root}")
    tokenizer = official["Qwen2Tokenizer"].from_pretrained(str(args.model_root), local_files_only=True)
    tokenizer, new_token_ids, num_new_tokens = official["add_special_tokens"](tokenizer)
    if num_new_tokens > 0:
        llm_config.vocab_size = len(tokenizer)

    config = official["BagelConfig"](
        visual_gen=True,
        visual_und=False,
        llm_config=llm_config,
        vit_config=None,
        vae_config=_vae_config(official),
        latent_patch_size=args.latent_patch_size,
        max_latent_size=args.max_latent_size,
        timestep_shift=args.timestep_shift,
    )
    _log("constructing official text+flow BAGEL model on meta parameters")
    with no_init_weights(), init_empty_weights():
        language_model = official["Qwen2ForCausalLM"](llm_config)
        model = official["Bagel"](language_model, None, config)

    state = _load_state(args.model_root / "ema.safetensors", device=device, dtype=torch_dtype)
    _log("assigning text+flow weights")
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    relevant_prefixes = ("language_model.", "vae2llm.", "llm2vae.", "time_embedder.", "latent_pos_embed.")
    unexpected_relevant = [key for key in unexpected if key.startswith(relevant_prefixes)]
    if unexpected_relevant:
        raise RuntimeError(f"Unexpected text+flow keys while loading official BAGEL: {unexpected_relevant[:20]}")
    missing_relevant = [
        key for key in missing if key.startswith(("language_model.", "vae2llm.", "llm2vae.", "time_embedder."))
    ]
    if missing_relevant:
        raise RuntimeError(f"Missing text+flow keys while loading official BAGEL: {missing_relevant[:20]}")

    _log("moving buffers to target device")
    model.to(device=device)
    model.eval()
    _log("model ready")
    return model, tokenizer, new_token_ids, device, torch_dtype


def _first_flow_timestep(num_timesteps: int, timestep_shift: float, device: torch.device) -> dict[str, torch.Tensor]:
    timesteps_full = torch.linspace(1, 0, num_timesteps, device=device)
    timesteps_shifted = timestep_shift * timesteps_full / (1 + (timestep_shift - 1) * timesteps_full)
    dts = timesteps_shifted[:-1] - timesteps_shifted[1:]
    return {
        "timesteps": timesteps_shifted[:-1],
        "dts": dts,
        "timestep": timesteps_shifted[:1],
        "dt": dts[:1],
    }


def _flow_step_count(args: argparse.Namespace, num_available_steps: int) -> int:
    flow_steps = int(args.capture_flow_steps)
    if flow_steps < 1:
        raise ValueError("capture_flow_steps must be at least 1.")
    if flow_steps > num_available_steps:
        raise ValueError(
            f"capture_flow_steps={flow_steps} exceeds available denoise steps {num_available_steps} "
            f"from num_timesteps={args.num_timesteps}."
        )
    return flow_steps


@torch.no_grad()
def _forward_flow_base(
    model: Any,
    *,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    latent_input: dict[str, torch.Tensor],
    past_key_values: Any,
) -> dict[str, torch.Tensor]:
    packed_text_ids = latent_input["packed_text_ids"]
    packed_text_indexes = latent_input["packed_text_indexes"]
    packed_vae_token_indexes = latent_input["packed_vae_token_indexes"]
    packed_vae_position_ids = latent_input["packed_vae_position_ids"]
    packed_seqlens = latent_input["packed_seqlens"]

    packed_text_embedding = model.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence = packed_text_embedding.new_zeros((int(packed_seqlens.sum().item()), model.hidden_size))
    packed_sequence[packed_text_indexes] = packed_text_embedding

    packed_timestep = torch.full((x_t.shape[0],), float(timestep.item()), device=x_t.device, dtype=x_t.dtype)
    packed_pos_embed = model.latent_pos_embed(packed_vae_position_ids)
    packed_timestep_embeds = model.time_embedder(packed_timestep)
    latent_embeds = model.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
    if latent_embeds.dtype != packed_sequence.dtype:
        latent_embeds = latent_embeds.to(packed_sequence.dtype)
    packed_sequence[packed_vae_token_indexes] = latent_embeds

    extra_inputs = {}
    if model.use_moe:
        extra_inputs = {
            "mode": "gen",
            "packed_vae_token_indexes": packed_vae_token_indexes,
            "packed_text_indexes": packed_text_indexes,
        }

    output = model.language_model.forward_inference(
        packed_query_sequence=packed_sequence,
        query_lens=packed_seqlens,
        packed_query_position_ids=latent_input["packed_position_ids"],
        packed_query_indexes=latent_input["packed_indexes"],
        past_key_values=past_key_values,
        key_values_lens=latent_input["key_values_lens"],
        packed_key_value_indexes=latent_input["packed_key_value_indexes"],
        update_past_key_values=False,
        is_causal=False,
        **extra_inputs,
    )
    velocity = model.llm2vae(output.packed_query_sequence)[packed_vae_token_indexes]
    return {
        "packed_sequence": packed_sequence.detach(),
        "latent_embeds": latent_embeds.detach(),
        "hidden_state": output.packed_query_sequence.detach(),
        "velocity": velocity.detach(),
    }


@torch.no_grad()
def capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.cfg_text_scale != 1.0 or args.cfg_img_scale != 1.0:
        raise ValueError("Base image-generation fixture capture requires CFG disabled: both CFG scales must be 1.0.")
    if args.num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2 to capture the first Euler update.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, new_token_ids, device, torch_dtype = _build_model(args)
    official = _import_official(args.official_repo)

    past_key_values = official["NaiveCache"](model.config.llm_config.num_hidden_layers)
    curr_kvlens = [0]
    curr_rope = [0]

    prompt_input, kv_lens_after_prompt, ropes_after_prompt = model.prepare_prompts(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        prompts=[args.prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    prompt_input = _move_tensors(prompt_input, device)
    with _autocast_context(device, torch_dtype):
        past_key_values = model.forward_cache_update_text(past_key_values, **prompt_input)
    cache_after_prompt = _cache_to_tensors(past_key_values)

    latent_input = model.prepare_vae_latent(
        curr_kvlens=kv_lens_after_prompt,
        curr_rope=ropes_after_prompt,
        image_sizes=[(args.image_height, args.image_width)],
        new_token_ids=new_token_ids,
    )
    latent_input = _move_tensors(latent_input, device)

    timestep_fields = _first_flow_timestep(args.num_timesteps, args.timestep_shift, device)
    flow_steps = _flow_step_count(args, int(timestep_fields["dts"].numel()))
    x_t0 = latent_input["packed_init_noises"]
    with _autocast_context(device, torch_dtype):
        flow_output = _forward_flow_base(
            model,
            x_t=x_t0,
            timestep=timestep_fields["timestep"],
            latent_input=latent_input,
            past_key_values=past_key_values,
        )
    x_t1 = x_t0 - flow_output["velocity"].to(x_t0.device) * timestep_fields["dt"][0]
    x_t_final = x_t0
    last_flow_output = flow_output
    # Capture the official Euler chain as the graph-level oracle for multi-step denoise parity.
    for step_index in range(flow_steps):
        timestep = timestep_fields["timesteps"][step_index : step_index + 1]
        with _autocast_context(device, torch_dtype):
            last_flow_output = _forward_flow_base(
                model,
                x_t=x_t_final,
                timestep=timestep,
                latent_input=latent_input,
                past_key_values=past_key_values,
            )
        x_t_final = x_t_final - last_flow_output["velocity"].to(x_t_final.device) * timestep_fields["dts"][step_index]

    return {
        "metadata": {
            "schema_version": 1,
            "case_id": "image_generation_one_step_velocity",
            "boundary": "official.prepare_prompts.forward_cache_update_text.prepare_vae_latent.one_step_forward_flow",
            "dtype": args.dtype,
            "seed": args.seed,
            "official_repo": str(args.official_repo),
            "official_checkpoint": str(args.model_root),
            "device": str(device),
            "multi_step_flow_steps": flow_steps,
        },
        "raw_input": {
            "prompt": args.prompt,
            "image_size": [args.image_height, args.image_width],
            "num_timesteps": args.num_timesteps,
            "timestep_shift": args.timestep_shift,
            "cfg_text_scale": args.cfg_text_scale,
            "cfg_img_scale": args.cfg_img_scale,
            "cfg_interval": [args.cfg_interval_start, args.cfg_interval_end],
            "cfg_renorm_min": args.cfg_renorm_min,
            "cfg_renorm_type": args.cfg_renorm_type,
            "enable_taylorseer": False,
        },
        "rng_state": _rng_state(),
        "tokenizer": {
            "new_token_ids": dict(new_token_ids),
            "encoded_prompt_ids": tokenizer.encode(args.prompt),
        },
        "prepared": {
            "prompt": _cpu_tensors(prompt_input),
            "kv_lens_after_prompt": list(kv_lens_after_prompt),
            "ropes_after_prompt": list(ropes_after_prompt),
            "latent": _cpu_tensors(latent_input),
            "timesteps": {
                "timesteps": timestep_fields["timesteps"].detach().cpu(),
                "dts": timestep_fields["dts"].detach().cpu(),
                "timestep": timestep_fields["timestep"].detach().cpu(),
                "dt": timestep_fields["dt"].detach().cpu(),
            },
        },
        "cache_after_prompt": cache_after_prompt,
        "one_step": {
            "x_t0": x_t0.detach().cpu(),
            "packed_sequence": flow_output["packed_sequence"].detach().cpu(),
            "latent_embeds": flow_output["latent_embeds"].detach().cpu(),
            "hidden_state": flow_output["hidden_state"].detach().cpu(),
            "velocity": flow_output["velocity"].detach().cpu(),
            "x_t1": x_t1.detach().cpu(),
        },
        "multi_step": {
            "flow_steps": flow_steps,
            "x_t_final": x_t_final.detach().cpu(),
            "last_velocity": last_flow_output["velocity"].detach().cpu(),
        },
        "tolerances": {
            "bf16": {
                "v2_parity": {
                    "max_abs_diff": 1.0e-2,
                    "mean_abs_diff": 1.0e-4,
                    "cosine_similarity_min": 0.9999,
                    "source": "initial V2-vs-official bf16 image-generation velocity gate",
                }
            }
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-height", type=int, default=1024)
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--cfg-text-scale", type=float, default=1.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="global")
    parser.add_argument("--latent-patch-size", type=int, default=2)
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--capture-flow-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_root.exists():
        raise FileNotFoundError(f"BAGEL checkpoint root does not exist: {args.model_root}")
    if not args.official_repo.exists():
        raise FileNotFoundError(f"Official BAGEL repo does not exist: {args.official_repo}")

    fixture = capture(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.output)
    print(json.dumps({"output": str(args.output), "case_id": fixture["metadata"]["case_id"]}, indent=2))


if __name__ == "__main__":
    main()
