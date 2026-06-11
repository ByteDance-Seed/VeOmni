"""Capture official BAGEL module-level backward fixtures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch
from safetensors import safe_open
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights


pytestmark = pytest.mark.skip(reason="BAGEL official gradient capture helper; run explicitly for parity fixtures.")

DEFAULT_OUTPUT = Path("outputs/bagel_v2/parity/gradient_ce_only_bf16.pt")
DEFAULT_TEXT = "Describe BAGEL in one short sentence."


def _log(message: str) -> None:
    print(f"[bagel-gradient-fixture] {message}", flush=True)


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
    from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    return {
        "add_special_tokens": add_special_tokens,
        "AutoEncoderParams": AutoEncoderParams,
        "Bagel": Bagel,
        "BagelConfig": BagelConfig,
        "Qwen2Config": Qwen2Config,
        "Qwen2ForCausalLM": Qwen2ForCausalLM,
        "Qwen2Tokenizer": Qwen2Tokenizer,
        "SiglipVisionConfig": SiglipVisionConfig,
        "SiglipVisionModel": SiglipVisionModel,
    }


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")


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


def _load_state(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    visual_und: bool,
    visual_gen: bool,
) -> dict[str, torch.Tensor]:
    prefixes = ["language_model."]
    if visual_und:
        prefixes.extend(["vit_model.", "connector.", "vit_pos_embed."])
    if visual_gen:
        prefixes.extend(["vae2llm.", "llm2vae.", "time_embedder.", "latent_pos_embed."])
    _log(f"loading {prefixes} weights from {path} to {device} as {dtype}")
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(tuple(prefixes)):
                state_dict[key] = f.get_tensor(key).to(device=device, dtype=dtype)
    _log(f"loaded {len(state_dict)} tensors")
    return state_dict


def _build_model(args: argparse.Namespace, *, visual_und: bool, visual_gen: bool) -> tuple[Any, Any, torch.device]:
    official = _import_official(args.official_repo)
    device = torch.device(args.device)
    torch_dtype = _resolve_dtype(args.dtype)

    llm_config = official["Qwen2Config"].from_json_file(str(args.model_root / "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = llm_config.bos_token_id

    vit_config = None
    if visual_und:
        vit_config = official["SiglipVisionConfig"].from_json_file(str(args.model_root / "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

    tokenizer = official["Qwen2Tokenizer"].from_pretrained(str(args.model_root), local_files_only=True)
    tokenizer, _, num_new_tokens = official["add_special_tokens"](tokenizer)
    if num_new_tokens > 0:
        llm_config.vocab_size = len(tokenizer)

    config = official["BagelConfig"](
        visual_gen=visual_gen,
        visual_und=visual_und,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=_vae_config(official) if visual_gen else None,
        latent_patch_size=args.latent_patch_size,
        max_latent_size=args.max_latent_size,
        vit_max_num_patch_per_side=args.vit_max_num_patch_per_side,
        timestep_shift=args.timestep_shift,
    )
    _log(f"constructing official BAGEL model visual_und={visual_und} visual_gen={visual_gen}")
    with no_init_weights(), init_empty_weights():
        language_model = official["Qwen2ForCausalLM"](llm_config)
        vit_model = official["SiglipVisionModel"](vit_config) if visual_und else None
        model = official["Bagel"](language_model, vit_model, config)
        if visual_und:
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    state = _load_state(
        args.model_root / "ema.safetensors",
        device=device,
        dtype=torch_dtype,
        visual_und=visual_und,
        visual_gen=visual_gen,
    )
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    relevant_prefixes = ["language_model."]
    if visual_und:
        relevant_prefixes.extend(["vit_model.", "connector.", "vit_pos_embed."])
    if visual_gen:
        relevant_prefixes.extend(["vae2llm.", "llm2vae.", "time_embedder."])
    unexpected_relevant = [key for key in unexpected if key.startswith(tuple(relevant_prefixes))]
    missing_relevant = [key for key in missing if key.startswith(tuple(relevant_prefixes))]
    if unexpected_relevant:
        raise RuntimeError(f"Unexpected keys while loading official BAGEL: {unexpected_relevant[:20]}")
    if missing_relevant:
        raise RuntimeError(f"Missing keys while loading official BAGEL: {missing_relevant[:20]}")

    model.to(device=device)
    model.train()
    return model, tokenizer, device


def _causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


def _text_ids(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text), device=device, dtype=torch.long)


def _base_text_fields(token_ids: torch.Tensor, *, start_index: int = 0) -> dict[str, torch.Tensor]:
    length = int(token_ids.numel())
    return {
        "packed_text_ids": token_ids,
        "packed_text_indexes": torch.arange(
            start_index, start_index + length, device=token_ids.device, dtype=torch.long
        ),
        "packed_position_ids": torch.arange(
            start_index, start_index + length, device=token_ids.device, dtype=torch.long
        ),
    }


def _apply_ce(batch: dict[str, Any], text_indexes: torch.Tensor, token_ids: torch.Tensor) -> None:
    ce_loss_indexes = torch.zeros(int(batch["sequence_length"]), device=token_ids.device, dtype=torch.bool)
    ce_loss_indexes[text_indexes[:-1]] = True
    batch["ce_loss_indexes"] = ce_loss_indexes
    batch["packed_label_ids"] = token_ids[1:].clone()


def _prepare_text_ce_batch(token_ids: torch.Tensor) -> dict[str, Any]:
    fields = _base_text_fields(token_ids)
    length = int(token_ids.numel())
    batch: dict[str, Any] = {
        "sequence_length": length,
        **fields,
        "sample_lens": [length],
        "nested_attention_masks": [_causal_attention_mask(length, token_ids.device)],
    }
    _apply_ce(batch, fields["packed_text_indexes"], token_ids)
    return batch


def _prepare_text_image_ce_batch(token_ids: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    device = token_ids.device
    dtype = _resolve_dtype(args.dtype)
    vit_tokens = torch.linspace(
        -1.0,
        1.0,
        steps=args.vit_tokens * args.vit_patch_dim,
        device=device,
        dtype=dtype,
    ).reshape(args.vit_tokens, args.vit_patch_dim)
    vit_indexes = torch.arange(args.vit_tokens, device=device, dtype=torch.long)
    text_fields = _base_text_fields(token_ids, start_index=args.vit_tokens)
    sequence_length = args.vit_tokens + int(token_ids.numel())
    batch: dict[str, Any] = {
        "sequence_length": sequence_length,
        **text_fields,
        "sample_lens": [sequence_length],
        "packed_position_ids": torch.cat(
            [
                torch.zeros(args.vit_tokens, device=device, dtype=torch.long),
                torch.arange(1, int(token_ids.numel()) + 1, device=device, dtype=torch.long),
            ]
        ),
        "nested_attention_masks": [_causal_attention_mask(sequence_length, device)],
        "packed_vit_tokens": vit_tokens,
        "packed_vit_token_indexes": vit_indexes,
        "packed_vit_position_ids": torch.arange(args.vit_tokens, device=device, dtype=torch.long),
        "vit_token_seqlens": torch.tensor([args.vit_tokens], device=device, dtype=torch.int32),
    }
    _apply_ce(batch, text_fields["packed_text_indexes"], token_ids)
    return batch


def _latent_position_ids(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    h, w = args.latent_grid
    rows = torch.arange(h, device=device, dtype=torch.long)[:, None] * args.max_latent_size
    cols = torch.arange(w, device=device, dtype=torch.long)[None]
    return (rows + cols).flatten()


def _prepare_gen_fields(token_ids: torch.Tensor, args: argparse.Namespace) -> tuple[dict[str, Any], torch.Tensor]:
    device = token_ids.device
    dtype = _resolve_dtype(args.dtype)
    h, w = args.latent_grid
    latent_channels = args.latent_channels
    p = args.latent_patch_size
    latent = torch.linspace(
        -0.75,
        0.75,
        steps=latent_channels * h * p * w * p,
        device=device,
        dtype=dtype,
    ).reshape(1, latent_channels, h * p, w * p)
    num_vae_tokens = h * w
    text_fields = _base_text_fields(token_ids)
    vae_indexes = torch.arange(int(token_ids.numel()), int(token_ids.numel()) + num_vae_tokens, device=device)
    sequence_length = int(token_ids.numel()) + num_vae_tokens
    batch: dict[str, Any] = {
        "sequence_length": sequence_length,
        **text_fields,
        "sample_lens": [sequence_length],
        "packed_position_ids": torch.cat(
            [
                torch.arange(int(token_ids.numel()), device=device, dtype=torch.long),
                torch.full((num_vae_tokens,), int(token_ids.numel()), device=device, dtype=torch.long),
            ]
        ),
        "nested_attention_masks": [_causal_attention_mask(sequence_length, device)],
        "padded_latent": latent,
        "patchified_vae_latent_shapes": [(h, w)],
        "packed_latent_position_ids": _latent_position_ids(args, device),
        "packed_vae_token_indexes": vae_indexes.to(dtype=torch.long),
        "packed_timesteps": torch.linspace(-0.5, 0.5, steps=num_vae_tokens, device=device),
        "mse_loss_indexes": torch.zeros(sequence_length, device=device, dtype=torch.bool),
    }
    batch["mse_loss_indexes"][batch["packed_vae_token_indexes"]] = True
    patch_dim = latent_channels * p * p
    fixed_noise = torch.linspace(-0.25, 0.25, steps=num_vae_tokens * patch_dim, device=device, dtype=dtype).reshape(
        num_vae_tokens, patch_dim
    )
    return batch, fixed_noise


def _prepare_mse_batch(token_ids: torch.Tensor, args: argparse.Namespace) -> tuple[dict[str, Any], torch.Tensor]:
    return _prepare_gen_fields(token_ids, args)


def _prepare_ce_mse_batch(token_ids: torch.Tensor, args: argparse.Namespace) -> tuple[dict[str, Any], torch.Tensor]:
    batch, fixed_noise = _prepare_gen_fields(token_ids, args)
    _apply_ce(batch, batch["packed_text_indexes"], token_ids)
    return batch, fixed_noise


def _cpu_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: [item.detach().cpu() for item in value]
        if isinstance(value, list) and all(torch.is_tensor(item) for item in value)
        else value.detach().cpu()
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


@contextmanager
def _patched_randn_like(fixed_noise: torch.Tensor | None) -> Iterator[None]:
    if fixed_noise is None:
        yield
        return
    orig = torch.randn_like

    def fake_randn_like(input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return fixed_noise.to(device=input_tensor.device, dtype=input_tensor.dtype)

    torch.randn_like = fake_randn_like
    try:
        yield
    finally:
        torch.randn_like = orig


def _param(model: torch.nn.Module, name: str) -> torch.nn.Parameter:
    params = dict(model.named_parameters())
    try:
        return params[name]
    except KeyError as exc:
        raise KeyError(f"Missing parameter {name!r}") from exc


def _sample_grad(param: torch.nn.Parameter, rows: torch.Tensor | None = None) -> torch.Tensor:
    grad = param.grad
    if grad is None:
        raise RuntimeError("Expected gradient, got None.")
    if rows is not None:
        return grad.detach().cpu()[rows.detach().cpu().to(dtype=torch.long)]
    if grad.dim() >= 2:
        return grad.detach().cpu()[:4, :4]
    return grad.detach().cpu()[:16]


def _maybe_grad_entry(
    model: torch.nn.Module,
    *,
    name: str,
    official_param: str,
    v2_module: str,
    v2_param: str,
    rows: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    param = dict(model.named_parameters()).get(official_param)
    if param is None:
        return None
    if param.grad is None:
        return None
    entry: dict[str, Any] = {
        "official_param": official_param,
        "v2_module": v2_module,
        "v2_param": v2_param,
        "grad": _sample_grad(param, rows),
    }
    if rows is not None:
        entry["rows"] = rows.detach().cpu().to(dtype=torch.long)
    return {name: entry}


def _capture_gradients(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    gradients: dict[str, Any] = {}
    text_ids = batch.get("packed_text_ids")
    label_ids = batch.get("packed_label_ids")
    last_layer = model.config.llm_config.num_hidden_layers - 1
    entries = [
        _maybe_grad_entry(
            model,
            name="early_q_proj",
            official_param="language_model.model.layers.0.self_attn.q_proj.weight",
            v2_module="qwen2_mot",
            v2_param="model.layers.0.self_attn.q_proj.weight",
        ),
        _maybe_grad_entry(
            model,
            name="late_mlp_down",
            official_param=f"language_model.model.layers.{last_layer}.mlp.down_proj.weight",
            v2_module="qwen2_mot",
            v2_param=f"model.layers.{last_layer}.mlp.down_proj.weight",
        ),
        _maybe_grad_entry(
            model,
            name="gen_q_proj",
            official_param="language_model.model.layers.0.self_attn.q_proj_moe_gen.weight",
            v2_module="qwen2_mot",
            v2_param="model.layers.0.self_attn.q_proj_moe_gen.weight",
        ),
        _maybe_grad_entry(
            model,
            name="gen_mlp_down",
            official_param=f"language_model.model.layers.{last_layer}.mlp_moe_gen.down_proj.weight",
            v2_module="qwen2_mot",
            v2_param=f"model.layers.{last_layer}.mlp_moe_gen.down_proj.weight",
        ),
    ]
    if torch.is_tensor(text_ids) and torch.is_tensor(label_ids):
        rows = torch.unique(text_ids.detach().cpu()).to(dtype=torch.long)
        entries.append(
            _maybe_grad_entry(
                model,
                name="text_embed_rows",
                official_param="language_model.model.embed_tokens.weight",
                v2_module="text_encoder",
                v2_param="embed_tokens.weight",
                rows=rows,
            )
        )
    if torch.is_tensor(label_ids):
        rows = torch.unique(label_ids.detach().cpu()).to(dtype=torch.long)
        entries.append(
            _maybe_grad_entry(
                model,
                name="lm_head_rows",
                official_param="language_model.lm_head.weight",
                v2_module="text_encoder",
                v2_param="lm_head.weight",
                rows=rows,
            )
        )
    entries.extend(
        [
            _maybe_grad_entry(
                model,
                name="siglip_patch_embed",
                official_param="vit_model.vision_model.embeddings.patch_embedding.weight",
                v2_module="siglip_navit",
                v2_param="vision_model.embeddings.patch_embedding.weight",
            ),
            _maybe_grad_entry(
                model,
                name="visual_connector_fc1",
                official_param="connector.fc1.weight",
                v2_module="siglip_navit",
                v2_param="connector.fc1.weight",
            ),
            _maybe_grad_entry(
                model,
                name="vae2llm",
                official_param="vae2llm.weight",
                v2_module="flow_connector",
                v2_param="vae2llm.weight",
            ),
            _maybe_grad_entry(
                model,
                name="llm2vae",
                official_param="llm2vae.weight",
                v2_module="flow_connector",
                v2_param="llm2vae.weight",
            ),
        ]
    )
    for entry in entries:
        if entry is not None:
            gradients.update(entry)
    return gradients


def _case_settings(case_id: str) -> tuple[bool, bool]:
    if case_id == "gradient_ce_only":
        return False, False
    if case_id == "gradient_text_image_ce":
        return True, False
    if case_id in {"gradient_mse_only", "gradient_ce_mse"}:
        return False, True
    raise ValueError(f"Unsupported gradient case: {case_id}")


def _prepare_case(
    args: argparse.Namespace, tokenizer: Any, device: torch.device
) -> tuple[dict[str, Any], torch.Tensor | None]:
    token_ids = _text_ids(tokenizer, args.text, device)
    if args.case == "gradient_ce_only":
        return _prepare_text_ce_batch(token_ids), None
    if args.case == "gradient_text_image_ce":
        return _prepare_text_image_ce_batch(token_ids, args), None
    if args.case == "gradient_mse_only":
        return _prepare_mse_batch(token_ids, args)
    if args.case == "gradient_ce_mse":
        return _prepare_ce_mse_batch(token_ids, args)
    raise ValueError(f"Unsupported gradient case: {args.case}")


def capture(args: argparse.Namespace) -> dict[str, Any]:
    _configure_determinism(args.seed)

    visual_und, visual_gen = _case_settings(args.case)
    model, tokenizer, device = _build_model(args, visual_und=visual_und, visual_gen=visual_gen)
    batch, fixed_noise = _prepare_case(args, tokenizer, device)

    model.zero_grad(set_to_none=True)
    autocast_context = (
        torch.amp.autocast("cuda", enabled=True, dtype=_resolve_dtype(args.dtype))
        if device.type == "cuda" and args.dtype != "fp32"
        else nullcontext()
    )
    with autocast_context, _patched_randn_like(fixed_noise):
        output = model(**batch)
    ce = output.get("ce")
    mse = output.get("mse")
    loss = None
    if ce is not None:
        loss = ce.mean()
    if mse is not None:
        mse_loss = mse.mean()
        loss = mse_loss if loss is None else loss + mse_loss
    if loss is None:
        raise RuntimeError(f"Case {args.case} produced no training loss.")
    loss.backward()

    prepared = _cpu_batch(batch)
    if fixed_noise is not None:
        prepared["fixed_noise"] = fixed_noise.detach().cpu()
        shifted_timesteps = torch.sigmoid(batch["packed_timesteps"])
        shifted_timesteps = (
            args.timestep_shift * shifted_timesteps / (1 + (args.timestep_shift - 1) * shifted_timesteps)
        )
        prepared["shifted_timesteps"] = shifted_timesteps.detach().cpu()

    return {
        "metadata": {
            "schema_version": 1,
            "case_id": args.case,
            "boundary": "official.Bagel.forward.backward",
            "dtype": args.dtype,
            "seed": args.seed,
            "official_repo": str(args.official_repo),
            "official_checkpoint": str(args.model_root),
            "device": str(device),
        },
        "raw_input": {
            "text": args.text,
            "visual_und": visual_und,
            "visual_gen": visual_gen,
        },
        "prepared": prepared,
        "losses": {
            "ce_vector": None if ce is None else ce.detach().cpu(),
            "mse_tensor": None if mse is None else mse.detach().cpu(),
            "total": loss.detach().cpu(),
        },
        "gradients": _capture_gradients(model, batch),
        "tolerances": {
            "bf16": {
                "max_abs_diff": 3e-3,
                "mean_abs_diff": 3e-4,
                "cosine_similarity_min": 0.997,
                "relative_l2_max": 0.08,
                "loss_max_abs_diff": 1e-2,
                "near_zero_norm": 1e-4,
            },
            "fp32": {
                "max_abs_diff": 1e-5,
                "mean_abs_diff": 1e-6,
                "cosine_similarity_min": 0.99999,
                "relative_l2_max": 1e-4,
                "loss_max_abs_diff": 1e-5,
                "near_zero_norm": 1e-8,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--case",
        choices=("gradient_ce_only", "gradient_text_image_ce", "gradient_mse_only", "gradient_ce_mse"),
        default="gradient_ce_only",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vit-tokens", type=int, default=2)
    parser.add_argument("--vit-patch-dim", type=int, default=3 * 14 * 14)
    parser.add_argument("--vit-max-num-patch-per-side", type=int, default=70)
    parser.add_argument("--latent-grid", type=int, nargs=2, default=(2, 2))
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--latent-patch-size", type=int, default=2)
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
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
