"""Capture official BAGEL text-only one-step inference fixtures.

The generated fixture is an oracle artifact for SeedOmni V2 parity work. It may
contain official ``packed_*`` field names, but those names must stay on the
capture/test side and must not become V2 runtime module inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights


pytestmark = pytest.mark.skip(reason="BAGEL official capture helper; run explicitly to generate parity fixtures.")

DEFAULT_OUTPUT = Path("outputs/bagel_v2/parity/text_only_one_step_logits.pt")
DEFAULT_PROMPT = "Describe BAGEL in one short sentence."


def _log(message: str) -> None:
    print(f"[bagel-text-fixture] {message}", flush=True)


def _import_official(official_repo: Path) -> dict[str, Any]:
    _log(f"importing official BAGEL from {official_repo}")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    sys.path.insert(0, str(official_repo))
    from data.data_utils import add_special_tokens
    from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel.qwen2_navit import NaiveCache
    from modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    return {
        "add_special_tokens": add_special_tokens,
        "Bagel": Bagel,
        "BagelConfig": BagelConfig,
        "NaiveCache": NaiveCache,
        "Qwen2Config": Qwen2Config,
        "Qwen2ForCausalLM": Qwen2ForCausalLM,
        "Qwen2Tokenizer": Qwen2Tokenizer,
    }


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


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


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


def _load_language_state(path: Path, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    _log(f"loading language weights from {path} to {device} as {dtype}")
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("language_model."):
                state_dict[key] = f.get_tensor(key).to(device=device, dtype=dtype)
    _log(f"loaded {len(state_dict)} language tensors")
    return state_dict


def _build_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, int], torch.device]:
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
        visual_gen=False,
        visual_und=False,
        llm_config=llm_config,
        vit_config=None,
        vae_config=None,
    )
    _log("constructing official text-only BAGEL model on meta parameters")
    with no_init_weights(), init_empty_weights():
        language_model = official["Qwen2ForCausalLM"](llm_config)
        model = official["Bagel"](language_model, None, config)

    text_state = _load_language_state(args.model_root / "ema.safetensors", device=device, dtype=torch_dtype)
    _log("assigning language weights")
    missing, unexpected = model.load_state_dict(text_state, strict=False, assign=True)
    del text_state
    del missing
    unexpected_non_text = [key for key in unexpected if key.startswith("language_model.")]
    if unexpected_non_text:
        raise RuntimeError(f"Unexpected text keys while loading official BAGEL: {unexpected_non_text[:20]}")

    _log("moving buffers to target device")
    model.to(device=device)
    model.eval()
    _log("model ready")
    return model, tokenizer, new_token_ids, device


@torch.no_grad()
def capture(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, new_token_ids, device = _build_model(args)
    official = _import_official(args.official_repo)

    past_key_values = official["NaiveCache"](model.config.llm_config.num_hidden_layers)
    curr_kvlens = [0]
    curr_rope = [0]

    prompt_input, kv_lens, ropes = model.prepare_prompts(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        prompts=[args.prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    prompt_input = _move_tensors(prompt_input, device)
    past_key_values = model.forward_cache_update_text(past_key_values, **prompt_input)

    cache_after_prefill = _cache_to_tensors(past_key_values)

    start_input = model.prepare_start_tokens(kv_lens, ropes, new_token_ids)
    start_input = _move_tensors(start_input, device)

    curr_tokens = start_input["packed_start_tokens"]
    key_values_lens = start_input["key_values_lens"]
    packed_key_value_indexes = start_input["packed_key_value_indexes"]
    packed_query_position_ids = start_input["packed_query_position_ids"]

    packed_text_embedding = model.language_model.model.embed_tokens(curr_tokens)
    query_lens = torch.ones_like(curr_tokens)
    packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
        0,
        len(key_values_lens),
        device=key_values_lens.device,
        dtype=key_values_lens.dtype,
    )

    unpacked_key_value_indexes = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
    for idx in range(len(unpacked_key_value_indexes)):
        unpacked_key_value_indexes[idx] += idx
    packed_key_value_indexes_for_step = torch.cat(unpacked_key_value_indexes, dim=0)

    extra_inputs = {"mode": "und"} if model.use_moe else {}
    output = model.language_model.forward_inference(
        packed_query_sequence=packed_text_embedding,
        query_lens=query_lens,
        packed_query_position_ids=packed_query_position_ids,
        packed_query_indexes=packed_query_indexes,
        past_key_values=past_key_values,
        key_values_lens=key_values_lens,
        packed_key_value_indexes=packed_key_value_indexes_for_step,
        update_past_key_values=True,
        is_causal=True,
        **extra_inputs,
    )
    logits = model.language_model.lm_head(output.packed_query_sequence)
    greedy_token = torch.argmax(logits, dim=-1)

    return {
        "metadata": {
            "schema_version": 1,
            "case_id": "text_only_one_step_logits",
            "boundary": "official.prepare_prompts.forward_cache_update_text.prepare_start_tokens.one_step_logits",
            "dtype": args.dtype,
            "seed": args.seed,
            "official_repo": str(args.official_repo),
            "official_checkpoint": str(args.model_root),
            "device": str(device),
        },
        "raw_input": {
            "prompt": args.prompt,
            "do_sample": False,
            "temperature": 1.0,
            "max_new_tokens": 1,
        },
        "rng_state": _rng_state(),
        "tokenizer": {
            "new_token_ids": dict(new_token_ids),
            "encoded_prompt_ids": tokenizer.encode(args.prompt),
        },
        "prepared": {
            "prompt": _cpu_tensors(prompt_input),
            "kv_lens_after_prompt": list(kv_lens),
            "ropes_after_prompt": list(ropes),
            "start": _cpu_tensors(start_input),
            "packed_query_indexes": packed_query_indexes.detach().cpu(),
            "packed_key_value_indexes_for_step": packed_key_value_indexes_for_step.detach().cpu(),
            "query_lens": query_lens.detach().cpu(),
        },
        "cache_after_prefill": cache_after_prefill,
        "one_step": {
            "hidden_state": output.packed_query_sequence.detach().cpu(),
            "logits": logits.detach().cpu(),
            "greedy_token": greedy_token.detach().cpu(),
            "cache_after_step": _cache_to_tensors(output.past_key_values),
        },
        "tolerances": {
            "bf16": {
                "v2_parity": {
                    "max_abs_diff": 1.0e-2,
                    "mean_abs_diff": 1.0e-4,
                    "cosine_similarity_min": 0.9999,
                    "source": "agreed V2-vs-official bf16 text parity gate",
                },
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
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
