# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert an upstream LLaDA checkpoint (OLMo-style ``LLaDAModelLM``) to the
Llama key layout that VeOmni's ``llada`` model reuses.

LLaDA-8B (``GSAI-ML/LLaDA-8B-Base``) is architecturally identical to Llama
(RMSNorm pre-norm, separate Q/K/V, MHA + RoPE ``rotate_half``, SwiGLU, no bias,
no weight tying), but ships OLMo-style weight names (``transformer.blocks.*``).
Because the RoPE convention (contiguous-half ``rotate_half`` + ``cat(freqs,freqs)``)
and the RMSNorm formulation (``weight * x_normed``) match Llama exactly, the
conversion is a **pure key rename** — no weight permutation or value transform.

Usage:
    python scripts/convert_llada_to_veomni.py \\
        --src /path/to/LLaDA-8B-Base \\
        --dst /path/to/LLaDA-8B-Base-veomni

    # Validate the key mapping without any checkpoint (no weights needed):
    python scripts/convert_llada_to_veomni.py --self-test
"""

import argparse
import json
import os
import re
import shutil
from typing import Optional


# Per-layer block keys. ``transformer.blocks.{i}.<suffix>`` -> ``model.layers.{i}.<llama suffix>``.
_BLOCK_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ff_norm.weight": "post_attention_layernorm.weight",
    "q_proj.weight": "self_attn.q_proj.weight",
    "k_proj.weight": "self_attn.k_proj.weight",
    "v_proj.weight": "self_attn.v_proj.weight",
    "attn_out.weight": "self_attn.o_proj.weight",
    "ff_proj.weight": "mlp.gate_proj.weight",
    "up_proj.weight": "mlp.up_proj.weight",
    "ff_out.weight": "mlp.down_proj.weight",
}

# Top-level (non-block) keys.
_TOP_LEVEL_MAP = {
    "transformer.wte.weight": "model.embed_tokens.weight",
    "transformer.ln_f.weight": "model.norm.weight",
    "transformer.ff_out.weight": "lm_head.weight",
}

_BLOCK_RE = re.compile(r"^transformer\.blocks\.(\d+)\.(.+)$")


def convert_key(key: str) -> Optional[str]:
    """Map a single LLaDA weight key to its Llama equivalent.

    Returns ``None`` for keys that should be dropped (non-persistent buffers /
    caches that Llama recomputes at runtime, e.g. rotary inv_freq).
    """
    # Some HF exports wrap the OLMo stack under a top-level ``model.`` (the
    # ``LLaDAModelLM`` wrapper holds ``self.model = LLaDAModel(...)``), so keys
    # look like ``model.transformer.blocks.{i}.*``. Strip it so the mapping works
    # for both the wrapped and the bare ``transformer.*`` layouts.
    if key.startswith("model."):
        key = key[len("model.") :]

    # Non-persistent buffers / caches that Llama recomputes (rotary, masks).
    # Checked first because some live under a ``transformer.blocks.{i}.`` prefix.
    if "rotary" in key or key.endswith(("inv_freq", "rope_pos_sin", "rope_pos_cos", "causal_attention_bias")):
        return None

    if key in _TOP_LEVEL_MAP:
        return _TOP_LEVEL_MAP[key]

    m = _BLOCK_RE.match(key)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        if suffix in _BLOCK_SUFFIX_MAP:
            return f"model.layers.{layer_idx}.{_BLOCK_SUFFIX_MAP[suffix]}"
        raise KeyError(f"Unknown per-block weight suffix: {suffix!r} (full key: {key!r})")

    raise KeyError(f"Unrecognized LLaDA weight key: {key!r}")


def convert_config(src_cfg: dict) -> dict:
    """Build a Llama-style ``config.json`` dict (model_type='llada')."""
    d_model = src_cfg["d_model"]
    n_heads = src_cfg["n_heads"]
    n_kv_heads = src_cfg.get("n_kv_heads") or n_heads

    out = {
        "model_type": "llada",
        "architectures": ["LLaDAForCausalLM"],
        "hidden_size": d_model,
        "intermediate_size": src_cfg["mlp_hidden_size"],
        "num_hidden_layers": src_cfg["n_layers"],
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": d_model // n_heads,
        "hidden_act": "silu",
        "vocab_size": src_cfg.get("embedding_size") or src_cfg["vocab_size"],
        "max_position_embeddings": src_cfg.get("max_sequence_length", 4096),
        "rms_norm_eps": src_cfg.get("rms_norm_eps", 1e-5),
        "rope_theta": src_cfg.get("rope_theta", 500000.0),
        "tie_word_embeddings": bool(src_cfg.get("weight_tying", False)),
        # LLaDA has no biases anywhere (include_bias / include_qkv_bias = False).
        "attention_bias": bool(src_cfg.get("include_bias", False) or src_cfg.get("include_qkv_bias", False)),
        "mlp_bias": bool(src_cfg.get("include_bias", False)),
        "attention_dropout": src_cfg.get("attention_dropout", 0.0),
        # dLLM-specific + special tokens.
        "mask_token_id": src_cfg.get("mask_token_id", 126336),
        "eos_token_id": src_cfg.get("eos_token_id", 126081),
        "pad_token_id": src_cfg.get("pad_token_id", 126081),
        "torch_dtype": "bfloat16",
    }
    if "bos_token_id" in src_cfg and src_cfg["bos_token_id"] is not None:
        out["bos_token_id"] = src_cfg["bos_token_id"]
    return out


def _list_safetensors(src: str) -> list[str]:
    return sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))


def _reject_identical_src_dst(src: str, dst: str) -> None:
    """Refuse a destination that resolves to the source, which the in-place rewrite would destroy."""
    if os.path.realpath(src) == os.path.realpath(dst):
        raise ValueError(f"--dst resolves to --src ({os.path.realpath(src)}); converting in place deletes the source.")


def convert_checkpoint(src: str, dst: str, copy_tokenizer: bool = True) -> None:
    from safetensors.torch import load_file, save_file

    _reject_identical_src_dst(src, dst)
    os.makedirs(dst, exist_ok=True)

    # --- config.json ---
    with open(os.path.join(src, "config.json")) as f:
        src_cfg = json.load(f)
    out_cfg = convert_config(src_cfg)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
    print(f"[config] wrote {dst}/config.json (model_type=llada, {out_cfg['num_hidden_layers']} layers)")

    # --- weights (preserve sharding) ---
    shards = _list_safetensors(src)
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found under {src}")

    new_weight_map: dict[str, str] = {}
    total_params = 0
    for shard in shards:
        sd = load_file(os.path.join(src, shard))
        new_sd = {}
        for key, tensor in sd.items():
            new_key = convert_key(key)
            if new_key is None:
                print(f"[weights]   drop  {key}")
                continue
            new_sd[new_key] = tensor
            new_weight_map[new_key] = shard
            total_params += tensor.numel()
        save_file(new_sd, os.path.join(dst, shard), metadata={"format": "pt"})
        print(f"[weights] {shard}: {len(new_sd)} tensors")

    # --- index.json (only if the source was sharded) ---
    src_index = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(src_index) or len(shards) > 1:
        index = {
            "metadata": {"total_parameters": total_params},
            "weight_map": new_weight_map,
        }
        with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
        print(f"[index] wrote model.safetensors.index.json ({len(new_weight_map)} entries)")

    # --- tokenizer / aux files ---
    if copy_tokenizer:
        for fn in os.listdir(src):
            if fn.startswith(("tokenizer", "special_tokens", "generation_config", "vocab", "merges", "chat_template")):
                shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
                print(f"[aux] copied {fn}")

    print(f"\nDone. Converted checkpoint at: {dst}")


def self_test(n_layers: int = 2) -> None:
    """Validate the key mapping over a synthetic LLaDA state-dict key set."""
    keys = ["transformer.wte.weight", "transformer.ln_f.weight", "transformer.ff_out.weight"]
    for i in range(n_layers):
        for suffix in _BLOCK_SUFFIX_MAP:
            keys.append(f"transformer.blocks.{i}.{suffix}")

    mapped = {}
    for k in keys:
        out = convert_key(k)
        assert out is not None, f"key unexpectedly dropped: {k}"
        assert out not in mapped, f"collision: {k} and {mapped[out]} both map to {out}"
        mapped[out] = k

    # Every Llama target name we expect must be present.
    expected_top = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    assert expected_top <= set(mapped), f"missing top-level targets: {expected_top - set(mapped)}"
    for i in range(n_layers):
        for tgt in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ):
            assert f"model.layers.{i}.{tgt}" in mapped, f"missing layer target: {i}.{tgt}"

    # Dropped buffer example.
    assert convert_key("transformer.blocks.0.rotary_emb.inv_freq") is None

    # The ``model.``-wrapped HF layout (LLaDAModelLM) maps identically.
    assert convert_key("model.transformer.wte.weight") == "model.embed_tokens.weight"
    assert convert_key("model.transformer.blocks.3.q_proj.weight") == "model.layers.3.self_attn.q_proj.weight"

    cfg = convert_config(
        {
            "d_model": 4096,
            "n_heads": 32,
            "n_kv_heads": 32,
            "n_layers": 32,
            "mlp_hidden_size": 12288,
            "vocab_size": 126464,
            "max_sequence_length": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "weight_tying": False,
            "mask_token_id": 126336,
        }
    )
    assert cfg["hidden_size"] == 4096 and cfg["head_dim"] == 128
    assert cfg["model_type"] == "llada" and cfg["mask_token_id"] == 126336
    assert cfg["tie_word_embeddings"] is False and cfg["attention_bias"] is False
    print(f"self-test OK: {len(mapped)} keys mapped for n_layers={n_layers}, config conversion valid.")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert LLaDA (OLMo-style) checkpoint to VeOmni Llama layout.")
    p.add_argument("--src", type=str, default=None, help="Source LLaDA HF checkpoint dir.")
    p.add_argument("--dst", type=str, default=None, help="Output dir for the converted checkpoint.")
    p.add_argument("--no-tokenizer", action="store_true", help="Do not copy tokenizer/aux files.")
    p.add_argument("--self-test", action="store_true", help="Run key-mapping self-test (no checkpoint needed).")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.src or not args.dst:
        p.error("--src and --dst are required (or use --self-test)")
    convert_checkpoint(args.src, args.dst, copy_tokenizer=not args.no_tokenizer)


if __name__ == "__main__":
    main()
