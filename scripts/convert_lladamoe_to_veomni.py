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
"""Convert an upstream LLaDA-MoE checkpoint to the Qwen3-MoE layout VeOmni reuses.

``inclusionAI/LLaDA-MoE-7B-A1B-Instruct`` ships a custom ``LLaDAMoEModelLM``
(``model_type="llada"``) whose transformer block is **architecturally identical
to Qwen3-MoE**: qk-layernorm attention + RoPE, a ``mlp.gate`` router over
per-expert SwiGLU experts (``mlp.experts.{j}.{gate,up,down}_proj``), softmax
routing with ``norm_topk_prob=False``, every layer MoE, no shared experts. The
only forward-pass difference is that LLaDA is a **mask predictor** (bidirectional
/ diffusion LM), which the VeOmni ``llada_moe`` modeling handles, NOT the weights.

Crucially the **weight key layout already matches Qwen3-MoE byte-for-byte**
(verified against the safetensors index), so this conversion is a **pure config
rewrite** — no weight rename / transform. We:
  1. emit a Qwen3-MoE-schema ``config.json`` with ``model_type="llada_moe"``
     (a distinct type so it does not collide with the dense LLaDA-8B ``llada``
     family);
  2. symlink the (unchanged) safetensors shards + copy the index / tokenizer.

Usage:
    python scripts/convert_lladamoe_to_veomni.py \\
        --src /path/to/LLaDA-MoE-7B-A1B-Instruct \\
        --dst /path/to/LLaDA-MoE-7B-A1B-Instruct-qwen3moe

    # Validate the config mapping without any checkpoint:
    python scripts/convert_lladamoe_to_veomni.py --self-test
"""

import argparse
import json
import os
import shutil


# Reserved [MASK] token id used by the dLLM forward (masking) process. The
# upstream LLaDA-MoE ``config.json`` does not carry it; the value matches the
# official ``generate`` reference (model card) / the ``<|mdm_mask|>`` token.
_DEFAULT_MASK_TOKEN_ID = 156895

# Source fields the Qwen3-MoE block implements only at these values. A checkpoint that
# sets any of them otherwise has a different architecture and would convert silently wrong.
_REQUIRED_SOURCE_VALUES = {
    "qk_layernorm": True,
    "clip_qkv": None,
    "moe_router_enable_expert_bias": False,
    "moe_router_score_function": "softmax",
    "routed_scaling_factor": 1,
    "partial_rotary_factor": 1,
    "rope_scaling": None,
}


def convert_config(src_cfg: dict) -> dict:
    """Map a LLaDA-MoE ``config.json`` dict to a Qwen3-MoE-schema dict.

    ``model_type='llada_moe'`` + ``architectures=['LLaDAMoEForCausalLM']`` so the
    weights load via VeOmni's LLaDA-MoE modeling
    (``veomni/models/transformers/llada_moe``) — the bidirectional (mask-predictor)
    Qwen3-MoE stack, with the Qwen3-MoE key layout reused byte-for-byte.
    """
    unsupported = {
        key: src_cfg[key]
        for key, expected in _REQUIRED_SOURCE_VALUES.items()
        if key in src_cfg and src_cfg[key] != expected
    }
    if unsupported:
        raise ValueError(f"LLaDA-MoE config fields outside what the Qwen3-MoE block implements: {unsupported}")
    hidden_size = src_cfg["hidden_size"]
    n_heads = src_cfg["num_attention_heads"]
    n_kv_heads = src_cfg.get("num_key_value_heads", n_heads)

    out = {
        "model_type": "llada_moe",
        "architectures": ["LLaDAMoEForCausalLM"],
        "hidden_size": hidden_size,
        # Dense ``intermediate_size`` is unused (all layers are MoE: moe_layer_freq
        # all 1 / decoder_sparse_step=1) but kept for schema completeness.
        "intermediate_size": src_cfg.get("dense_intermediate_size", 4 * hidden_size),
        "moe_intermediate_size": src_cfg["expert_intermediate_size"],
        "num_hidden_layers": src_cfg["num_hidden_layers"],
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": src_cfg.get("head_dim") or hidden_size // n_heads,
        "num_experts": src_cfg["num_experts"],
        "num_experts_per_tok": src_cfg["num_experts_per_tok"],
        # LLaDA-MoE hardcodes norm_topk_prob=False in its SparseMoeBlock.
        "norm_topk_prob": False,
        # All layers are MoE -> sparse every layer, no dense-only layers.
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "hidden_act": src_cfg.get("hidden_act", "silu"),
        "vocab_size": src_cfg["vocab_size"],
        "max_position_embeddings": src_cfg.get("max_position_embeddings", 8192),
        "rms_norm_eps": src_cfg.get("rms_norm_eps", 1e-5),
        "rope_theta": src_cfg.get("rope_theta", 50000),
        "attention_bias": bool(src_cfg.get("attention_bias", False)),
        "attention_dropout": src_cfg.get("attention_dropout", 0.0),
        "tie_word_embeddings": bool(src_cfg.get("tie_word_embeddings", False)),
        "router_aux_loss_coef": src_cfg.get("router_aux_loss_coef", 0.001),
        "output_router_logits": False,
        "sliding_window": None,
        "torch_dtype": src_cfg.get("torch_dtype", "bfloat16"),
        # dLLM-specific + special tokens.
        "mask_token_id": src_cfg.get("mask_token_id", _DEFAULT_MASK_TOKEN_ID),
        "eos_token_id": src_cfg.get("eos_token_id", 156892),
        "pad_token_id": src_cfg.get("pad_token_id", 156892),
    }
    if src_cfg.get("bos_token_id") is not None:
        out["bos_token_id"] = src_cfg["bos_token_id"]
    return out


def _reject_identical_src_dst(src: str, dst: str) -> None:
    """Refuse a destination that resolves to the source, which the symlinking below would destroy."""
    if os.path.realpath(src) == os.path.realpath(dst):
        raise ValueError(f"--dst resolves to --src ({os.path.realpath(src)}); converting in place deletes the source.")


def convert_checkpoint(src: str, dst: str, copy_tokenizer: bool = True) -> None:
    _reject_identical_src_dst(src, dst)
    os.makedirs(dst, exist_ok=True)

    with open(os.path.join(src, "config.json")) as f:
        src_cfg = json.load(f)
    out_cfg = convert_config(src_cfg)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
    print(
        f"[config] wrote {dst}/config.json "
        f"(model_type=llada_moe, {out_cfg['num_hidden_layers']} layers, "
        f"{out_cfg['num_experts']} experts, mask_token_id={out_cfg['mask_token_id']})"
    )

    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found under {src}")
    for shard in shards:
        link = os.path.join(dst, shard)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(os.path.join(src, shard)), link)
    print(f"[weights] symlinked {len(shards)} safetensors shard(s) (no rename)")

    src_index = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(src_index):
        shutil.copy2(src_index, os.path.join(dst, "model.safetensors.index.json"))
        print("[index] copied model.safetensors.index.json")

    if copy_tokenizer:
        for fn in os.listdir(src):
            if fn.startswith(("tokenizer", "special_tokens", "generation_config", "vocab", "merges", "chat_template")):
                shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
                print(f"[aux] copied {fn}")

    print(f"\nDone. Converted checkpoint at: {dst}")


def self_test() -> None:
    """Validate the config mapping over a synthetic LLaDA-MoE config dict."""
    src = {
        "model_type": "llada",
        "architectures": ["LLaDAMoEModel"],
        "hidden_size": 2048,
        "dense_intermediate_size": 8192,
        "expert_intermediate_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "vocab_size": 157184,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 50000,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "router_aux_loss_coef": 0.01,
        "eos_token_id": 156892,
        "pad_token_id": 156892,
    }
    cfg = convert_config(src)
    assert cfg["model_type"] == "llada_moe"
    assert cfg["architectures"] == ["LLaDAMoEForCausalLM"]
    assert cfg["moe_intermediate_size"] == 1024
    assert cfg["num_experts"] == 64 and cfg["num_experts_per_tok"] == 8
    assert cfg["head_dim"] == 128
    assert cfg["norm_topk_prob"] is False
    assert cfg["decoder_sparse_step"] == 1 and cfg["mlp_only_layers"] == []
    assert cfg["rope_theta"] == 50000
    assert cfg["mask_token_id"] == _DEFAULT_MASK_TOKEN_ID
    assert cfg["tie_word_embeddings"] is False
    try:
        convert_config({**src, "moe_router_score_function": "sigmoid"})
    except ValueError:
        pass
    else:
        raise AssertionError("a sigmoid-routed source config must be refused")
    print("self-test OK: LLaDA-MoE -> Qwen3-MoE config mapping valid.")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert LLaDA-MoE checkpoint to VeOmni Qwen3-MoE layout.")
    p.add_argument("--src", type=str, default=None, help="Source LLaDA-MoE HF checkpoint dir.")
    p.add_argument("--dst", type=str, default=None, help="Output dir (Qwen3-MoE layout, model_type=llada_moe).")
    p.add_argument("--no-tokenizer", action="store_true", help="Do not copy tokenizer/aux files.")
    p.add_argument("--self-test", action="store_true", help="Run config-mapping self-test (no checkpoint needed).")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.src or not args.dst:
        p.error("--src and --dst are required (or use --self-test)")
    convert_checkpoint(args.src, args.dst, copy_tokenizer=not args.no_tokenizer)


if __name__ == "__main__":
    main()
