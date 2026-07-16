"""Compare official and VeOmni sparse-attention kernels on identical traced Q/KV inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

from veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import (
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
)
from veomni.ops.kernels.deepseek_v4 import sparse_attn_tilelang


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo / "inference"))
    import kernel as official_kernel  # noqa: PLC0415
    import model as official_model  # noqa: PLC0415

    with (args.repo / "inference" / "config.json").open() as handle:
        config = official_model.ModelArgs(**json.load(handle))

    trace = torch.load(args.trace, map_location="cpu", weights_only=True)
    detail = trace["details"][args.layer]
    device = torch.device("cuda")
    q_b_proj = detail["q_b_proj"].to(device)
    if q_b_proj.shape[-1] % config.head_dim:
        raise ValueError(f"Traced q_b_proj width {q_b_proj.shape[-1]} is not divisible by head_dim {config.head_dim}")
    local_heads = q_b_proj.shape[-1] // config.head_dim
    q_raw = q_b_proj.view(*q_b_proj.shape[:-1], local_heads, config.head_dim)
    q_official_norm_unrotated = q_raw.clone()
    q_official_norm_unrotated *= torch.rsqrt(
        q_official_norm_unrotated.square().mean(-1, keepdim=True) + config.norm_eps
    )
    q_fp32_norm_unrotated = q_raw * torch.rsqrt(q_raw.float().square().mean(-1, keepdim=True) + config.norm_eps).to(
        q_raw.dtype
    )
    q_norm_diff = (q_official_norm_unrotated.float() - q_fp32_norm_unrotated.float()).abs()
    kv_raw = detail["kv_norm"].to(device)

    ratio = config.compress_ratios[args.layer]
    original_seq_len = config.original_seq_len if ratio else 0
    rope_theta = config.compress_rope_theta if ratio else config.rope_theta
    freqs = official_model.precompute_freqs_cis(
        config.rope_head_dim,
        q_raw.shape[1],
        original_seq_len,
        rope_theta,
        config.rope_factor,
        config.beta_fast,
        config.beta_slow,
    ).to(device)
    q_official_rope = q_official_norm_unrotated.clone()
    q_fp32_norm_official_rope = q_fp32_norm_unrotated.clone()
    official_model.apply_rotary_emb(q_official_rope[..., -config.rope_head_dim :], freqs)
    official_model.apply_rotary_emb(q_fp32_norm_official_rope[..., -config.rope_head_dim :], freqs)
    kv_official_rope = kv_raw.clone()
    official_model.apply_rotary_emb(kv_official_rope[..., -config.rope_head_dim :], freqs)

    hf_config = AutoConfig.from_pretrained(args.checkpoint)
    layer_type = "compress" if ratio else "main"
    rotary = DeepseekV4RotaryEmbedding(hf_config).to(device)
    position_ids = torch.arange(q_raw.shape[1], device=device).unsqueeze(0)
    cos, sin = rotary(q_raw, position_ids, layer_type=layer_type)
    cos_fp32, sin_fp32 = rotary(q_raw.float(), position_ids, layer_type=layer_type)
    q_hf_rope_official_norm = apply_rotary_pos_emb(q_official_norm_unrotated.transpose(1, 2), cos, sin).transpose(1, 2)
    q_hf_rope_fp32_norm = apply_rotary_pos_emb(q_fp32_norm_unrotated.transpose(1, 2), cos, sin).transpose(1, 2)
    q_hf_fp32_rope_official_norm = apply_rotary_pos_emb(
        q_official_norm_unrotated.transpose(1, 2), cos_fp32, sin_fp32
    ).transpose(1, 2)
    kv_hf_rope = apply_rotary_pos_emb(kv_raw.unsqueeze(1), cos, sin)[:, 0]
    kv_hf_fp32_rope = apply_rotary_pos_emb(kv_raw.unsqueeze(1), cos_fp32, sin_fp32)[:, 0]

    official_kernel.act_quant(
        kv_official_rope[..., : -config.rope_head_dim],
        64,
        config.scale_fmt,
        torch.float8_e8m0fnu,
        True,
    )
    official_kernel.act_quant(
        kv_hf_rope[..., : -config.rope_head_dim],
        64,
        config.scale_fmt,
        torch.float8_e8m0fnu,
        True,
    )
    official_kernel.act_quant(
        kv_hf_fp32_rope[..., : -config.rope_head_dim],
        64,
        config.scale_fmt,
        torch.float8_e8m0fnu,
        True,
    )

    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    sink_key = f"layers.{args.layer}.attn.attn_sink"
    sink_file = index["weight_map"][sink_key]
    with safe_open(args.checkpoint / sink_file, framework="pt", device="cpu") as checkpoint_file:
        sinks = checkpoint_file.get_tensor(sink_key)[: q_raw.shape[2]].float().to(device)
    topk = trace["attention_topk"][args.layer].int().to(device)
    scale = config.head_dim**-0.5

    official = official_kernel.sparse_attn(q_official_rope, kv_official_rope, sinks, topk, scale)
    veomni_official_norm = sparse_attn_tilelang(
        q_official_rope.contiguous(), kv_official_rope.contiguous(), sinks.contiguous(), topk.contiguous(), scale
    )
    veomni_fp32_norm = sparse_attn_tilelang(
        q_fp32_norm_official_rope.contiguous(),
        kv_official_rope.contiguous(),
        sinks.contiguous(),
        topk.contiguous(),
        scale,
    )
    veomni_hf_rope_official_norm = sparse_attn_tilelang(
        q_hf_rope_official_norm.contiguous(),
        kv_hf_rope.contiguous(),
        sinks.contiguous(),
        topk.contiguous(),
        scale,
    )
    veomni_hf_rope_fp32_norm = sparse_attn_tilelang(
        q_hf_rope_fp32_norm.contiguous(),
        kv_hf_rope.contiguous(),
        sinks.contiguous(),
        topk.contiguous(),
        scale,
    )
    veomni_hf_fp32_rope_official_norm = sparse_attn_tilelang(
        q_hf_fp32_rope_official_norm.contiguous(),
        kv_hf_fp32_rope.contiguous(),
        sinks.contiguous(),
        topk.contiguous(),
        scale,
    )

    def diff_stats(left, right):
        diff = (left.float() - right.float()).abs()
        return {
            "mean_abs_diff": float(diff.mean()),
            "rms_diff": float(diff.square().mean().sqrt()),
            "max_abs_diff": float(diff.max()),
        }

    print(
        {
            "shape": list(official.shape),
            "query_norm": {
                "mean_abs_diff": float(q_norm_diff.mean()),
                "rms_diff": float(q_norm_diff.square().mean().sqrt()),
                "max_abs_diff": float(q_norm_diff.max()),
            },
            "query_rope": diff_stats(q_official_rope, q_hf_rope_official_norm),
            "query_fp32_rope": diff_stats(q_official_rope, q_hf_fp32_rope_official_norm),
            "kv_rope": diff_stats(kv_official_rope, kv_hf_rope),
            "kv_fp32_rope": diff_stats(kv_official_rope, kv_hf_fp32_rope),
            "same_official_norm": diff_stats(official, veomni_official_norm),
            "fp32_norm_vs_official": diff_stats(official, veomni_fp32_norm),
            "hf_rope_official_norm_vs_official": diff_stats(official, veomni_hf_rope_official_norm),
            "hf_rope_fp32_norm_vs_official": diff_stats(official, veomni_hf_rope_fp32_norm),
            "hf_fp32_rope_official_norm_vs_official": diff_stats(official, veomni_hf_fp32_rope_official_norm),
        }
    )


if __name__ == "__main__":
    main()
