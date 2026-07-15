"""Compare official and VeOmni sparse-attention kernels on identical traced Q/KV inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

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
    q = q_b_proj.view(*q_b_proj.shape[:-1], local_heads, config.head_dim)
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + config.norm_eps)
    kv = detail["kv_norm"].to(device)

    ratio = config.compress_ratios[args.layer]
    original_seq_len = config.original_seq_len if ratio else 0
    rope_theta = config.compress_rope_theta if ratio else config.rope_theta
    freqs = official_model.precompute_freqs_cis(
        config.rope_head_dim,
        q.shape[1],
        original_seq_len,
        rope_theta,
        config.rope_factor,
        config.beta_fast,
        config.beta_slow,
    ).to(device)
    official_model.apply_rotary_emb(q[..., -config.rope_head_dim :], freqs)
    official_model.apply_rotary_emb(kv[..., -config.rope_head_dim :], freqs)
    official_kernel.act_quant(
        kv[..., : -config.rope_head_dim],
        64,
        config.scale_fmt,
        torch.float8_e8m0fnu,
        True,
    )

    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    sink_key = f"layers.{args.layer}.attn.attn_sink"
    sink_file = index["weight_map"][sink_key]
    with safe_open(args.checkpoint / sink_file, framework="pt", device="cpu") as checkpoint_file:
        sinks = checkpoint_file.get_tensor(sink_key)[: q.shape[2]].float().to(device)
    topk = trace["attention_topk"][args.layer].int().to(device)
    scale = config.head_dim**-0.5

    official = official_kernel.sparse_attn(q, kv, sinks, topk, scale)
    veomni = sparse_attn_tilelang(q.contiguous(), kv.contiguous(), sinks.contiguous(), topk.contiguous(), scale)
    diff = (official.float() - veomni.float()).abs()
    print(
        {
            "shape": list(diff.shape),
            "mean_abs_diff": float(diff.mean()),
            "rms_diff": float(diff.square().mean().sqrt()),
            "max_abs_diff": float(diff.max()),
        }
    )


if __name__ == "__main__":
    main()
