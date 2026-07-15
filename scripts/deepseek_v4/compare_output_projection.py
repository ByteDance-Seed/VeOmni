"""Replay the official tensor-parallel output projection on identical inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def load_tensor(checkpoint: Path, filename: str, key: str, device: torch.device) -> torch.Tensor:
    with safe_open(checkpoint / filename, framework="pt", device="cpu") as checkpoint_file:
        return checkpoint_file.get_tensor(key).to(device)


def diff_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    diff = (left.float() - right.float()).abs()
    return {
        "mean_abs_diff": float(diff.mean()),
        "rms_diff": float(diff.square().mean().sqrt()),
        "max_abs_diff": float(diff.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    if args.world_size < 1:
        raise ValueError("--world-size must be positive")

    sys.path.insert(0, str(args.repo / "inference"))
    import kernel as official_kernel  # noqa: PLC0415

    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda")
    trace = torch.load(args.trace, map_location="cpu", weights_only=True)
    detail = trace["details"][args.layer]
    x = detail["o_a_proj_full"].to(device).flatten(2)
    reference = detail["o_b_proj"].to(device)
    if x.shape[-1] % args.world_size:
        raise ValueError(f"projection width {x.shape[-1]} is not divisible by world size {args.world_size}")
    rank_width = x.shape[-1] // args.world_size
    scale_dtype = torch.float8_e8m0fnu

    partials = []
    for rank in range(args.world_size):
        checkpoint_file = f"model{rank}-mp{args.world_size}.safetensors"
        weight_key = f"layers.{args.layer}.attn.wo_b.weight"
        scale_key = f"layers.{args.layer}.attn.wo_b.scale"
        weight = load_tensor(args.official_checkpoint, checkpoint_file, weight_key, device).contiguous()
        weight_scale = load_tensor(args.official_checkpoint, checkpoint_file, scale_key, device).contiguous()
        local_x = x[..., rank * rank_width : (rank + 1) * rank_width].contiguous()
        quantized_x, input_scale = official_kernel.act_quant(local_x, 128, "ue8m0", scale_dtype)
        partials.append(official_kernel.fp8_gemm(quantized_x, input_scale, weight, weight_scale, scale_dtype))

    partitioned_fp32_sum = torch.stack([partial.float() for partial in partials]).sum(0).to(x.dtype)
    partitioned_bf16_sum = torch.stack(partials).sum(0)

    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    full_weight_key = f"layers.{args.layer}.attn.wo_b.weight"
    full_scale_key = f"layers.{args.layer}.attn.wo_b.scale"
    full_weight_file = index["weight_map"][full_weight_key]
    full_scale_file = index["weight_map"][full_scale_key]
    full_weight = load_tensor(args.checkpoint, full_weight_file, full_weight_key, device).contiguous()
    full_weight_scale = load_tensor(args.checkpoint, full_scale_file, full_scale_key, device).contiguous()
    quantized_x, input_scale = official_kernel.act_quant(x.contiguous(), 128, "ue8m0", scale_dtype)
    full_gemm = official_kernel.fp8_gemm(quantized_x, input_scale, full_weight, full_weight_scale, scale_dtype)

    print(
        json.dumps(
            {
                "shape": list(reference.shape),
                "partitioned_fp32_sum_vs_official": diff_stats(partitioned_fp32_sum, reference),
                "partitioned_bf16_sum_vs_official": diff_stats(partitioned_bf16_sum, reference),
                "full_gemm_vs_official": diff_stats(full_gemm, reference),
                "full_gemm_vs_partitioned_fp32_sum": diff_stats(full_gemm, partitioned_fp32_sum),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
