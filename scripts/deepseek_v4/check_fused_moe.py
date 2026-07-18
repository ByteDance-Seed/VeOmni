"""Compare VeOmni's Triton fused MoE against an eager actual-weight reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from veomni.models.transformers.deepseek_v4.checkpoint_tensor_converter import _dequantize_scaled_weight
from veomni.ops.kernels.moe.group_gemm import MergedFc1TritonFusedMoeExpertFunction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    return parser.parse_args()


def load_tensor(checkpoint: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(checkpoint / weight_map[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def load_expert_weight(
    checkpoint: Path,
    weight_map: dict[str, str],
    layer: int,
    expert: int,
    projection: str,
) -> torch.Tensor:
    prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}"
    weight = load_tensor(checkpoint, weight_map, f"{prefix}.weight").cuda()
    scale = load_tensor(checkpoint, weight_map, f"{prefix}.scale").cuda()
    return _dequantize_scaled_weight(weight, scale, packed_fp4=True).to(torch.bfloat16)


def eager_reference(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    for expert in range(gate_up.shape[0]):
        token, topk_position = torch.where(selected_experts == expert)
        if token.numel() == 0:
            continue
        projected = F.linear(hidden_states[token], gate_up[expert])
        gate, up = projected.chunk(2, dim=-1)
        activated = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        activated *= routing_weights[token, topk_position, None]
        output.index_add_(0, token, F.linear(activated, down[expert]))
    return output


def main() -> None:
    args = parse_args()
    with (args.checkpoint / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]

    gate, up, down = [], [], []
    for expert in range(args.experts):
        gate.append(load_expert_weight(args.checkpoint, weight_map, args.layer, expert, "w1"))
        up.append(load_expert_weight(args.checkpoint, weight_map, args.layer, expert, "w3"))
        down.append(load_expert_weight(args.checkpoint, weight_map, args.layer, expert, "w2"))
    gate_up = torch.cat((torch.stack(gate), torch.stack(up)), dim=1)
    down_weight = torch.stack(down)

    torch.manual_seed(20260715)
    hidden = torch.randn(args.tokens, gate_up.shape[-1], device="cuda", dtype=torch.bfloat16)
    selected = torch.randint(args.experts, (args.tokens, args.top_k), device="cuda")
    routing = torch.rand(args.tokens, args.top_k, device="cuda", dtype=torch.bfloat16)
    routing /= routing.sum(dim=-1, keepdim=True)
    limit = 10.0

    with torch.no_grad():
        expected = eager_reference(hidden, selected, routing, gate_up, down_weight, limit)
        actual = MergedFc1TritonFusedMoeExpertFunction.apply(
            args.experts,
            routing,
            selected,
            hidden,
            gate_up,
            down_weight,
            limit,
        )
    diff = (actual.float() - expected.float()).abs()
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    print(
        json.dumps(
            {
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "p99_abs_diff": float(diff.quantile(0.99)),
                "reference_rms": float(expected.float().square().mean().sqrt()),
                "cosine_similarity": float(cosine),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
