"""Replay the DeepSeek V4 LM head on an official terminal hidden tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    trace = torch.load(args.trace, map_location="cpu", weights_only=True)
    hidden_states = trace["terminal"]["normalized"].cuda()
    input_ids = trace["input_ids"].cuda()
    reference = trace["logprobs"].float().cuda()

    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    weight_key = "head.weight"
    with safe_open(args.checkpoint / index["weight_map"][weight_key], framework="pt", device="cpu") as checkpoint:
        weight = checkpoint.get_tensor(weight_key).cuda()
    weight_fp32 = weight.float()

    pieces = []
    for start in range(0, hidden_states.shape[1] - 1, args.chunk_size):
        end = min(hidden_states.shape[1] - 1, start + args.chunk_size)
        logits = F.linear(hidden_states[:, start:end].float(), weight_fp32)
        targets = input_ids[:, start + 1 : end + 1]
        pieces.append(logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1))
    replay = torch.cat(pieces, dim=1)
    diff = (replay - reference).abs()
    print(
        json.dumps(
            {
                "shape": list(replay.shape),
                "mean_abs_diff": float(diff.mean()),
                "rms_diff": float(diff.square().mean().sqrt()),
                "max_abs_diff": float(diff.max()),
                "p99_abs_diff": float(diff.quantile(0.99)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
