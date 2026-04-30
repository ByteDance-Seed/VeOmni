#!/usr/bin/env python3
"""Simple FSDP2 training smoke test on Intel XPU.

This test intentionally uses FSDP2's ``fully_shard`` API (not FSDPv1's
FullyShardedDataParallel class) so it validates the real FSDP2/XCCL path.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.optim as optim
from transformers import AutoModelForCausalLM


try:
    from torch.distributed.fsdp import FSDPModule, fully_shard
except ImportError:
    from torch.distributed.fsdp._fully_shard import FSDPModule, fully_shard


def setup_distributed() -> tuple[int, int, int]:
    dist.init_process_group("xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


def create_model(model_name: str, device: torch.device) -> torch.nn.Module:
    print(f"[Rank {dist.get_rank()}] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # FSDP2 path: shard layers then root module with fully_shard.
    model = model.to(device)
    for layer in model.model.layers:
        fully_shard(layer)
    fully_shard(model)

    # Ensure we are actually testing FSDP2 wrappers.
    fsdp2_count = sum(1 for m in model.modules() if isinstance(m, FSDPModule))
    if fsdp2_count == 0:
        raise RuntimeError("FSDP2 wrapping failed: no FSDPModule instances found")
    if dist.get_rank() == 0:
        print(f"[Rank 0] FSDP2 modules wrapped: {fsdp2_count}")

    return model


def create_dummy_batch(batch_size: int = 2, seq_len: int = 128) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :-1] = labels[:, 1:].clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main(args: argparse.Namespace) -> None:
    rank, world_size, local_rank = setup_distributed()

    if not torch.xpu.is_available():
        raise RuntimeError("XPU is not available")
    torch.xpu.set_device(local_rank)
    device = torch.device(f"xpu:{local_rank}")

    print(f"[Rank {rank}/{world_size}] Device: {device}")

    model = create_model(args.model, device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print(f"[Rank {rank}] Starting {args.steps} training steps...")
    for step in range(1, args.steps + 1):
        batch = create_dummy_batch(batch_size=args.batch_size, seq_len=args.seq_len)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if rank == 0 and step % max(1, args.steps // 5) == 0:
            print(f"Step {step}/{args.steps}: loss={loss.item():.4f}")

    dist.destroy_process_group()
    print(f"[Rank {rank}] Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    main(parser.parse_args())
