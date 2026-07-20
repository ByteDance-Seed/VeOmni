#!/usr/bin/env python3
"""Small all-GPU NCCL collective test, intended to run under torchrun."""

from __future__ import annotations

import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    expected_gpus = int(os.getenv("EXPECTED_GPU_COUNT", "8"))

    assert world_size == expected_gpus, f"expected world size {expected_gpus}, got {world_size}"
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)

    try:
        probe = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(probe)
        expected_sum = world_size * (world_size + 1) / 2
        assert probe.item() == expected_sum, (probe.item(), expected_sum)

        element_count = int(os.getenv("NCCL_SMOKE_NUMEL", str(8 * 1024 * 1024)))
        payload = torch.ones(element_count, device=device, dtype=torch.float32)

        dist.all_reduce(payload)
        torch.cuda.synchronize()
        payload.fill_(1)
        dist.barrier()

        started = time.perf_counter()
        dist.all_reduce(payload)
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - started, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

        assert torch.all(payload == world_size)
        if rank == 0:
            size_mib = payload.numel() * payload.element_size() / (1024**2)
            print(
                f"NCCL all-reduce passed on {world_size} GPUs: "
                f"{size_mib:.1f} MiB per rank, max elapsed {elapsed.item():.4f}s"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
