"""Two-rank CPU worker for the HF rank-0 broadcast cache-advice test."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from veomni.models.checkpoint import weights


def main():
    checkpoint, output = map(Path, sys.argv[1:])
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        if os.getenv("VEOMNI_TEST_INVALID_CACHE_RANK") == str(rank):
            os.environ["VEOMNI_HF_DROP_READ_CACHE"] = "invalid"
        weights.get_parallel_state = lambda: SimpleNamespace(global_rank=rank)
        # This test exercises real Gloo collectives with CPU parameters even
        # when its host also has CUDA/NPU devices.
        weights._get_communication_device = lambda device: torch.device("cpu")
        original_advice = os.posix_fadvise
        calls = []

        def track_advice(fd, *args):
            path = os.path.realpath(os.readlink(f"/proc/self/fd/{fd}"))
            result = original_advice(fd, *args)
            calls.append(path)
            return result

        os.posix_fadvise = track_advice
        model = torch.nn.Linear(4, 4)
        model.config = SimpleNamespace()
        weights.rank0_load_and_broadcast_weights(model, str(checkpoint), init_device="cpu")
        equal = bool(torch.all(model.weight == 1.5) and torch.all(model.bias == 0.25))
        rows = [None] * dist.get_world_size()
        dist.all_gather_object(rows, {"rank": rank, "equal": equal, "paths": calls})
        if rank == 0:
            output.write_text(json.dumps(rows, sort_keys=True) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
