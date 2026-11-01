"""Two-rank DCP roundtrip and one-rank cache-advice failure worker."""

import errno
import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor, Shard, init_device_mesh


def main():
    root, mode = Path(sys.argv[1]), sys.argv[2]
    source = Path(__file__).resolve().parents[2] / "veomni/checkpoint/bounded_writer.py"
    spec = importlib.util.spec_from_file_location("isolated_bounded_writer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dist.init_process_group("gloo", timeout=timedelta(seconds=45))
    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (dist.get_world_size(),))

    def value(offset):
        local = (torch.arange(16, dtype=torch.float32) + rank * 16 + offset) / 1000
        return DTensor.from_local(local, mesh, [Shard(0)], run_check=False, shape=torch.Size([32]), stride=(1,))

    source = {"master": value(0.0), "exp_avg": value(0.2), "exp_avg_sq": value(0.4)}
    if mode == "fail" and rank == 1:

        def fail(*args):
            raise OSError(errno.EIO, "rank1 injected cache-advice failure")

        module.os.posix_fadvise = fail
        module.os.POSIX_FADV_DONTNEED = getattr(module.os, "POSIX_FADV_DONTNEED", 4)
    error = None
    try:
        dcp.save(source, storage_writer=module.BoundedFileSystemWriter(root, max_bytes=128, release_page_cache=True))
    except BaseException as exc:
        error = repr(exc)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if mode == "fail":
        if any(item is None or "rank1 injected cache-advice failure" not in item for item in errors):
            raise RuntimeError("One-rank writer failure did not reach every rank: " + repr(errors))
        if (root / ".metadata").exists():
            raise RuntimeError("Failed multi-rank save published DCP metadata")
    else:
        if any(item is not None for item in errors):
            raise RuntimeError("Multi-rank save failed: " + repr(errors))
        restored = {
            name: DTensor.from_local(
                torch.zeros_like(tensor.to_local()),
                mesh,
                [Shard(0)],
                run_check=False,
                shape=tensor.shape,
                stride=tensor.stride(),
            )
            for name, tensor in source.items()
        }
        dcp.load(restored, checkpoint_id=root)
        if any(not torch.equal(restored[name].to_local(), tensor.to_local()) for name, tensor in source.items()):
            raise RuntimeError("Multi-rank DCP roundtrip changed a local shard")
    dist.barrier()
    if rank == 0:
        print("BOUNDED_DCP_GLOO_PASS " + mode, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
