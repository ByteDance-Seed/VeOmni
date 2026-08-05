"""CP collectives reproduce the single-rank tensors and carry gradients."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


CP_SIZE = 4
LOCAL_LEN = 16
SEQ_LEN = CP_SIZE * LOCAL_LEN
DIM = 8
COUNTS = [4, 4, 4, 3]


def _run(rank: int, world_size: int, init_file: str) -> None:
    device_type = get_device_type()
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from veomni.distributed.context_parallel.dsa_cp import (
        all_gather_compressed_rows,
        all_gather_kv,
        exchange_compressor_halos,
    )

    group = dist.group.WORLD
    torch.manual_seed(0)
    full_kv = torch.randn(1, 1, SEQ_LEN, DIM, device=device_type)
    dist.broadcast(full_kv, src=0)
    local_kv = full_kv[:, :, rank * LOCAL_LEN : (rank + 1) * LOCAL_LEN].clone().requires_grad_(True)

    gathered = all_gather_kv(local_kv, group)
    torch.testing.assert_close(gathered, full_kv)

    # Gradient reaches only this rank's slice, with the value the full-tensor
    # backward would have produced there. Each rank backprops through only its
    # OWN slice of the (identical, replicated) gathered tensor -- summing the
    # WHOLE tensor on every rank would make every rank's contribution identical
    # rather than distinct, and the all-reduce in `_Gather.backward` would then
    # inflate the result by `world_size` instead of reproducing the single-rank
    # baseline.
    gathered[:, :, rank * LOCAL_LEN : (rank + 1) * LOCAL_LEN].sum().backward()
    torch.testing.assert_close(local_kv.grad, torch.ones_like(local_kv))

    # Uneven compressed rows land in global window order.
    counts = torch.tensor(COUNTS, device=device_type)
    offset = sum(COUNTS[:rank])
    local_rows = (
        torch.arange(offset, offset + COUNTS[rank], device=device_type, dtype=torch.float32)
        .view(1, COUNTS[rank], 1)
        .expand(1, COUNTS[rank], DIM)
        .contiguous()
    )
    rows = all_gather_compressed_rows(local_rows, counts, group)
    assert rows.shape == (1, sum(COUNTS), DIM)
    torch.testing.assert_close(rows[0, :, 0], torch.arange(sum(COUNTS), device=device_type, dtype=torch.float32))

    # Halos come from the neighbours, with zeros beyond the ends.
    halo = 4
    local_flat = full_kv[0, 0, rank * LOCAL_LEN : (rank + 1) * LOCAL_LEN].unsqueeze(0)
    kv_ext, gate_ext = exchange_compressor_halos(local_flat, local_flat.clone(), halo, group)
    assert kv_ext.shape == (1, halo + LOCAL_LEN + halo, DIM)
    torch.testing.assert_close(kv_ext[:, halo : halo + LOCAL_LEN], local_flat)
    if rank == 0:
        torch.testing.assert_close(kv_ext[:, :halo], torch.zeros_like(kv_ext[:, :halo]))
    else:
        expected = full_kv[0, 0, rank * LOCAL_LEN - halo : rank * LOCAL_LEN].unsqueeze(0)
        torch.testing.assert_close(kv_ext[:, :halo], expected)
    if rank == world_size - 1:
        torch.testing.assert_close(kv_ext[:, -halo:], torch.zeros_like(kv_ext[:, -halo:]))
    else:
        expected = full_kv[0, 0, (rank + 1) * LOCAL_LEN : (rank + 1) * LOCAL_LEN + halo].unsqueeze(0)
        torch.testing.assert_close(kv_ext[:, -halo:], expected)
    torch.testing.assert_close(gate_ext, kv_ext)

    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < CP_SIZE, reason="needs 4 devices")
def test_cp_collectives():
    # A bare ``tempfile.NamedTemporaryFile`` races with the ``file://`` init
    # method: PyTorch's file store itself removes the file once every rank has
    # rendezvoused, so the file's own ``__exit__`` unlink then raises
    # ``FileNotFoundError``. Passing a path inside a ``TemporaryDirectory``
    # (as `tests/parallel/ulysses/test_deepseek_v4_ulysses.py` does) avoids the
    # double-delete: only the file disappears, not the directory.
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(_run, args=(CP_SIZE, init_file), nprocs=CP_SIZE, join=True)
