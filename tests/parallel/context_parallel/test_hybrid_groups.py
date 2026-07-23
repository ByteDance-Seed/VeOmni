import os
import tempfile

import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.distributed.context_parallel.topology import (
    coords_from_sp_rank,
    cp_global_ranks_for_ulysses_rank,
    ulysses_global_ranks_for_cp_rank,
)
from veomni.distributed.sequence_parallel.comm import (
    get_context_parallel_group,
    get_ulysses_sequence_parallel_group,
    get_unified_sequence_parallel_group,
    init_sequence_parallel,
)


def _worker(rank: int, world_size: int, file_name: str, errors: mp.Queue):
    try:
        store = dist.FileStore(file_name, world_size)
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
        cp_size, ulysses_size = 2, 4
        init_sequence_parallel(ulysses_size=ulysses_size, sep_dp=True, cp_size=cp_size)

        cp_rank, ulysses_rank = coords_from_sp_rank(rank % (cp_size * ulysses_size), cp_size, ulysses_size)
        expected_cp = cp_global_ranks_for_ulysses_rank(
            dp_index=0, ulysses_rank=ulysses_rank, cp_size=cp_size, ulysses_size=ulysses_size
        )
        expected_ulysses = ulysses_global_ranks_for_cp_rank(
            dp_index=0, cp_rank=cp_rank, cp_size=cp_size, ulysses_size=ulysses_size
        )

        cp_group = get_context_parallel_group()
        ulysses_group = get_ulysses_sequence_parallel_group()
        sp_group = get_unified_sequence_parallel_group()

        assert sorted(dist.get_process_group_ranks(cp_group)) == expected_cp
        assert sorted(dist.get_process_group_ranks(ulysses_group)) == expected_ulysses
        assert sorted(dist.get_process_group_ranks(sp_group)) == list(range(world_size))
        assert dist.get_rank(cp_group) == expected_cp.index(rank)
        assert dist.get_rank(ulysses_group) == expected_ulysses.index(rank)

        dist.destroy_process_group()
        errors.put(None)
    except Exception as exc:  # noqa: BLE001
        errors.put(repr(exc))


def test_u4_cp2_init_sequence_parallel_groups():
    world_size = 8
    with tempfile.TemporaryDirectory() as tmp:
        file_name = os.path.join(tmp, "pg")
        ctx = mp.get_context("spawn")
        errors = ctx.Queue()
        processes = [
            ctx.Process(target=_worker, args=(rank, world_size, file_name, errors))
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=180)
            assert process.exitcode == 0, f"worker exited with {process.exitcode}"
        for _ in range(world_size):
            err = errors.get(timeout=5)
            assert err is None, err
