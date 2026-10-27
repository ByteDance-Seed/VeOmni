"""Checkpoint file-profile preflight must fail every participating rank."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec


def _worker(rank, root, mode):
    from veomni.checkpoint.dcp_checkpointer import _file_mode_for_checkpoint
    from veomni.optim import ple_world_profile

    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=8),
    )
    try:
        receipt = object()
        optimizer = SimpleNamespace(
            _ple_file_mode_enabled=True,
            _ple_world_profile=True,
            _ple_scratch_root="/test-scratch",
            _ple_world_receipt=receipt,
        )

        def validate(value):
            if mode == "invalid" and rank == 0:
                raise ValueError("injected invalid receipt")

        ple_world_profile._validate_receipt = validate
        group = None
        if mode == "missing" and rank == 0:
            receipt = None
        elif mode == "mismatch" and rank == 0:
            optimizer._ple_file_mode_enabled = False
        elif mode == "ordinary":
            optimizer = None
            receipt = None
        elif mode in ("subgroup", "nonmember"):
            group = dist.new_group(ranks=[0], backend="gloo", timeout=timedelta(seconds=8))
            optimizer = None
            receipt = None
        elif mode == "inspect-error" and rank == 0:

            class BrokenOptimizer:
                @property
                def _ple_file_mode_enabled(self):
                    raise ValueError("injected optimizer inspection failure")

            optimizer = BrokenOptimizer()
        elif mode == "signature-error" and rank == 0:

            class BrokenSignature:
                _ple_file_mode_enabled = True
                _ple_world_profile = True

                @property
                def _ple_scratch_root(self):
                    raise ValueError("injected scratch signature failure")

            optimizer = BrokenSignature()

        try:
            if mode == "subgroup" and rank == 1:
                result = "nonmember-not-called"
            else:
                result = "returned:" + str(_file_mode_for_checkpoint(optimizer, group=group, receipt=receipt))
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"
        (root / f"rank-{rank}.json").write_text(json.dumps({"outcome": result}))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize(
    "mode", ["missing", "invalid", "mismatch", "inspect-error", "signature-error", "ordinary", "subgroup", "nonmember"]
)
def test_checkpoint_preflight_coordinates_profile_errors(tmp_path, mode):
    mp.spawn(_worker, args=(str(tmp_path), mode), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f"rank-{rank}.json").read_text())["outcome"] for rank in range(2)]
    if mode in ("missing", "invalid", "mismatch", "inspect-error", "signature-error"):
        assert all(not row.startswith("returned:") and "timed out" not in row.lower() for row in rows)
        if mode == "missing":
            assert "receipt" in rows[0].lower() and "another rank" in rows[1].lower()
        elif mode == "invalid":
            assert "injected invalid receipt" in rows[0] and "another rank" in rows[1].lower()
        elif mode == "inspect-error":
            assert "injected optimizer inspection failure" in rows[0] and "another rank" in rows[1].lower()
        elif mode == "signature-error":
            assert "injected scratch signature failure" in rows[0] and "another rank" in rows[1].lower()
        else:
            assert all("differs across ranks" in row for row in rows)
    elif mode == "ordinary":
        assert rows == ["returned:False", "returned:False"]
    elif mode == "subgroup":
        assert rows == ["returned:False", "nonmember-not-called"]
    else:
        assert rows[0] == "returned:False" and "non-member" in rows[1]


def _subgroup_worker(rank, root):
    from veomni.utils.dist_utils import require_same_rank_value

    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'subgroup-rendezvous'}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=10),
    )
    try:
        even = dist.new_group(ranks=[0, 2], backend="gloo")
        odd = dist.new_group(ranks=[1, 3], backend="gloo")
        group = even if rank % 2 == 0 else odd
        try:
            require_same_rank_value(rank if rank % 2 == 0 else "same", "subgroup control", group=group)
            outcome = "success"
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
        (root / f"subgroup-rank-{rank}.json").write_text(json.dumps({"outcome": outcome}))
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_control_value_disagreement_stays_within_its_subgroup(tmp_path):
    mp.spawn(_subgroup_worker, args=(str(tmp_path),), nprocs=4, join=True)
    rows = [json.loads((tmp_path / f"subgroup-rank-{rank}.json").read_text())["outcome"] for rank in range(4)]
    assert "differs across ranks" in rows[0] and "differs across ranks" in rows[2]
    assert rows[1] == rows[3] == "success"
