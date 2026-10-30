"""Exact multisource resume rejects missing sampler counters before applying state."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.trainer.callbacks.global_state_callback import _validate_exact_environ_meter, _validate_exact_epoch
from veomni.utils.helper import MultiSourceCounterItem, MultiSourceInfoTracker


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _valid_meter():
    return {
        "consume_tokens": 1024.0,
        "consume_chunks": 2.0,
        "multisource_tracker": {
            "batch_idx": 1,
            "accumulate_counter": {0: MultiSourceCounterItem(num_tokens=1024, num_samples=2, num_steps=1)},
        },
    }


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "tokens",
        "tracker",
        "batch",
        "counters",
        "negative",
        "source-index",
        "step-count",
        "empty",
        "fractional",
        "wrong-type",
        "batch-mismatch",
        "partial",
    ],
)
def test_invalid_multisource_meter_is_rejected(corruption):
    trainer = SimpleNamespace(
        environ_meter=SimpleNamespace(
            enable_multisource=True, multisource_tracker=SimpleNamespace(names=["source", "other"])
        )
    )
    saved = _valid_meter()
    if corruption == "missing":
        saved = None
    elif corruption == "tokens":
        del saved["consume_tokens"]
    elif corruption == "tracker":
        del saved["multisource_tracker"]
    elif corruption == "batch":
        saved["multisource_tracker"]["batch_idx"] = -1
    elif corruption == "counters":
        del saved["multisource_tracker"]["accumulate_counter"]
    elif corruption == "negative":
        saved["multisource_tracker"]["accumulate_counter"][0].num_samples = -1
    elif corruption == "source-index":
        saved["multisource_tracker"]["accumulate_counter"][2] = saved["multisource_tracker"]["accumulate_counter"].pop(
            0
        )
    elif corruption == "step-count":
        saved["multisource_tracker"]["accumulate_counter"][0].num_steps = None
    elif corruption == "empty":
        saved["multisource_tracker"]["accumulate_counter"] = {}
    elif corruption == "fractional":
        saved["multisource_tracker"]["accumulate_counter"][0].num_samples = 2.5
    elif corruption == "wrong-type":
        saved["multisource_tracker"]["accumulate_counter"][0] = SimpleNamespace(
            num_tokens=1024, num_samples=2, num_steps=1
        )
    elif corruption == "batch-mismatch":
        saved["multisource_tracker"]["batch_idx"] = 2
    elif corruption == "partial":
        saved["multisource_tracker"]["accumulate_counter"][0].num_tokens = 512
        saved["multisource_tracker"]["accumulate_counter"][0].num_samples = 1
    with pytest.raises(RuntimeError):
        _validate_exact_environ_meter(trainer, saved, 1)


def test_real_tracker_restores_and_advances_after_valid_meter(monkeypatch):
    updates = []
    tracker = MultiSourceInfoTracker.__new__(MultiSourceInfoTracker)
    tracker.names = ["source"]
    tracker.boundary_type = "token"
    tracker.parallel_state = SimpleNamespace(dp_size=1, dp_group=None, tp_enabled=False, tp_rank=0)
    tracker.dataloader = SimpleNamespace(update_consumed_tokens=updates.append)
    saved = _valid_meter()
    trainer = SimpleNamespace(environ_meter=SimpleNamespace(enable_multisource=True, multisource_tracker=tracker))
    _validate_exact_environ_meter(trainer, saved, 1)
    tracker.load_state_dict(saved["multisource_tracker"])
    monkeypatch.setattr(dist, "all_gather_object", lambda result, obj, group: result.__setitem__(0, obj))
    tracker.step([0], [512])
    assert updates == [(1, 1536)]
    assert tracker.batch_idx == 2
    assert tracker.accumulate_counter[0].num_samples == 3


def test_exact_resume_rejects_unsupported_single_source_meter():
    trainer = SimpleNamespace(environ_meter=SimpleNamespace(enable_multisource=False))
    with pytest.raises(RuntimeError, match="verified multisource"):
        _validate_exact_environ_meter(trainer, _valid_meter(), 1)


@pytest.mark.parametrize(("step", "epoch", "valid"), [(1, 0, True), (10, 0, False), (3, 1, False), (3, None, False)])
def test_exact_epoch_refuses_ambiguous_position(step, epoch, valid):
    trainer = SimpleNamespace(args=SimpleNamespace(train_steps=10))
    if valid:
        _validate_exact_epoch(trainer, step, epoch)
    else:
        with pytest.raises(RuntimeError, match="epoch"):
            _validate_exact_epoch(trainer, step, epoch)


def _worker(rank, root):
    import veomni.trainer.callbacks.global_state_callback as callback
    import veomni.utils.dist_utils as dist_utils
    from veomni.trainer.callbacks.base import TrainerState

    root = Path(root)
    dist_utils.get_device_type = lambda: "cpu"
    callback.get_device_type = lambda: "cpu"
    dist.init_process_group(
        "gloo", init_method=f"file://{root / 'rendezvous'}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    try:
        callback._has_file_profile = lambda trainer: True
        callback._validate_exact_loader = lambda *args, **kwargs: None
        callback._resume_identity = lambda trainer: "same"
        callback.layout.read_manifest = lambda path: {"world_size": 2, "global_step": 1}
        meter = SimpleNamespace(
            enable_multisource=True,
            multisource_tracker=SimpleNamespace(names=["source"]),
            load_state_dict=lambda state: None,
        )
        trainer = SimpleNamespace(
            args=SimpleNamespace(
                train=SimpleNamespace(global_rank=rank, world_size=2, checkpoint=SimpleNamespace(load_path=str(root))),
                train_steps=10,
            ),
            state=TrainerState(global_step=0),
            environ_meter=meter,
            train_dataloader=SimpleNamespace(load_state_dict=lambda state: None),
        )
        c = callback.GlobalStateCallback.__new__(callback.GlobalStateCallback)
        c.trainer = trainer
        payload = {
            "global_step": 1,
            "train_dataloader": {"cursor": 1},
            "torch_rng_state": torch.get_rng_state(),
            "device_rng_state": torch.get_rng_state(),
            "file_profile_resume_identity": "same",
            "file_profile_epoch": 0,
            "environ_meter": _valid_meter(),
        }
        if rank == 1:
            del payload["environ_meter"]["multisource_tracker"]
        c._read_current = lambda _: payload
        try:
            c.load_global_state()
            outcome = "unexpected-success"
        except RuntimeError as exc:
            outcome = str(exc)
        (root / f"rank-{rank}.json").write_text(json.dumps({"outcome": outcome, "step": trainer.state.global_step}))
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_one_rank_missing_tracker_fails_all_before_applying_state(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(row["step"] == 0 and row["outcome"] != "unexpected-success" for row in rows)
    assert "tracker" in rows[1]["outcome"]


def _unserializable_worker(rank, root):
    from veomni.utils.dist_utils import require_same_rank_value

    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'unserializable-rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        value = (lambda: None) if rank == 0 else {"valid": True}
        try:
            require_same_rank_value(value, "test control value")
            result = "unexpected-success"
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"
        (root / f"unserializable-rank-{rank}.json").write_text(json.dumps({"outcome": result}))
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_unserializable_control_value_fails_all_ranks_without_timeout(tmp_path):
    mp.spawn(_unserializable_worker, args=(str(tmp_path),), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f"unserializable-rank-{rank}.json").read_text())["outcome"] for rank in range(2)]
    assert "pickle" in rows[0].lower()
    assert "serializing test control value failed on another rank" in rows[1]
    assert all("timed out" not in row.lower() and "connection closed" not in row.lower() for row in rows)
