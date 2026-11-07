# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Trainer state must match the optimizer step restored from DCP."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.checkpoints.test_checkpoint_callback import _make_mock_trainer
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint import layout
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _complete_stub_checkpoint(path, world_size):
    weights = Path(layout.weights_dir(str(path)))
    weights.mkdir(parents=True, exist_ok=True)
    (weights / layout.DCP_MARKER_FILENAME).touch()
    layout.write_manifest(str(path), global_step=3, world_size=world_size)
    assert layout.checkpoint_is_complete(str(path))


def _trainer(tmp_path, *, optimizer_step=3, strict=True):
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.load_path = str(tmp_path)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.train_dataloader = None
    trainer.data_iterator = SimpleNamespace(state_dict=lambda: {})
    trainer.environ_meter.state_dict.return_value = {}
    trainer.channel_loss_callback = None
    trainer.offline_cache_writer = None
    runtime = SimpleNamespace(optimizer=SimpleNamespace(_is_multi_optimizer=strict))
    runtime.checkpoint = SimpleNamespace(
        runtime=runtime,
        _loaded_optimizer_global_step=optimizer_step if strict else None,
        _loaded_optimizer_generation="generation" if strict else None,
    )
    trainer.model = runtime
    return trainer


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "wrong_step",
        "invalid_step",
        "optimizer_step_missing",
        "generation_missing",
        "cursor_generation_missing",
        "cursor_generation_wrong",
        "matching",
        "legacy_missing",
    ],
)
def test_trainer_state_cannot_silently_diverge_from_restored_optimizer(tmp_path, mode):
    _complete_stub_checkpoint(tmp_path, world_size=1)
    trainer = _trainer(tmp_path, strict=mode != "legacy_missing")
    if mode == "optimizer_step_missing":
        trainer.model.checkpoint._loaded_optimizer_global_step = None
    if mode == "generation_missing":
        trainer.model.checkpoint._loaded_optimizer_generation = None
    callback = GlobalStateCallback(trainer)
    state = (
        None
        if mode in ("missing", "legacy_missing")
        else {
            "global_step": 4 if mode == "wrong_step" else 3,
            "_veomni_optimizer_generations": {"model": "generation"},
            "torch_rng_state": torch.get_rng_state(),
        }
    )
    if mode == "invalid_step":
        state["global_step"] = True
    if mode == "cursor_generation_missing":
        state.pop("_veomni_optimizer_generations")
    if mode == "cursor_generation_wrong":
        state["_veomni_optimizer_generations"]["model"] = "older-generation"
    with (
        patch.object(callback, "_read_current", return_value=state),
        patch("veomni.checkpoint.legacy_v0_1_12.apply_legacy_global_state", return_value=None),
        patch("veomni.trainer.callbacks.global_state_callback.set_device_rng_state"),
    ):
        if mode in (
            "missing",
            "wrong_step",
            "invalid_step",
            "optimizer_step_missing",
            "generation_missing",
            "cursor_generation_missing",
            "cursor_generation_wrong",
        ):
            expected = {
                "missing": (RuntimeError, "Trainer state is missing"),
                "wrong_step": (ValueError, "checkpoint steps differ"),
                "invalid_step": (ValueError, "global_step must be a non-negative integer"),
                "optimizer_step_missing": (RuntimeError, "optimizer checkpoint steps"),
                "generation_missing": (RuntimeError, "proven optimizer checkpoint generation"),
                "cursor_generation_missing": (ValueError, "generations differ"),
                "cursor_generation_wrong": (ValueError, "generations differ"),
            }
            error_type, message = expected[mode]
            with pytest.raises(error_type, match=message):
                callback.load_global_state()
            assert trainer.state.global_step == 0
        else:
            callback.load_global_state()
            assert trainer.state.global_step == (3 if mode == "matching" else 0)


@pytest.mark.parametrize("mode", ["missing", "stale_generation", "matching"])
@pytest.mark.parametrize("instrumented", ["none", "source", "target", "both"])
def test_real_dcp_with_scheduler_requires_matching_trainer_generation(tmp_path, mode, instrumented):
    import torch

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime
    from veomni.checkpoint import layout
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.optim.lr_scheduler import build_lr_scheduler
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.trainer.callbacks.base import TrainerState

    def build(with_instrumentation):
        runtime = _make_mock_runtime(save_path=str(tmp_path))
        runtime.model = torch.nn.Linear(2, 2)
        leaf = torch.optim.AdamW(runtime.model.parameters(), lr=0.01)
        runtime.optimizer = MultiOptimizer(runtime.model, {"model": leaf}, ["model"])
        runtime.lr_scheduler = build_lr_scheduler(runtime.optimizer, train_steps=100, lr=0.01)
        if with_instrumentation:
            scheduler = runtime.lr_scheduler["model"]
            original_state_dict = scheduler.state_dict

            def observed_state_dict():
                # A profiling wrapper delegates to the scheduler and keeps the
                # callable itself out of the serializable scheduler snapshot.
                return {key: value for key, value in original_state_dict().items() if key != "state_dict"}

            scheduler.state_dict = observed_state_dict
        runtime.parallel_state = SimpleNamespace(global_rank=0, dp_mode="ddp", extra_parallel_names=[])
        return runtime

    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch("torch.distributed.barrier"),
    ):
        source = build(instrumented in ("source", "both"))
        manager = ModelCheckpointManager(source)
        for _ in range(3):
            source.model(torch.ones(1, 2)).sum().backward()
            source.optimizer.step()
            source.optimizer.zero_grad()
            source.lr_scheduler.step()
        manager.save_dcp(TrainerState(global_step=3))
        source.checkpoint = manager
        source_trainer = _make_mock_trainer(save_path=str(tmp_path))
        source_trainer.model = source
        source_trainer.train_dataloader = None
        source_trainer.data_iterator = SimpleNamespace(state_dict=lambda: {})
        source_trainer.environ_meter.state_dict.return_value = {}
        source_trainer.channel_loss_callback = None
        source_trainer.offline_cache_writer = None
        GlobalStateCallback(source_trainer).save_global_state(TrainerState(global_step=3))
        sidecar = Path(layout.extra_state_path(str(tmp_path / "global_step_3"), 0))
        if mode == "stale_generation":
            cursor = torch.load(sidecar, map_location="cpu", weights_only=False)
            cursor["_veomni_optimizer_generations"] = {"model": "stale"}
            torch.save(cursor, sidecar)
        elif mode == "missing":
            sidecar.unlink()
        target = build(instrumented in ("target", "both"))
        target.train_args.checkpoint.load_path = str(tmp_path / "global_step_3")
        target.checkpoint = ModelCheckpointManager(target)
        target.checkpoint.load()
    assert target.checkpoint._loaded_optimizer_global_step == 3
    assert target.lr_scheduler["model"].last_epoch == 3
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.load_path = target.train_args.checkpoint.load_path
    trainer.model = target
    trainer.state = SimpleNamespace(global_step=0)
    trainer.train_dataloader = None
    if mode == "matching":
        GlobalStateCallback(trainer).load_global_state()
        assert trainer.state.global_step == 3
        for runtime in (source, target):
            runtime.optimizer.zero_grad()
            runtime.model(torch.ones(1, 2)).sum().backward()
            runtime.optimizer.step()
            runtime.lr_scheduler.step()
        for actual, expected in zip(target.model.parameters(), source.model.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert target.lr_scheduler["model"].get_last_lr() == source.lr_scheduler["model"].get_last_lr()
    else:
        with pytest.raises((RuntimeError, ValueError), match="Trainer state is missing|generations differ"):
            GlobalStateCallback(trainer).load_global_state()
        assert trainer.state.global_step == 0
    assert target.lr_scheduler["model"].last_epoch == (4 if mode == "matching" else 3)


def test_train_end_sidecar_keeps_generation_after_optimizer_release(tmp_path):
    import torch

    from veomni.checkpoint import layout
    from veomni.trainer.callbacks.base import TrainerState

    trainer = _trainer(tmp_path)
    trainer.model.optimizer = None
    trainer.model.checkpoint._last_saved_step = 3
    trainer.model.checkpoint._last_saved_optimizer_generation = "saved-generation"
    trainer.environ_meter.state_dict.return_value = {}
    callback = GlobalStateCallback(trainer)
    callback.save_global_state(TrainerState(global_step=3))
    sidecar = layout.extra_state_path(str(tmp_path / "global_step_3"), 0)
    saved = torch.load(sidecar, map_location="cpu", weights_only=False)
    assert saved["_veomni_optimizer_generations"] == {"model": "saved-generation"}


def _resume_worker(rank, tmp_path, mode, endpoint):
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        if rank == 0:
            _complete_stub_checkpoint(tmp_path, world_size=2)
        dist.barrier()
        step = 4 if mode == "optimizer_step" and rank == 1 else 3
        asymmetric_policy = mode in ("asymmetric_dynamic_missing", "asymmetric_optimizer_owner")
        trainer = _trainer(tmp_path, optimizer_step=step, strict=not asymmetric_policy)
        trainer.args.train.global_rank = rank
        trainer.args.train.world_size = 2
        if mode == "asymmetric_dynamic_missing":
            from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader

            trainer._sft_dynamic_exact_resume = True
            if rank == 0:
                trainer.train_dataloader = object.__new__(DynamicBatchSizeDataLoader)
                trainer.train_dataloader._dataloader = SimpleNamespace(dataset=[])
        if mode == "asymmetric_optimizer_owner" and rank == 0:
            trainer.model.checkpoint.confirm_legacy_loaded_checkpoint_for_reuse = lambda *_args: None
        if mode == "asymmetric_generations" and rank == 1:
            trainer.model.checkpoint._loaded_optimizer_generation = "different-generation"
        if mode in ("asymmetric_loader", "asymmetric_loader_error"):
            trainer.args.train_steps = 3

            class Loader:
                def __init__(self):
                    self.iterations = 0

                def load_state_dict(self, state):
                    assert state == {"cursor": 2}

                def __iter__(self):
                    self.iterations += 1
                    if mode == "asymmetric_loader_error":
                        raise ValueError("injected iterator failure")
                    return iter(())

            trainer.train_dataloader = Loader() if rank == 0 else None
        callback = GlobalStateCallback(trainer)
        rng_before_resume = torch.get_rng_state().clone()
        state = (
            None
            if (mode == "missing" and rank == 1) or asymmetric_policy
            else {
                "global_step": 4 if mode == "trainer_step" and rank == 1 else 3,
                "_veomni_optimizer_generations": {
                    "model": "older" if mode == "trainer_generation" and rank == 1 else "generation"
                },
                "torch_rng_state": torch.get_rng_state(),
                "train_dataloader": {"cursor": 2},
            }
        )
        with (
            patch.object(callback, "_read_current", return_value=state),
            patch("veomni.checkpoint.legacy_v0_1_12.apply_legacy_global_state", return_value=None),
            patch("veomni.trainer.callbacks.global_state_callback.set_device_rng_state"),
        ):
            if mode in ("matching", "asymmetric_loader"):
                callback.load_global_state()
                assert trainer.state.global_step == 3
            elif mode == "asymmetric_loader_error":
                with pytest.raises((ValueError, RuntimeError), match="iterator failure|restored data iterator"):
                    callback.load_global_state()
            else:
                expected = {
                    "missing": (RuntimeError, "Trainer state is missing"),
                    "asymmetric_dynamic_missing": (RuntimeError, "resume policy differs across ranks"),
                    "asymmetric_optimizer_owner": (RuntimeError, "resume policy differs across ranks"),
                    "asymmetric_generations": (RuntimeError, "resume policy differs across ranks"),
                    "optimizer_step": (RuntimeError, "Restored training position differs across ranks"),
                    "trainer_step": (
                        (ValueError, "checkpoint steps differ")
                        if rank == 1
                        else (RuntimeError, "validating the training data cursor failed on another rank")
                    ),
                    "trainer_generation": (
                        (ValueError, "generations differ")
                        if rank == 1
                        else (RuntimeError, "validating the training data cursor failed on another rank")
                    ),
                }
                error_type, message = expected[mode]
                with pytest.raises(error_type, match=message):
                    callback.load_global_state()
                assert trainer.state.global_step == 0
                if asymmetric_policy or mode == "asymmetric_generations":
                    assert torch.equal(torch.get_rng_state(), rng_before_resume)
        if mode in ("asymmetric_loader", "asymmetric_loader_error") and rank == 0:
            assert trainer.train_dataloader.iterations == 1
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "optimizer_step",
        "trainer_step",
        "trainer_generation",
        "matching",
        "asymmetric_loader",
        "asymmetric_loader_error",
        "asymmetric_dynamic_missing",
        "asymmetric_optimizer_owner",
        "asymmetric_generations",
    ],
)
def test_multi_optimizer_trainer_resume_is_rank_coordinated(tmp_path, mode):
    spawn_with_rendezvous(_resume_worker, args=(str(tmp_path), mode), nprocs=2)


def _asymmetric_loader_epoch_worker(rank, tmp_path, multi, same_output, endpoint):
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state

    class Loader:
        def __init__(self):
            self.position = None
            self.iterations = 0

        def load_state_dict(self, state):
            self.position = state["position"]

        def __iter__(self):
            self.iterations += 1
            return iter(())

    class Owner:
        def __init__(self, runtime, path):
            self.runtime = runtime
            self._loaded_optimizer_global_step = 3 if multi else None
            self._loaded_optimizer_generation = "generation" if multi else None
            self._loaded_checkpoint_path = path
            self.confirmed = False

        def confirm_loaded_checkpoint_for_reuse(self, step, generation):
            assert (step, generation) == (3, "generation")
            self.confirmed = True

        def confirm_legacy_loaded_checkpoint_for_reuse(self, step, path):
            assert (step, path) == (3, self._loaded_checkpoint_path)
            self.confirmed = True

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        root = Path(tmp_path)
        load_path = layout.step_dir(str(root), 3) if same_output else str(root / "other-output")
        if rank == 0:
            _complete_stub_checkpoint(load_path, world_size=2)
        dist.barrier()
        trainer = _trainer(root, strict=multi)
        trainer.args.train.checkpoint.load_path = load_path
        trainer.args.train.global_rank = rank
        trainer.args.train.world_size = 2
        trainer.args.train_steps = 3
        trainer.train_dataloader = Loader() if rank == 0 else None
        trainer.data_iterator = None
        owner = trainer.model.checkpoint = Owner(trainer.model, load_path)
        state = {
            "global_step": 3,
            "_veomni_optimizer_generations": {"model": "generation"},
            "torch_rng_state": torch.get_rng_state(),
            "train_dataloader": {"position": 7} if rank == 0 else None,
        }
        callback = GlobalStateCallback(trainer)
        with patch.object(callback, "_read_current", return_value=state):
            callback.load_global_state()
        assert trainer.state.global_step == 3
        assert trainer.start_step == 0
        if rank == 0:
            assert trainer.train_dataloader.position == 7
            assert trainer.train_dataloader.iterations == 1
        assert owner.confirmed is same_output
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("same_output", [False, True])
def test_epoch_resume_coordinates_ranks_without_a_local_loader(tmp_path, multi, same_output):
    spawn_with_rendezvous(_asymmetric_loader_epoch_worker, args=(str(tmp_path), multi, same_output), nprocs=2)


def _save_failure_worker(rank, tmp_path, mode, endpoint):
    from contextlib import nullcontext

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.trainer.callbacks.base import TrainerState

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        trainer = _trainer(tmp_path)
        trainer.args.train.global_rank = rank
        trainer.args.train.world_size = 2
        trainer.model.checkpoint._last_saved_step = 3
        trainer.model.checkpoint._last_saved_optimizer_generation = (
            None if mode == "generation" and rank == 1 else "gen"
        )
        callback = GlobalStateCallback(trainer)
        if mode == "cursor" and rank == 1:
            callback.state_dict = lambda _: (_ for _ in ()).throw(ValueError("injected cursor failure"))
        if rank == 0:
            expected = (RuntimeError, "(collecting|preparing) trainer checkpoint state failed on another rank")
        elif mode == "cursor":
            expected = (ValueError, "injected cursor failure")
        elif mode == "generation_fatal":
            expected = (SystemExit, "injected generation failure")
        else:
            expected = (RuntimeError, "matching MultiOptimizer checkpoint generation")
        inject = (
            patch(
                "veomni.trainer.callbacks.global_state_callback._saved_multi_optimizer_generations",
                side_effect=SystemExit("injected generation failure"),
            )
            if mode == "generation_fatal" and rank == 1
            else nullcontext()
        )
        with inject, pytest.raises(expected[0], match=expected[1]):
            callback.save_global_state(TrainerState(global_step=3))
        dist.barrier()
        assert not (tmp_path / "global_step_3").exists()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("mode", ["cursor", "generation", "generation_fatal"])
def test_trainer_save_failure_is_coordinated_before_write(tmp_path, mode):
    spawn_with_rendezvous(_save_failure_worker, args=(tmp_path, mode), nprocs=2)
