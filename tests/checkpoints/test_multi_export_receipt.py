"""Same-step HF and epoch saves reuse the model owners' DCP transaction."""

import gc
import hashlib
import weakref
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime, _make_mock_trainer
from veomni.checkpoint import layout
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.models.checkpoint.manager import ModelCheckpointManager
from veomni.models.seed_omni.utils.checkpoint import OmniModuleCheckpointManager
from veomni.optim.optimizer import MultiOptimizer
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback, _export_wrote_dcp
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _multi_optimizer(model):
    return MultiOptimizer(model, {"model": torch.optim.AdamW(model.parameters())}, ["model"])


def _runtime(tmp_path):
    runtime = _make_mock_runtime(save_path=str(tmp_path))
    runtime.model = torch.nn.Linear(2, 2)
    runtime.optimizer = _multi_optimizer(runtime.model)
    runtime.lr_scheduler = None
    runtime.parallel_state = SimpleNamespace(global_rank=0, dp_mode="ddp", extra_parallel_names=[])
    return runtime


def test_export_does_not_reuse_an_unknown_owner():
    saved = SimpleNamespace(checkpoint=SimpleNamespace(_last_saved_step=1))
    frozen = SimpleNamespace(checkpoint=None, has_trainable_parameters=False)
    unknown = SimpleNamespace(checkpoint=None)
    trainer = SimpleNamespace(model=SimpleNamespace(module_runtimes={"saved": saved, "frozen": frozen}))
    assert _export_wrote_dcp(trainer, 1)
    trainer.model.module_runtimes["unknown"] = unknown
    assert not _export_wrote_dcp(trainer, 1)


def test_train_end_does_not_retain_optimizer_during_hf_export(tmp_path):
    runtime = _runtime(tmp_path)
    optimizer_ref = weakref.ref(runtime.optimizer)
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    state = TrainerState(global_step=1, epoch=0)
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch("torch.distributed.barrier"),
    ):
        manager = ModelCheckpointManager(runtime)
        runtime.checkpoint = manager
        trainer.model = runtime
        trainer.save_dcp.side_effect = manager.save_dcp

        def export(state, stage="step_end"):
            manager.prepare_export(state, stage)
            gc.collect()
            assert optimizer_ref() is None

        trainer.save_hf_or_lora.side_effect = export
        CheckpointCallback(trainer)._save_hf(state, stage="train_end")


def _real_multi_save_worker(rank, directory, scope, cadence, same_output, target_exists, endpoint):
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        torch.manual_seed(123)

        def make_runtime(name=None):
            runtime = _runtime(directory)
            if scope == "mixed" and name == "vision":
                runtime.optimizer = torch.optim.AdamW(runtime.model.parameters())
            runtime.module_name = name
            runtime.parallel_state = SimpleNamespace(global_rank=rank, dp_mode="ddp", extra_parallel_names=[])
            with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer):
                manager = OmniModuleCheckpointManager(runtime) if name else ModelCheckpointManager(runtime)
            runtime.checkpoint = manager
            runtime.save_dcp = lambda state: runtime.checkpoint.save_dcp(state)
            runtime.save_hf_or_lora = lambda state, stage="step_end": runtime.checkpoint.prepare_export(state, stage)
            return runtime

        names = ("vision", "language") if scope == "mixed" else ("language", "vision") if scope == "omni" else (None,)
        runtimes = [make_runtime(name) for name in names]
        for runtime in runtimes:
            runtime.model(torch.ones(1, 2)).sum().backward()
            runtime.optimizer.step()
            runtime.optimizer.zero_grad()
        trainer = _make_mock_trainer(save_path=directory)
        trainer.args.train.global_rank = rank
        trainer.args.train.world_size = 2
        trainer.args.train.checkpoint.save_steps = 0
        trainer.args.train.checkpoint.save_epochs = 1 if cadence == "step" else 2
        trainer.args.train.checkpoint.hf_save_steps = 1 if cadence == "step" else 0
        trainer.args.train.checkpoint.hf_save_epochs = 0 if cadence == "step" else 1
        trainer.train_dataloader = None
        trainer.data_iterator = None
        trainer.environ_meter.state_dict.return_value = {}
        trainer.channel_loss_callback = None
        trainer.offline_cache_writer = None
        if scope in ("omni", "mixed"):
            modules = {runtime.module_name: runtime for runtime in runtimes}
            modules["frozen"] = SimpleNamespace(
                optimizer=None,
                checkpoint=None,
                has_trainable_parameters=False,
                save_dcp=lambda state: None,
                save_hf_or_lora=lambda state, stage: None,
            )
            trainer.model = OmniModelRuntime(SimpleNamespace(), module_runtimes=modules)
            trainer.save_dcp.side_effect = trainer.model.save_dcp
            trainer.save_hf_or_lora.side_effect = trainer.model.save_hf_or_lora
        else:
            trainer.model = runtimes[0]
            trainer.save_dcp.side_effect = runtimes[0].save_dcp
            trainer.save_hf_or_lora.side_effect = runtimes[0].save_hf_or_lora
        for runtime in runtimes:
            runtime.checkpoint.save_dcp = Mock(wraps=runtime.checkpoint.save_dcp)
        callback = CheckpointCallback(trainer)
        global_callback = GlobalStateCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)
        if cadence == "step":
            callback.on_step_end(state)
            global_callback.on_step_end(state)
        else:
            callback.on_epoch_end(state)
            global_callback.on_epoch_end(state)
        generations = [runtime.checkpoint._last_saved_optimizer_generation for runtime in runtimes]
        assert any(generations) and (scope != "omni" or all(generations))
        snapshot = None
        if rank == 0:
            root = Path(directory)
            snapshot = {
                p.relative_to(root): hashlib.sha256(p.read_bytes()).digest() for p in root.rglob("*") if p.is_file()
            }
            assert snapshot
        next_state = state if cadence == "step" else TrainerState(global_step=1, epoch=1)
        callback.on_epoch_end(next_state)
        global_callback.on_epoch_end(next_state)
        trainer.save_dcp.assert_called_once_with(state)
        for runtime in runtimes:
            runtime.checkpoint.save_dcp.assert_called_once_with(state)
        assert [runtime.checkpoint._last_saved_optimizer_generation for runtime in runtimes] == generations
        if rank == 0:
            assert snapshot == {
                p.relative_to(root): hashlib.sha256(p.read_bytes()).digest() for p in root.rglob("*") if p.is_file()
            }

        target_snapshot = None
        if target_exists:
            target_path = Path(directory) / "new-output"
            if target_exists == "stat_error":
                if rank == 0:
                    target_path.mkdir(parents=True, exist_ok=True)
                    (target_path / "sentinel").write_bytes(b"keep")
                dist.barrier()
            else:
                torch.manual_seed(456)
                target_runtimes = [make_runtime(name) for name in names]
                for runtime in target_runtimes:
                    runtime.train_args.checkpoint.save_path = str(target_path)
                    runtime.model(torch.ones(1, 2)).sum().backward()
                    runtime.optimizer.step()
                    runtime.optimizer.zero_grad()
                    runtime.checkpoint.save_dcp(state)
                target_trainer = _make_mock_trainer(save_path=str(target_path))
                target_trainer.args.train.global_rank = rank
                target_trainer.args.train.world_size = 2
                target_trainer.train_dataloader = None
                target_trainer.data_iterator = None
                target_trainer.environ_meter.state_dict.return_value = {}
                target_trainer.channel_loss_callback = None
                target_trainer.offline_cache_writer = None
                target_trainer.model = OmniModelRuntime(
                    SimpleNamespace(), module_runtimes={runtime.module_name: runtime for runtime in target_runtimes}
                )
                GlobalStateCallback(target_trainer).save_global_state(state)
            if rank == 0:
                target_snapshot = {
                    p.relative_to(target_path): hashlib.sha256(p.read_bytes()).digest()
                    for p in target_path.rglob("*")
                    if p.is_file()
                }
                assert target_snapshot

        restored = [make_runtime(name) for name in names]
        for before, after, generation in zip(runtimes, restored, generations):
            after.train_args.checkpoint.load_path = layout.step_dir(directory, 1)
            after.checkpoint.load()
            assert after.checkpoint._loaded_optimizer_generation == generation
            for original_param, loaded_param in zip(before.model.parameters(), after.model.parameters()):
                torch.testing.assert_close(original_param, loaded_param, rtol=0, atol=0)
                original_optimizer = getattr(before.optimizer, "optimizers_dict", {}).get("model", before.optimizer)
                loaded_optimizer = getattr(after.optimizer, "optimizers_dict", {}).get("model", after.optimizer)
                original_state = original_optimizer.state[original_param]
                loaded_state = loaded_optimizer.state[loaded_param]
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(original_state[key], loaded_state[key], rtol=0, atol=0)
        if scope in ("omni", "mixed"):
            modules = {runtime.module_name: runtime for runtime in restored}
            modules["frozen"] = SimpleNamespace(
                optimizer=None,
                checkpoint=None,
                has_trainable_parameters=False,
                save_dcp=lambda state: None,
                save_hf_or_lora=lambda state, stage: None,
            )
            trainer.model = OmniModelRuntime(SimpleNamespace(), module_runtimes=modules)
        else:
            trainer.model = restored[0]
        trainer.args.train.checkpoint.load_path = layout.step_dir(directory, 1)
        if not same_output:
            new_output = str(Path(directory) / "new-output")
            trainer.args.train.checkpoint.save_path = new_output
            for runtime in restored:
                runtime.train_args.checkpoint.save_path = new_output
        trainer.state = TrainerState()
        trainer.checkpoint_callback = CheckpointCallback(trainer)
        resume_state = GlobalStateCallback(trainer)
        assert resume_state.load_global_state()["global_step"] == 1
        for runtime in restored:
            runtime.checkpoint.save_dcp = Mock(wraps=runtime.checkpoint.save_dcp)
        trainer.save_dcp.side_effect = trainer.model.save_dcp
        trainer.save_hf_or_lora.side_effect = trainer.model.save_hf_or_lora
        resume_step = TrainerState(global_step=1, epoch=1)
        if target_exists:
            import errno
            import os

            original_stat = os.stat
            manifest = layout.manifest_path(layout.step_dir(str(target_path), 1))

            def stat_with_error(path, *args, **kwargs):
                if target_exists == "stat_error" and rank == 1 and str(path) == manifest:
                    raise OSError(errno.EIO, "injected manifest stat error")
                return original_stat(path, *args, **kwargs)

            with (
                patch("veomni.trainer.callbacks.checkpoint_callback.os.stat", side_effect=stat_with_error),
                pytest.raises(
                    (OSError, RuntimeError),
                    match="Cannot overwrite a completed checkpoint|failed on another rank|manifest stat error",
                ) as caught,
            ):
                if target_exists == "dcp":
                    trainer.checkpoint_callback.dcp_every_n_steps = 1
                    trainer.checkpoint_callback.on_step_end(resume_step)
                else:
                    trainer.checkpoint_callback.on_train_end(resume_step)
            if target_exists == "stat_error":
                expected_error = "manifest stat error" if rank == 1 else "failed on another rank"
                assert expected_error in str(caught.value)
            else:
                assert repr(layout.step_dir(str(target_path), 1)) in str(caught.value)
                assert "choose a new checkpoint save_path" in str(caught.value)
            for runtime in restored:
                runtime.checkpoint.save_dcp.assert_not_called()
            if rank == 0:
                assert target_snapshot == {
                    p.relative_to(target_path): hashlib.sha256(p.read_bytes()).digest()
                    for p in target_path.rglob("*")
                    if p.is_file()
                }
            dist.barrier()
            return
        trainer.checkpoint_callback.on_train_end(resume_step)
        resume_state.on_train_end(resume_step)
        for runtime in restored:
            if same_output:
                runtime.checkpoint.save_dcp.assert_not_called()
            else:
                runtime.checkpoint.save_dcp.assert_called_once_with(resume_step)
        if same_output and rank == 0:
            assert snapshot == {
                p.relative_to(root): hashlib.sha256(p.read_bytes()).digest() for p in root.rglob("*") if p.is_file()
            }
        dist.barrier()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.parametrize("scope", ["single", "omni", "mixed"])
@pytest.mark.parametrize("cadence", ["step", "empty_epoch"])
@pytest.mark.parametrize("same_output", [True, False])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_real_multi_dcp_hf_epoch_on_two_ranks(tmp_path, scope, cadence, same_output):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_real_multi_save_worker, args=(str(tmp_path), scope, cadence, same_output, False), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("entry", ["dcp", "hf", "stat_error"])
def test_mixed_omni_existing_target_is_rejected_before_any_owner_writes(tmp_path, entry):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_real_multi_save_worker, args=(str(tmp_path), "mixed", "step", False, entry), nprocs=2)


def _fresh_process_export_worker(rank, root, phase, endpoint):
    import json

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:

        class StatefulLoader:
            def __init__(self, position):
                self.position = position

            def state_dict(self):
                return {"position": self.position}

            def load_state_dict(self, state):
                self.position = state["position"]

        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        torch.manual_seed(123)
        directory = Path(root) / "checkpoint-output"
        runtime = _runtime(directory)
        runtime.parallel_state = SimpleNamespace(global_rank=rank, dp_mode="ddp", extra_parallel_names=[])
        with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer):
            manager = ModelCheckpointManager(runtime)
        runtime.checkpoint = manager
        trainer = _make_mock_trainer(save_path=str(directory))
        trainer.args.train.global_rank = rank
        trainer.args.train.world_size = 2
        trainer.args.train.checkpoint.save_steps = 0
        trainer.args.train.checkpoint.save_epochs = 1
        trainer.args.train.checkpoint.hf_save_steps = 1
        trainer.args.train.checkpoint.hf_save_epochs = 0
        trainer.model = runtime
        trainer.train_dataloader = StatefulLoader(7 + rank if phase == "save" else -1)
        trainer.data_iterator = None
        trainer.environ_meter.state_dict.return_value = {}
        trainer.channel_loss_callback = None
        trainer.offline_cache_writer = None
        trainer.save_dcp.side_effect = manager.save_dcp
        trainer.save_hf_or_lora.side_effect = lambda state, stage="step_end": manager.prepare_export(state, stage)
        trainer.checkpoint_callback = CheckpointCallback(trainer)
        global_callback = GlobalStateCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        def snapshot_optimizer():
            leaf = runtime.optimizer.optimizers_dict["model"]
            return {
                name: {
                    "parameter": parameter.detach().clone(),
                    **{key: leaf.state[parameter][key].detach().clone() for key in ("step", "exp_avg", "exp_avg_sq")},
                }
                for name, parameter in runtime.model.named_parameters()
            }

        def assert_snapshot(expected):
            current = snapshot_optimizer()
            assert current.keys() == expected.keys()
            for name in current:
                for key in current[name]:
                    torch.testing.assert_close(current[name][key], expected[name][key], rtol=0, atol=0)

        def next_update():
            runtime.optimizer.zero_grad()
            runtime.model(torch.full((1, 2), 2.0)).sum().backward()
            runtime.optimizer.step()

        if phase == "save":
            runtime.model(torch.ones(1, 2)).sum().backward()
            runtime.optimizer.step()
            runtime.optimizer.zero_grad()
            trainer.checkpoint_callback.on_step_end(state)
            global_callback.on_step_end(state)
            if rank == 0:
                hashes = {
                    str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in directory.rglob("*")
                    if p.is_file()
                }
                (Path(root) / "saved-hashes.json").write_text(json.dumps(hashes, sort_keys=True))
                before = snapshot_optimizer()
                next_update()
                torch.save({"before": before, "after": snapshot_optimizer()}, Path(root) / "expected.pt")
        else:
            runtime.train_args.checkpoint.load_path = layout.step_dir(str(directory), 1)
            trainer.args.train.checkpoint.load_path = runtime.train_args.checkpoint.load_path
            manager.load()
            trainer.state = TrainerState()
            saved_sidecar = torch.load(
                layout.extra_state_path(runtime.train_args.checkpoint.load_path, rank), weights_only=False
            )
            torch.manual_seed(777 + rank)
            assert not torch.equal(torch.get_rng_state(), saved_sidecar["torch_rng_state"])
            if phase in ("resume_bad_rng", "resume_missing_cursor"):
                if rank == 0:
                    expected = (RuntimeError, "validating the training data cursor failed on another rank")
                elif phase == "resume_bad_rng":
                    expected = (ValueError, "CPU and device RNG states conflict")
                else:
                    expected = (ValueError, "Trainer dataloader cursor is missing")
                with pytest.raises(expected[0], match=expected[1]):
                    global_callback.load_global_state()
                assert trainer.checkpoint_callback._last_dcp_step == -1
                assert manager._last_saved_step == -1
                dist.barrier()
                return
            assert global_callback.load_global_state()["global_step"] == 1
            assert torch.equal(torch.get_rng_state(), saved_sidecar["torch_rng_state"])
            assert trainer.train_dataloader.position == 7 + rank
            expected = torch.load(Path(root) / "expected.pt", weights_only=True)
            assert_snapshot(expected["before"])
            if phase == "resume_update":
                next_update()
                assert_snapshot(expected["after"])
            else:
                manager.save_dcp = Mock(wraps=manager.save_dcp)
                trainer.checkpoint_callback.on_train_end(state)
                global_callback.on_train_end(state)
                manager.save_dcp.assert_not_called()
                if rank == 0:
                    before = json.loads((Path(root) / "saved-hashes.json").read_text())
                    after = {
                        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in directory.rglob("*")
                        if p.is_file()
                    }
                    assert before == after
        dist.barrier()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_fresh_process_same_output_final_export_reuses_proven_dcp(tmp_path):
    from tests.tools.launch_utils import spawn_with_rendezvous

    for phase in ("save", "resume_export", "resume_update"):
        spawn_with_rendezvous(_fresh_process_export_worker, args=(str(tmp_path), phase), nprocs=2)
    sidecar = Path(layout.extra_state_path(layout.step_dir(str(tmp_path / "checkpoint-output"), 1), 1))
    payload = torch.load(sidecar, weights_only=False)
    payload["device_rng_state"] = payload["torch_rng_state"].clone()
    payload["device_rng_state"][0] ^= 1
    torch.save(payload, sidecar)
    spawn_with_rendezvous(_fresh_process_export_worker, args=(str(tmp_path), "resume_bad_rng"), nprocs=2)
    payload["device_rng_state"] = payload["torch_rng_state"].clone()
    torch.save(payload, sidecar)
    loader_file = Path(layout.loader_path(layout.step_dir(str(tmp_path / "checkpoint-output"), 1), 1))
    torch.save({}, loader_file)
    spawn_with_rendezvous(_fresh_process_export_worker, args=(str(tmp_path), "resume_missing_cursor"), nprocs=2)


def _ordinary_restore_failure_worker(rank, root, endpoint):
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        runtime = _runtime(root)
        runtime.optimizer = torch.optim.AdamW(runtime.model.parameters())
        runtime.train_args.checkpoint.load_path = str(root)
        if rank == 1:
            runtime.load_extra_state.side_effect = OSError("local scheduler read failed")
        with patch("veomni.models.checkpoint.manager.build_checkpointer"):
            manager = ModelCheckpointManager(runtime)
        with pytest.raises((OSError, RuntimeError), match="local scheduler read failed|failed on another rank"):
            manager.load()
        assert manager._loaded_checkpoint_path is None
        dist.barrier()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_ordinary_owner_restore_failure_reaches_every_rank(tmp_path):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_ordinary_restore_failure_worker, args=(str(tmp_path),), nprocs=2)
