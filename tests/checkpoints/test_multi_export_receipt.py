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

        def export(state, stage="step_end"):
            manager.prepare_export(state, stage)
            gc.collect()
            assert optimizer_ref() is None

        trainer.save_hf_or_lora.side_effect = export
        CheckpointCallback(trainer)._save_hf(state, stage="train_end")


def _real_multi_save_worker(rank, directory, scope, cadence, endpoint):
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        torch.manual_seed(123)

        def make_runtime(name=None):
            runtime = _runtime(directory)
            runtime.module_name = name
            runtime.parallel_state = SimpleNamespace(global_rank=rank, dp_mode="ddp", extra_parallel_names=[])
            with patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer):
                manager = OmniModuleCheckpointManager(runtime) if name else ModelCheckpointManager(runtime)
            runtime.checkpoint = manager
            runtime.save_dcp = lambda state: runtime.checkpoint.save_dcp(state)
            runtime.save_hf_or_lora = lambda state, stage="step_end": runtime.checkpoint.prepare_export(state, stage)
            return runtime

        names = ("language", "vision") if scope == "omni" else (None,)
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
        if scope == "omni":
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
        assert all(generations)
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
        if cadence == "step":
            trainer.save_dcp.assert_called_once_with(state)
        else:
            trainer.save_dcp.assert_not_called()
        for runtime in runtimes:
            runtime.checkpoint.save_dcp.assert_called_once_with(state)
        assert [runtime.checkpoint._last_saved_optimizer_generation for runtime in runtimes] == generations
        if rank == 0:
            assert snapshot == {
                p.relative_to(root): hashlib.sha256(p.read_bytes()).digest() for p in root.rglob("*") if p.is_file()
            }

        restored = [make_runtime(name) for name in names]
        for before, after, generation in zip(runtimes, restored, generations):
            after.train_args.checkpoint.load_path = layout.step_dir(directory, 1)
            after.checkpoint.load()
            assert after.checkpoint._loaded_optimizer_generation == generation
            for original_param, loaded_param in zip(before.model.parameters(), after.model.parameters()):
                torch.testing.assert_close(original_param, loaded_param, rtol=0, atol=0)
                original_state = before.optimizer.optimizers_dict["model"].state[original_param]
                loaded_state = after.optimizer.optimizers_dict["model"].state[loaded_param]
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(original_state[key], loaded_state[key], rtol=0, atol=0)
        if scope == "omni":
            modules = {runtime.module_name: runtime for runtime in restored}
            modules["frozen"] = SimpleNamespace(optimizer=None, checkpoint=None, has_trainable_parameters=False)
            trainer.model = OmniModelRuntime(SimpleNamespace(), module_runtimes=modules)
        else:
            trainer.model = restored[0]
        trainer.args.train.checkpoint.load_path = layout.step_dir(directory, 1)
        trainer.state = TrainerState()
        assert GlobalStateCallback(trainer).load_global_state()["global_step"] == 1
        dist.barrier()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.parametrize("scope", ["single", "omni"])
@pytest.mark.parametrize("cadence", ["step", "empty_epoch"])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_real_multi_dcp_hf_epoch_on_two_ranks(tmp_path, scope, cadence):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_real_multi_save_worker, args=(str(tmp_path), scope, cadence), nprocs=2)
