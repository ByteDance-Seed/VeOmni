"""Same-step exports must not combine different optimizer generations."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.trainer.callbacks.checkpoint_callback import _check_same_step_receipt, _export_wrote_dcp


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))


def _generation_worker(rank, mode, endpoint):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_trainer
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.trainer.callbacks.base import TrainerState
    from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        generation = (
            None
            if mode == "legacy" or (mode == "missing" and rank == 1)
            else "different"
            if mode == "mismatch" and rank == 1
            else "original"
        )

        class Checkpoint:
            def __init__(self):
                self._last_saved_step = 3
                self._last_saved_optimizer_generation = generation
                self._last_export_validated_step = 3
                self._last_export_released_optimizer = True
                self._last_legacy_export_dcp_step = 3 if mode == "legacy" else None
                self.runtime = SimpleNamespace(optimizer=None)
                self.checkpointer = SimpleNamespace(_async_save_failed=False)

            def checkpoint_receipt_is_current(self, step):
                return step == self._last_saved_step

        trainer = _make_mock_trainer()
        trainer.model = SimpleNamespace(checkpoint=Checkpoint())
        if mode in ("mismatch", "missing"):
            with pytest.raises(RuntimeError, match="checkpoint generations differ across ranks"):
                _check_same_step_receipt(trainer, 3, True)
            with pytest.raises(RuntimeError, match="checkpoint generations differ across ranks"):
                _export_wrote_dcp(trainer, 3)
            callback = CheckpointCallback(trainer)
            callback._last_hf_step = 3
            with pytest.raises(RuntimeError, match="checkpoint generations differ across ranks"):
                callback._save_hf(TrainerState(global_step=3), stage="train_end")
            assert callback._last_hf_step == -1
            assert callback._last_dcp_step == -1
            trainer.model.checkpoint._last_saved_optimizer_generation = "original"
            with pytest.raises(RuntimeError, match="previously disagreed"):
                callback._save_hf(TrainerState(global_step=3), stage="train_end")
        else:
            _check_same_step_receipt(trainer, 3, True)
            assert _export_wrote_dcp(trainer, 3)
            callback = CheckpointCallback(trainer)
            callback._save_hf(TrainerState(global_step=3), stage="train_end")
            assert callback._last_hf_step == callback._last_dcp_step == 3
        dist.barrier()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["matching", "mismatch", "missing", "legacy"])
def test_same_step_generation_agreement_with_released_optimizer(mode):
    spawn_with_rendezvous(_generation_worker, args=(mode,), nprocs=2)


def _real_manager_retry_worker(rank, endpoint):
    import weakref
    from unittest.mock import MagicMock, patch

    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime, _make_mock_trainer
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.trainer.callbacks.base import TrainerState
    from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        runtime = _make_mock_runtime()
        runtime.optimizer = MagicMock(spec=[])
        runtime.optimizer._is_multi_optimizer = True
        with patch("veomni.models.checkpoint.manager.build_checkpointer"):
            manager = ModelCheckpointManager(runtime)
        manager._last_saved_step = 3
        manager._last_saved_optimizer_ref = weakref.ref(runtime.optimizer)
        manager._last_saved_optimizer_generation = "first" if rank == 0 else "second"
        runtime.checkpoint = manager
        trainer = _make_mock_trainer()
        trainer.model = runtime
        trainer.save_hf_or_lora.side_effect = lambda state, stage: manager.prepare_export(state, stage)
        callback = CheckpointCallback(trainer)
        state = TrainerState(global_step=3)
        with pytest.raises(RuntimeError, match="checkpoint generations differ across ranks"):
            callback._save_hf(state, stage="train_end")
        assert runtime.optimizer is None
        assert callback._last_hf_step == -1
        with pytest.raises(RuntimeError, match="previously disagreed"):
            callback._save_hf(state, stage="train_end")
        with pytest.raises(RuntimeError, match="previously disagreed"):
            callback.on_epoch_end(TrainerState(global_step=4, epoch=1))
        trainer.save_hf_or_lora.assert_called_once()
        dist.barrier()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


def test_generation_mismatch_stays_failed_after_real_train_end_release():
    spawn_with_rendezvous(_real_manager_retry_worker, nprocs=2)


def _restore_generation_worker(rank, endpoint):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_trainer
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = "/unused/checkpoint"
        trainer.model = SimpleNamespace(
            optimizer=SimpleNamespace(_is_multi_optimizer=True),
            checkpoint=SimpleNamespace(
                _loaded_optimizer_global_step=3,
                _loaded_optimizer_generation="first" if rank == 0 else "second",
            ),
        )
        trainer.checkpoint_callback = CheckpointCallback(trainer)
        assert trainer.checkpoint_callback._invalid_generation_step is None
        with pytest.raises(RuntimeError, match="Restored optimizer checkpoint generations differ across ranks"):
            GlobalStateCallback(trainer).load_global_state()
        assert trainer.checkpoint_callback._invalid_generation_step == 3
        dist.barrier()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


def test_restore_rejects_cross_rank_generation_mismatch_before_clearing_latch():
    spawn_with_rendezvous(_restore_generation_worker, nprocs=2)
