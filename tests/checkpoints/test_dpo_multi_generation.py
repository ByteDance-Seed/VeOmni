"""DPO epoch-boundary sidecars retain the loaded optimizer generation."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.checkpoints.test_checkpoint_callback import _make_mock_trainer
from veomni.checkpoint import layout
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _DPOTrainer:
    def __init__(self, base):
        self.base = base
        self._dpo_epoch_step = 3
        self._dpo_epoch_complete = False
        self._dpo_pending_epoch_end = None
        self._dpo_pending_loader_state = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def validate_dpo_provenance_checkpoint(self, saved):
        pass

    def dpo_provenance_checkpoint_state(self):
        return {"fixture": "deterministic"}


def test_dpo_pending_epoch_refresh_preserves_loaded_generation_and_cursor(tmp_path):
    base = _make_mock_trainer(save_path=str(tmp_path))
    base.args.train.checkpoint.save_steps = 0
    base.args.train.checkpoint.save_epochs = 0
    base.args.train.checkpoint.save_hf_weights = False
    base.args.train.checkpoint.hf_save_steps = 0
    base.args.train.checkpoint.hf_save_epochs = 0
    base.args.train.checkpoint.load_path = layout.step_dir(str(tmp_path), 3)
    base.train_dataloader = None
    base.data_iterator = None
    base.environ_meter.state_dict.return_value = {}
    base.model.optimizer = SimpleNamespace(_is_multi_optimizer=True)
    manager = base.model.checkpoint = SimpleNamespace(
        _last_saved_step=3,
        _last_saved_optimizer_generation="source",
        _loaded_optimizer_global_step=3,
        _loaded_optimizer_generation="source",
    )
    trainer = _DPOTrainer(base)
    state = TrainerState(global_step=3, epoch=0)
    first = GlobalStateCallback(trainer)
    first.save_global_state(state, require_optimizer_checkpoint=True)
    manager._last_saved_step = -1
    manager._last_saved_optimizer_generation = None
    trainer._dpo_pending_epoch_end = (0, 3)
    trainer._dpo_pending_loader_state = {"buffer": [1, 2]}
    trainer._dpo_epoch_complete = True
    resumed = GlobalStateCallback(trainer)
    resumed._last_saved_step = 3
    resumed._last_saved_optimizer_generations = {"model": "source"}
    resumed.on_epoch_end(state)
    root = layout.step_dir(str(tmp_path), 3)
    extra = torch.load(layout.extra_state_path(root, 0), map_location="cpu", weights_only=False)
    cursor = torch.load(layout.loader_path(root, 0), map_location="cpu", weights_only=False)
    assert extra["_veomni_optimizer_generations"] == {"model": "source"}
    assert extra["dpo_position"]["epoch_complete"] is True
    assert cursor["train_dataloader"] == {"buffer": [1, 2]}


def test_dpo_verified_loaded_receipt_prevents_rewriting_same_epoch_step(tmp_path):
    base = _make_mock_trainer(save_path=str(tmp_path))
    base.args.train.checkpoint.save_epochs = 1
    base.model.checkpoint = SimpleNamespace(
        _last_saved_step=3, _last_saved_optimizer_generation="source", _loaded_optimizer_generation="source"
    )
    callback = CheckpointCallback(_DPOTrainer(base))
    callback.on_epoch_end(TrainerState(global_step=3, epoch=0))
    base.save_dcp.assert_not_called()


def _path_agreement_worker(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.trainer.callbacks.global_state_callback import _agree_dpo_output_checkpoint

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        with pytest.raises(RuntimeError, match="ranks disagree"):
            _agree_dpo_output_checkpoint(rank == 0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_dpo_output_path_disagreement_fails_on_both_ranks(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_path_agreement_worker, args=(str(tmp_path / "rdzv"),), nprocs=2, join=True)
