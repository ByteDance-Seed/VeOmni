"""Reject an incompatible cursor on all ranks before any loader restores."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint import layout
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))


def _restore_worker(rank, root, failure):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:

        class Source:
            def load_state_dict(self, state):
                raise AssertionError("source restore must not be called on any rank")

        checkpoint = SimpleNamespace(
            save_steps=0, save_epochs=0, save_hf_weights=False, hf_save_steps=0, hf_save_epochs=0, load_path=str(root)
        )
        trainer = SimpleNamespace(
            args=SimpleNamespace(train=SimpleNamespace(global_rank=rank, checkpoint=checkpoint), train_steps=10),
            state=SimpleNamespace(global_step=0),
            train_dataloader=DynamicBatchSizeDataLoader(
                Source(), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=1
            ),
            channel_loss_callback=None,
            offline_cache_writer=None,
            environ_meter=None,
        )
        with patch("veomni.trainer.callbacks.base.get_parallel_state", return_value=SimpleNamespace()):
            callback = GlobalStateCallback(trainer)

        def saved_state(_path):
            if rank == 0 and failure == "not_mapping":
                return []
            if rank == 0 and failure == "missing_step":
                return {}
            return {"global_step": 1, "train_dataloader": {"_veomni_source_exhausted": rank == 0}}

        callback._read_current = saved_state
        with pytest.raises(
            (RuntimeError, AttributeError, KeyError),
            match="exhausted one-shot|validating the training data cursor|has no attribute|global_step",
        ):
            callback.load_global_state()
        assert trainer.state.global_step == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", ["exhausted", "not_mapping", "missing_step"])
def test_rank_local_loader_restore_failure_is_coordinated(tmp_path, failure):
    mp.spawn(_restore_worker, args=(str(tmp_path), failure), nprocs=2, join=True)


def _offline_missing_cursor_worker(rank, root):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/offline-init", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:

        class Loader:
            def load_state_dict(self, state):
                raise AssertionError("no loader may restore an incomplete checkpoint")

        class Writer:
            def validate_state_dict(self, state):
                pass

            def load_state_dict(self, state):
                raise AssertionError("no writer may restore an incomplete checkpoint")

        checkpoint = SimpleNamespace(
            save_steps=0, save_epochs=0, save_hf_weights=False, hf_save_steps=0, hf_save_epochs=0, load_path=str(root)
        )
        trainer = SimpleNamespace(
            args=SimpleNamespace(
                train=SimpleNamespace(
                    global_rank=rank, world_size=2, checkpoint=checkpoint, training_task="offline_embedding"
                ),
                train_steps=10,
            ),
            model=SimpleNamespace(optimizer=None, checkpoint=None),
            state=SimpleNamespace(global_step=0),
            train_dataloader=Loader(),
            channel_loss_callback=None,
            offline_cache_writer=Writer(),
            environ_meter=None,
        )
        with patch("veomni.trainer.callbacks.base.get_parallel_state", return_value=SimpleNamespace()):
            callback = GlobalStateCallback(trainer)
        callback._read_current = lambda _: {
            "global_step": 3,
            "offline_cache_writer": {"shard_index": 1},
            **({"train_dataloader": {"cursor": 3}} if rank else {}),
        }
        before_rng = torch.get_rng_state().clone()
        with pytest.raises((RuntimeError, ValueError), match="validating the training data cursor|dataloader cursor"):
            callback.load_global_state()
        assert trainer.state.global_step == 0
        assert torch.equal(before_rng, torch.get_rng_state())
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2
    finally:
        dist.destroy_process_group()


def test_offline_missing_cursor_is_rejected_on_all_ranks(tmp_path):
    layout.write_manifest(str(tmp_path), global_step=3, world_size=2, model_state=False)
    mp.spawn(_offline_missing_cursor_worker, args=(str(tmp_path),), nprocs=2, join=True)
