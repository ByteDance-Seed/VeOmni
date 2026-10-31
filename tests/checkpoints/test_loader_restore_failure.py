"""Reject an incompatible cursor on all ranks before any loader restores."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


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
