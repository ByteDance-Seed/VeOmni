"""Reject an incompatible cursor on all ranks before any loader restores."""

from collections import deque
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.data_loader import DistributedDataloader
from veomni.data.dataset import MappingDataset
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _restore_worker(rank, root, failure, dpo):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:

        class Source:
            def load_state_dict(self, state):
                raise AssertionError("source restore must not be called on any rank")

        source = Source()
        if dpo:
            samples = MappingDataset(list(range(4)))
            sampler = StatefulDistributedSampler(samples, num_replicas=1, rank=0, shuffle=False, seed=42)
            source = DistributedDataloader(samples, sampler=sampler, batch_size=1, num_workers=0, drop_last=True)
            source.load_state_dict = Source().load_state_dict

        checkpoint = SimpleNamespace(
            save_steps=0, save_epochs=0, save_hf_weights=False, hf_save_steps=0, hf_save_epochs=0, load_path=str(root)
        )

        class DPOTrainer(SimpleNamespace):
            def validate_dpo_provenance_checkpoint(self, _state):
                pass

        trainer = (DPOTrainer if dpo else SimpleNamespace)(
            args=SimpleNamespace(
                train=SimpleNamespace(
                    global_rank=rank,
                    checkpoint=checkpoint,
                    num_train_epochs=2,
                    seed=42,
                    dataloader_batch_size=1,
                    dyn_bsz=True,
                ),
                data=SimpleNamespace(
                    dataloader=SimpleNamespace(
                        num_workers=0, shuffle=False, persistent_workers=False, in_order=True, drop_last=True
                    )
                ),
                train_steps=10,
            ),
            state=SimpleNamespace(global_step=0),
            train_dataloader=DynamicBatchSizeDataLoader(
                source, TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=1
            ),
            channel_loss_callback=None,
            offline_cache_writer=None,
            environ_meter=None,
            policy_model=SimpleNamespace(parallel_state=SimpleNamespace(dp_size=1, dp_rank=0)),
        )
        with patch("veomni.trainer.callbacks.base.get_parallel_state", return_value=SimpleNamespace()):
            callback = GlobalStateCallback(trainer)

        def saved_state(_path):
            if rank == 0 and failure == "not_mapping":
                return []
            if rank == 0 and failure == "missing_step":
                return {}
            return {
                "global_step": 1,
                "dpo_position": {"epoch": 0, "step_in_epoch": 1, "epoch_complete": False},
                "train_dataloader": {
                    "_veomni_source_exhausted": rank == 0,
                    "_veomni_resume_phase": "after_yield",
                    "num_micro_batch": 1,
                    "step": 1,
                    "dataloader_item_buffer": deque(),
                    "item_buffer": deque(),
                    "dataloader_state": {"cursor": 1},
                    "batching_strategy": TextBatchingStrategy(token_micro_bsz=1, buffer_size=1),
                },
            }

        callback._read_current = saved_state
        with pytest.raises(
            (RuntimeError, AttributeError, KeyError, ValueError),
            match=(
                "exhausted one-shot|validating the training data cursor|reading the trainer state|"
                "has no attribute|global_step|DPO checkpoint recovery"
            ),
        ):
            callback.load_global_state()
        assert trainer.state.global_step == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dpo", [False, True])
@pytest.mark.parametrize("failure", ["exhausted", "not_mapping", "missing_step"])
def test_rank_local_loader_restore_failure_is_coordinated(tmp_path, failure, dpo):
    mp.spawn(_restore_worker, args=(str(tmp_path), failure, dpo), nprocs=2, join=True)
