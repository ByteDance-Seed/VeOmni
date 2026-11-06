"""Cross-rank failure paths for weighted SFT and DPO input handling."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous


def _worker(rank, endpoint):
    from veomni.data import data_collator
    from veomni.trainer.base import BaseTrainer
    from veomni.trainer.text_dpo_trainer import TextDPOTrainer
    from veomni.utils import loss_utils

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        data_collator.get_parallel_state = lambda: SimpleNamespace(sp_enabled=False)
        loss_utils.get_parallel_state = lambda: SimpleNamespace(sp_enabled=False)
        base = object.__new__(BaseTrainer)
        base.state = SimpleNamespace(global_step=3)
        base.model = SimpleNamespace()
        base._reset_async_activation_offload_if_enabled = lambda _model: None
        base.on_step_begin = lambda **kwargs: None
        feature = {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "position_ids": torch.tensor([0, 1]),
            "loss_weights": torch.tensor([0.5] if rank == 1 else [0.5, 0.5]),
        }
        packer = data_collator.PackingCollator()
        loader = DataLoader([feature], batch_size=1, collate_fn=lambda samples: [packer(samples)])
        with pytest.raises(RuntimeError, match="data loading failed"):
            base._weighted_sft_preflight(iter(loader), base)
        assert base.state.global_step == 3
        with pytest.raises(RuntimeError, match="end of data at different times"):
            loss_utils.fetch_training_micro_batches(iter([] if rank == 0 else [[{"labels": torch.tensor([1])}]]))
        with pytest.raises(StopIteration):
            loss_utils.fetch_training_micro_batches(iter([]))

        base.model = SimpleNamespace(
            model_config=SimpleNamespace(model_type="qwen4_exp"),
            unwrapped_module=SimpleNamespace(veomni_ce=SimpleNamespace(impl="chunk_loss")),
        )
        batches = [{"labels": torch.tensor([1, 2]), "loss_weights": torch.tensor([0.5, 0.5])}]

        def mutate(micro_batches):
            if rank == 1:
                micro_batches[0].pop("loss_weights")

        base.on_step_begin = mutate
        with pytest.raises(RuntimeError, match="callback or loss-token"):
            base._weighted_sft_preflight(iter([batches]), base)
        assert base.state.global_step == 3

        def add_padding(micro_batches):
            if rank == 1:
                micro_batches[0]["padding_flag"] = True

        base.on_step_begin = add_padding
        batches = [{"labels": torch.tensor([1, 2]), "loss_weights": torch.tensor([0.5, 0.5])}]
        with pytest.raises(RuntimeError, match="callback or loss-token"):
            base._weighted_sft_preflight(iter([batches]), base)
        assert base.state.global_step == 3

        trainer = object.__new__(TextDPOTrainer)
        trainer.base = SimpleNamespace(
            state=SimpleNamespace(global_step=3),
            _reset_async_activation_offload_if_enabled=lambda _model: None,
            on_step_begin=lambda micro_batches, **kwargs: micro_batches[0].pop("loss_weights", None),
        )
        trainer.policy_model = SimpleNamespace()
        batch = {"labels": torch.tensor([1, 2])}
        if rank == 1:
            batch["loss_weights"] = torch.tensor([0.5, 0.5])
        with pytest.raises(RuntimeError, match="DPO preforward validation failed"):
            trainer.train_step(iter([[batch]]))
        assert trainer.state.global_step == 3

        def invalidate(micro_batches, **kwargs):
            if rank == 1:
                micro_batches[0] = None

        trainer.base.on_step_begin = invalidate
        with pytest.raises(RuntimeError, match="DPO preforward validation failed"):
            trainer.train_step(iter([[{"labels": torch.tensor([1, 2])}]]))
        assert trainer.state.global_step == 3
        auxiliary = {"labels": torch.tensor([[-100, 1]])}
        auxiliary["image_labels" if rank == 0 else "audio_labels"] = torch.tensor([1, 2, 3] if rank == 0 else [1])
        _, _, totals = loss_utils.prepare_loss_token_counts([auxiliary])
        assert totals["image_tokens"] == 3 and totals["audio_tokens"] == 1
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_weighted_cross_rank_errors_fail_before_forward():
    spawn_with_rendezvous(_worker, nprocs=2)
