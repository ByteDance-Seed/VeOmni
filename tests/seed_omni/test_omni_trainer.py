"""Unit tests for the :class:`OmniTrainer` orchestration pieces that need no process group."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.omni_callbacks import OmniModuleHfCallback
from veomni.trainer.omni.omni_trainer import MultiLRScheduler, OmniTrainer, cascade_module_reshard


def _hf_callback(*, hf_save_steps: int = 0, hf_save_epochs: int = 0) -> OmniModuleHfCallback:
    checkpoint = SimpleNamespace(save_hf_weights=True, hf_save_steps=hf_save_steps, hf_save_epochs=hf_save_epochs)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(checkpoint=checkpoint)))
    trainer.save_hf_or_lora = MagicMock()
    return OmniModuleHfCallback(trainer)


def test_hf_export_at_train_end_is_not_skipped_by_a_dcp_save_at_the_same_step():
    """The module managers' ``last_saved_step`` counts DCP saves; HF export must not read it."""
    callback = _hf_callback()
    state = TrainerState(global_step=2, stage="train_end")

    callback.on_train_end(state)

    callback.trainer.save_hf_or_lora.assert_called_once_with(state)


def test_hf_export_is_written_once_per_step():
    callback = _hf_callback(hf_save_steps=2)
    state = TrainerState(global_step=2, stage="step_end")
    callback.on_step_end(state)
    state.stage = "train_end"
    callback.on_train_end(state)

    assert callback.trainer.save_hf_or_lora.call_count == 1


def test_failed_hf_export_is_retried_at_train_end():
    callback = _hf_callback(hf_save_steps=2)
    callback.trainer.save_hf_or_lora.side_effect = [RuntimeError("disk full"), None]
    state = TrainerState(global_step=2, stage="step_end")
    with pytest.raises(RuntimeError):
        callback.on_step_end(state)
    state.stage = "train_end"
    callback.on_train_end(state)

    assert callback.trainer.save_hf_or_lora.call_count == 2


@pytest.mark.parametrize(
    ("micro_step", "num_micro_steps", "expected"),
    [(0, 1, None), (0, 3, False), (1, 3, None), (2, 3, True)],
)
def test_cascade_module_reshard_keeps_params_gathered_between_micro_steps(micro_step, num_micro_steps, expected):
    runtimes = {"a": MagicMock(), "b": MagicMock()}

    cascade_module_reshard(runtimes, micro_step, num_micro_steps)

    for runtime in runtimes.values():
        if expected is None:
            runtime._model_reshard.assert_not_called()
        else:
            runtime._model_reshard.assert_called_once_with(expected)


def _accumulate_grads(num_micro_steps: int) -> torch.Tensor:
    weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(micro_batch):
        loss = (weight * micro_batch["x"]).pow(2).mean()
        return {"loss": loss, "losses": {}}

    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.model = SimpleNamespace(forward=forward, module_runtimes={})
    trainer.args = SimpleNamespace(train=SimpleNamespace(enable_batch_invariant_mode=False))
    trainer.fwd_activation_offload_ctx = nullcontext()
    trainer.bwd_activation_offload_ctx = nullcontext()
    trainer.preforward = lambda micro_batch: micro_batch

    micro_batch = {"x": torch.tensor([3.0, -1.0])}
    for micro_step in range(num_micro_steps):
        trainer.forward_backward_step(micro_batch, micro_step=micro_step, num_micro_steps=num_micro_steps)
    return weight.grad


def test_gradient_accumulation_averages_micro_batch_gradients():
    """Accumulating N copies of one micro-batch must match a single step on it."""
    torch.testing.assert_close(_accumulate_grads(2), _accumulate_grads(1))


def test_multi_lr_scheduler_without_schedulers_reports_zero_lr():
    assert MultiLRScheduler({}).get_last_lr() == [0.0]


def test_offline_cache_is_rejected_before_any_setup():
    args = SimpleNamespace(train=SimpleNamespace(train_type="offline_cache"))

    with pytest.raises(NotImplementedError, match="offline_cache"):
        OmniTrainer(args)
