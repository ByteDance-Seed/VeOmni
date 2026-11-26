"""Tests for ``train.loss_reduction``, memory logging, and non-finite grad-norm guard on ``OmniTrainer``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.omni_arguments_types import OmniTrainingArguments
from veomni.trainer.omni.omni_trainer import OmniTrainer
from veomni.utils.constants import IGNORE_INDEX


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _trainer_with(*, loss_reduction: str, device: str = "cpu") -> OmniTrainer:
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.device = torch.device(device)
    trainer.args = SimpleNamespace(train=SimpleNamespace(loss_reduction=loss_reduction, log_memory_steps=0))
    return trainer


def test_count_supervised_tokens_ignores_masked_positions():
    trainer = _trainer_with(loss_reduction="token_mean")
    # The count follows the model's causal shift (`labels[1:]` + a trailing IGNORE),
    # so the leading ``5`` is not counted: position 0 never carries a target.
    labels = torch.tensor([5, 6, IGNORE_INDEX, 7, IGNORE_INDEX], dtype=torch.long)
    assert trainer._count_supervised_tokens({"packed_labels": labels}) == 2.0


def test_count_supervised_tokens_accepts_a_list_of_rows():
    """The collator's packed layout is only guaranteed after ``preforward``."""
    trainer = _trainer_with(loss_reduction="token_mean")
    # Each row is shifted on its own: [5, IGN] -> [IGN] and [6,7,8] -> [7,8].
    rows = [torch.tensor([5, IGNORE_INDEX]), torch.tensor([6, 7, 8])]
    assert trainer._count_supervised_tokens({"labels": rows}) == 2.0


def test_count_supervised_tokens_returns_zero_without_labels():
    trainer = _trainer_with(loss_reduction="token_mean")
    assert trainer._count_supervised_tokens({}) == 0


def test_count_supervised_tokens_supports_conversation_list_batches():
    """``conversation_list`` batches count per-sample shifted text labels, ignoring dummy items."""
    from veomni.models.seed_omni.utils.conversation import ConversationItem

    trainer = _trainer_with(loss_reduction="token_mean")
    conv = [
        [
            ConversationItem(type="text", value=None, role="user", meta={"labels": torch.tensor([IGNORE_INDEX, 10])}),
            ConversationItem(type="text", value=None, role="assistant", meta={"labels": torch.tensor([11, 12])}),
            ConversationItem(
                type="text", value=None, role="user", is_dummy=True, meta={"labels": torch.tensor([99, 99])}
            ),
        ],
        [
            ConversationItem(
                type="text", value=None, role="assistant", meta={"labels": torch.tensor([20, 21, IGNORE_INDEX])}
            ),
        ],
    ]
    # Sample 0: [IGN, 10, 11, 12] shifted -> [10, 11, 12] (3 tokens); dummy ignored.
    # Sample 1: [20, 21, IGN] shifted -> [21, IGN] (1 token).
    assert trainer._count_supervised_tokens({"conversation_list": conv}) == 4


def test_count_supervised_tokens_matches_the_models_causal_shift():
    """The count must use the same denominator ``ForCausalLMLoss`` divides by."""
    trainer = _trainer_with(loss_reduction="token_mean")

    # prompt-masked head: shift removes an IGNORE, so the count is unchanged
    assert trainer._count_supervised_tokens({"packed_labels": torch.tensor([IGNORE_INDEX] * 2 + [5, 6, 7])}) == 3.0
    # first position supervised: shift drops it, so 3 becomes 2
    assert trainer._count_supervised_tokens({"packed_labels": torch.tensor([9, IGNORE_INDEX, 5, 6])}) == 2.0
    # last position supervised: shift only appends an IGNORE
    assert trainer._count_supervised_tokens({"packed_labels": torch.tensor([IGNORE_INDEX, 5, 6, 7])}) == 3.0
    # a packed row keeps its batch dim
    assert trainer._count_supervised_tokens({"packed_labels": torch.tensor([[IGNORE_INDEX, 5, 6, 7]])}) == 3.0


def _drive_train_step(
    loss_reduction: str,
    token_counts: list[int],
    per_token_means: list[float],
    *,
    grad_norm: float = 1.0,
):
    """Run ``train_step``'s weighting over fake micro-batches and return what it did."""
    trainer = _trainer_with(loss_reduction=loss_reduction)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.args.train.sync_each_train_step = False

    # Leading IGNORE so the causal shift (which drops position 0) leaves exactly
    # `n` supervised targets.
    micro_batches = [
        {"packed_labels": torch.cat([torch.tensor([IGNORE_INDEX]), torch.full((n,), 7, dtype=torch.long)])}
        for n in token_counts
    ]
    scales: list[float | None] = []
    optimizer = SimpleNamespace(step=MagicMock(), zero_grad=MagicMock())
    lr_scheduler = SimpleNamespace(step=MagicMock())

    trainer._reset_async_activation_offload_if_enabled = lambda: None
    trainer.on_step_begin = lambda **kwargs: None
    trainer.sync_before_train_step = lambda: None
    trainer.model_reshard = lambda *a, **k: None
    trainer.model = SimpleNamespace(
        clip_grad_norm=lambda: torch.tensor(grad_norm),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    def fake_step(micro_batch, *, num_micro_steps=1, backward_scale=None):
        scales.append(backward_scale)
        idx = len(scales) - 1
        loss = torch.tensor(per_token_means[idx], requires_grad=True)
        return loss, {"decode": loss}

    trainer.forward_backward_step = fake_step

    captured = {}
    trainer.on_step_end = lambda **kwargs: captured.update(kwargs)
    trainer.train_step(iter([micro_batches]))
    return scales, captured["loss"], captured["loss_dict"], optimizer, lr_scheduler


def test_token_mean_backward_uses_the_token_share():
    scales, logged, loss_dict, optimizer, lr_scheduler = _drive_train_step("token_mean", [1000, 100], [0.5, 4.0])

    assert scales[0] == pytest.approx(1000 / 1100)
    assert scales[1] == pytest.approx(100 / 1100)
    # Logged loss and per-node loss_dict both reflect the window's token mean.
    expected = (1000 * 0.5 + 100 * 4.0) / 1100
    assert logged == pytest.approx(expected)
    assert loss_dict["decode"] == pytest.approx(expected)
    assert logged != pytest.approx((0.5 + 4.0) / 2)
    optimizer.step.assert_called_once()
    lr_scheduler.step.assert_called_once()
    optimizer.zero_grad.assert_called_once()


def test_sample_mean_backward_stays_equal_weighted():
    scales, logged, loss_dict, _, _ = _drive_train_step("sample_mean", [1000, 100], [0.5, 4.0])

    assert scales == [None, None]
    assert logged == pytest.approx((0.5 + 4.0) / 2)
    assert loss_dict["decode"] == pytest.approx((0.5 + 4.0) / 2)


def test_token_mean_falls_back_to_equal_weights_when_window_has_zero_supervised_tokens():
    scales, logged, _, _, _ = _drive_train_step("token_mean", [0, 0], [0.5, 1.5])

    assert scales == [pytest.approx(0.5), pytest.approx(0.5)]
    assert logged == pytest.approx(1.0)


@pytest.mark.parametrize("bad_norm", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_grad_norm_skips_optimizer_step(bad_norm):
    _, _, _, optimizer, lr_scheduler = _drive_train_step("sample_mean", [4], [1.0], grad_norm=bad_norm)

    optimizer.step.assert_not_called()
    lr_scheduler.step.assert_called_once()
    optimizer.zero_grad.assert_called_once()


def test_the_memory_probe_is_a_no_op_on_a_backend_without_allocator_counters():
    """``torch.cpu`` has no ``memory_allocated``, so the probe must skip rather than raise."""
    trainer = _trainer_with(loss_reduction="sample_mean")
    trainer.args.train.log_memory_steps = 1
    trainer.state = SimpleNamespace(global_step=1)

    trainer._log_memory("unit-test")  # must not raise


def test_omni_training_arguments_validates_loss_reduction_and_log_memory_steps():
    args = OmniTrainingArguments(loss_reduction="token_mean", log_memory_steps=10)
    assert args.loss_reduction == "token_mean"
    assert args.log_memory_steps == 10

    with pytest.raises(ValueError, match="train.loss_reduction"):
        OmniTrainingArguments(loss_reduction="invalid")

    with pytest.raises(ValueError, match="train.log_memory_steps"):
        OmniTrainingArguments(log_memory_steps=-1)
