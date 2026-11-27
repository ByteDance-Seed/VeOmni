"""Tests for ``train.loss_reduction`` (via ``_loss_tokens``) and non-finite grad-norm guard on ``OmniTrainer``."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.omni_arguments_types import OmniTrainingArguments
from veomni.models.seed_omni.graphs.training_graph import TrainingGraph
from veomni.models.seed_omni.modeling_omni import LOSS_KEY, LOSS_TOKENS_KEY, OmniModel
from veomni.models.seed_omni.modules.text_encoder.configuration import TextEncoderConfig
from veomni.models.seed_omni.modules.text_encoder.modeling import TextEncoder
from veomni.trainer.omni.omni_trainer import OmniTrainer
from veomni.utils.constants import IGNORE_INDEX


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _build_trainer(*, loss_reduction: str, model: SimpleNamespace) -> OmniTrainer:
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.device = torch.device("cpu")
    trainer.state = SimpleNamespace(global_step=0)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            loss_reduction=loss_reduction,
            sync_each_train_step=False,
            enable_batch_invariant_mode=False,
        )
    )
    trainer.model = model
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer._reset_async_activation_offload_if_enabled = lambda: None
    trainer.on_step_begin = lambda **kwargs: None
    trainer.sync_before_train_step = lambda: None
    trainer.model_reshard = lambda *a, **k: None
    trainer.preforward = lambda mb: mb
    return trainer


def test_text_encoder_decode_returns_unclamped_loss_tokens():
    """``TextEncoder.decode`` reports the exact supervised token count alongside ``loss``."""
    cfg = TextEncoderConfig(vocab_size=16, hidden_size=8, tie_word_embeddings=False)
    enc = TextEncoder(cfg)
    hidden = torch.randn(4, 8)
    shift_labels = torch.tensor([1, IGNORE_INDEX, 3, IGNORE_INDEX], dtype=torch.long)

    out = enc.decode(hidden_states=hidden, shift_labels=shift_labels)
    assert out["loss"] is not None
    assert int(out["loss_tokens"].item()) == 2

    all_ignored = torch.full((4,), IGNORE_INDEX, dtype=torch.long)
    out_empty = enc.decode(hidden_states=hidden, shift_labels=all_ignored)
    assert out_empty["loss"].item() == pytest.approx(0.0)
    assert int(out_empty["loss_tokens"].item()) == 0


def test_omni_model_forward_collects_loss_tokens_per_node():
    """``OmniModel.forward`` collects optional ``_loss_tokens`` alongside ``_loss`` per node."""

    class _LossNode(torch.nn.Module):
        def decode(self, batch):
            batch[LOSS_KEY] = torch.tensor(2.0)
            batch[LOSS_TOKENS_KEY] = torch.tensor(5, dtype=torch.int64)

    model = OmniModel.__new__(OmniModel)
    torch.nn.Module.__init__(model)
    model.training_graph = TrainingGraph([{"from": "head.decode", "to": "end"}])
    model._losses = {}
    model._loss_tokens = {}
    node = _LossNode()
    model.get_module = lambda name: node

    def runner(mod, node_def, batch):
        getattr(mod, node_def.method)(batch)

    out = model.forward({}, node_runner=runner)
    assert out["loss"].item() == pytest.approx(2.0)
    assert out["losses"]["head.decode"].item() == pytest.approx(2.0)
    assert int(out["loss_tokens"]["head.decode"].item()) == 5


def test_token_mean_accumulated_gradient_and_logged_loss_match_single_step_reference():
    """Scaling backward by ``_loss_tokens`` and dividing grads by window tokens matches exact token mean."""
    param_acc = torch.nn.Parameter(torch.tensor([1.5, -0.5]))
    param_ref = torch.nn.Parameter(param_acc.detach().clone())

    # Two micro-batches with unequal token counts: 3 tokens vs 1 token.
    x_long = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    t_long = torch.tensor([1.0, 0.0, 2.0])
    x_short = torch.tensor([[-2.0, 4.0]])
    t_short = torch.tensor([-1.0])

    # Single-step token-mean reference over all 4 tokens concatenated.
    x_all = torch.cat([x_long, x_short], dim=0)
    t_all = torch.cat([t_long, t_short], dim=0)
    ref_loss = ((x_all @ param_ref - t_all) ** 2).mean()
    ref_loss.backward()

    grad_at_step: list[torch.Tensor] = []
    optimizer = SimpleNamespace(
        param_groups=[{"params": [param_acc]}],
        step=lambda: grad_at_step.append(param_acc.grad.detach().clone()),
        zero_grad=MagicMock(),
    )
    lr_scheduler = SimpleNamespace(step=MagicMock())

    def forward(mb):
        loss = ((mb["x"] @ param_acc - mb["t"]) ** 2).mean()
        n = torch.tensor(mb["x"].shape[0], dtype=torch.int64)
        return {"loss": loss, "losses": {"text_enc.decode": loss}, "loss_tokens": {"text_enc.decode": n}}

    model = SimpleNamespace(
        forward=forward,
        clip_grad_norm=lambda: torch.tensor(1.0),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer = _build_trainer(loss_reduction="token_mean", model=model)
    captured = {}
    trainer.on_step_end = lambda **kwargs: captured.update(kwargs)

    trainer.train_step(iter([[{"x": x_long, "t": t_long}, {"x": x_short, "t": t_short}]]))

    assert len(grad_at_step) == 1
    assert torch.allclose(grad_at_step[0], param_ref.grad, atol=1e-6, rtol=1e-6)
    assert captured["loss"] == pytest.approx(ref_loss.item(), rel=1e-6)
    assert captured["loss_dict"]["text_enc.decode"] == pytest.approx(ref_loss.item(), rel=1e-6)
    lr_scheduler.step.assert_called_once()
    optimizer.zero_grad.assert_called_once()


def test_token_mean_scales_grads_through_nested_multi_optimizer():
    """``_scale_optimizer_grads`` handles ``veomni.optim.optimizer.MultiOptimizer`` (Muon / FSDP2+EP)."""
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import (
        MultiOptimizer as OmniMultiOptimizer,
    )
    from veomni.optim.optimizer import MultiOptimizer as ExtraParallelMultiOptimizer

    p_dense = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p_expert = torch.nn.Parameter(torch.tensor([-1.5, 0.5]))
    dummy_root = torch.nn.Module()
    inner_opt = ExtraParallelMultiOptimizer(
        dummy_root,
        {
            "non_extra_parallel": torch.optim.SGD([p_dense], lr=0.1),
            "ep": torch.optim.SGD([p_expert], lr=0.1),
        },
        key_names=["non_extra_parallel", "ep"],
    )
    outer_opt = OmniMultiOptimizer({"llm": inner_opt})

    grad_snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig_step = outer_opt.step

    def capture_step():
        grad_snapshots.append((p_dense.grad.detach().clone(), p_expert.grad.detach().clone()))
        orig_step()

    outer_opt.step = capture_step
    lr_scheduler = SimpleNamespace(step=MagicMock())

    def forward(mb):
        loss = (p_dense * mb["a"]).sum() + (p_expert * mb["b"]).sum()
        return {
            "loss": loss,
            "losses": {"text_enc.decode": loss},
            "loss_tokens": {"text_enc.decode": torch.tensor(mb["tokens"], dtype=torch.int64)},
        }

    model = SimpleNamespace(
        forward=forward,
        clip_grad_norm=lambda: torch.tensor(1.0),
        optimizer=outer_opt,
        lr_scheduler=lr_scheduler,
    )
    trainer = _build_trainer(loss_reduction="token_mean", model=model)
    trainer.on_step_end = lambda **kwargs: None

    # Micro-batch 0 (tokens=3): grad_dense=[3, 0], grad_expert=[0, 3].
    # Micro-batch 1 (tokens=1): grad_dense=[0, 1], grad_expert=[1, 0].
    # Total tokens=4 -> expected token-mean grads: dense=[0.75, 0.25], expert=[0.25, 0.75].
    micro_batches = [
        {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0]), "tokens": 3},
        {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([1.0, 0.0]), "tokens": 1},
    ]
    trainer.train_step(iter([micro_batches]))

    assert len(grad_snapshots) == 1
    assert torch.allclose(grad_snapshots[0][0], torch.tensor([0.75, 0.25]))
    assert torch.allclose(grad_snapshots[0][1], torch.tensor([0.25, 0.75]))


def test_sample_mean_stays_equal_weighted():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = SimpleNamespace(param_groups=[{"params": [param]}], step=MagicMock(), zero_grad=MagicMock())
    lr_scheduler = SimpleNamespace(step=MagicMock())

    def forward(mb):
        loss = param * mb["val"]
        return {
            "loss": loss,
            "losses": {"text_enc.decode": loss},
            "loss_tokens": {"text_enc.decode": torch.tensor(mb["tokens"], dtype=torch.int64)},
        }

    model = SimpleNamespace(
        forward=forward,
        clip_grad_norm=lambda: torch.tensor(1.0),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer = _build_trainer(loss_reduction="sample_mean", model=model)
    captured = {}
    trainer.on_step_end = lambda **kwargs: captured.update(kwargs)

    trainer.train_step(iter([[{"val": 0.5, "tokens": 1000}, {"val": 4.0, "tokens": 100}]]))

    assert captured["loss"] == pytest.approx((0.5 + 4.0) / 2)
    assert captured["loss_dict"]["text_enc.decode"] == pytest.approx((0.5 + 4.0) / 2)
    assert param.grad.item() == pytest.approx((0.5 + 4.0) / 2)


def test_token_mean_handles_zero_token_window_cleanly():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = SimpleNamespace(param_groups=[{"params": [param]}], step=MagicMock(), zero_grad=MagicMock())
    lr_scheduler = SimpleNamespace(step=MagicMock())

    def forward(mb):
        loss = param * 0.0
        return {
            "loss": loss,
            "losses": {"text_enc.decode": loss},
            "loss_tokens": {"text_enc.decode": torch.tensor(0, dtype=torch.int64)},
        }

    model = SimpleNamespace(
        forward=forward,
        clip_grad_norm=lambda: torch.tensor(0.0),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer = _build_trainer(loss_reduction="token_mean", model=model)
    captured = {}
    trainer.on_step_end = lambda **kwargs: captured.update(kwargs)

    trainer.train_step(iter([[{}, {}]]))

    assert captured["loss"] == pytest.approx(0.0)
    assert captured["loss_dict"]["text_enc.decode"] == pytest.approx(0.0)
    assert param.grad.item() == pytest.approx(0.0)


def test_token_mean_rejects_multi_loss_or_missing_loss_tokens():
    param = torch.nn.Parameter(torch.tensor(1.0))

    # Multi-loss output (e.g. text + image or router aux loss) -> rejected before backward.
    model_multi = SimpleNamespace(
        forward=lambda mb: {
            "loss": param * 2.0,
            "losses": {"text_enc.decode": param * 1.5, "llm.forward": param * 0.5},
            "loss_tokens": {"text_enc.decode": torch.tensor(4, dtype=torch.int64)},
        }
    )
    trainer_multi = _build_trainer(loss_reduction="token_mean", model=model_multi)
    with pytest.raises(ValueError, match="_loss_tokens"):
        trainer_multi.forward_backward_step({}, num_micro_steps=1)
    assert param.grad is None

    # Single loss without `_loss_tokens` -> rejected before backward.
    model_missing = SimpleNamespace(
        forward=lambda mb: {
            "loss": param * 1.0,
            "losses": {"flow.decode_velocity": param * 1.0},
            "loss_tokens": {},
        }
    )
    trainer_missing = _build_trainer(loss_reduction="token_mean", model=model_missing)
    with pytest.raises(ValueError, match="_loss_tokens"):
        trainer_missing.forward_backward_step({}, num_micro_steps=1)
    assert param.grad is None


@pytest.mark.parametrize("bad_norm", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_grad_norm_skips_optimizer_step(bad_norm):
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = SimpleNamespace(param_groups=[{"params": [param]}], step=MagicMock(), zero_grad=MagicMock())
    lr_scheduler = SimpleNamespace(step=MagicMock())
    model = SimpleNamespace(
        forward=lambda mb: {"loss": param * 1.0, "losses": {"decode": param * 1.0}},
        clip_grad_norm=lambda: torch.tensor(bad_norm),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer = _build_trainer(loss_reduction="sample_mean", model=model)
    trainer.on_step_end = lambda **kwargs: None

    trainer.train_step(iter([[{}]]))

    optimizer.step.assert_not_called()
    lr_scheduler.step.assert_called_once()
    optimizer.zero_grad.assert_called_once()


def test_omni_training_arguments_validates_loss_reduction():
    args = OmniTrainingArguments(loss_reduction="token_mean")
    assert args.loss_reduction == "token_mean"

    with pytest.raises(ValueError, match="train.loss_reduction"):
        OmniTrainingArguments(loss_reduction="invalid")
