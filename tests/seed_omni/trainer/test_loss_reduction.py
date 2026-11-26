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
    """``conversation_list`` batches count per-sample shifted text labels, preserving media-boundary targets."""
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
        [
            # Non-text media segment adds a masked placeholder row before the shift,
            # so the first supervised text label (30) is NOT dropped.
            ConversationItem(type="image", value=None, role="user"),
            ConversationItem(type="text", value=None, role="assistant", meta={"labels": torch.tensor([30, 31])}),
        ],
    ]
    # Sample 0: [IGN, 10, 11, 12] shifted -> [10, 11, 12] (3 tokens); dummy ignored.
    # Sample 1: [20, 21, IGN] shifted -> [21, IGN] (1 token).
    # Sample 2: [IGN (image), 30, 31] shifted -> [30, 31] (2 tokens).
    assert trainer._count_supervised_tokens({"conversation_list": conv}) == 6


def test_count_supervised_tokens_per_item_shift_matches_bagel_decoder():
    """In ``per_item`` shift mode (``BagelTextEncoder``), every text item shifts independently."""
    from veomni.models.seed_omni.utils.conversation import ConversationItem

    trainer = _trainer_with(loss_reduction="token_mean")
    trainer._text_label_shift_mode = "per_item"
    conv = [
        [
            ConversationItem(type="image", value=None, role="user"),
            ConversationItem(type="text", value=None, role="assistant", meta={"labels": torch.tensor([30, 31, 32])}),
            ConversationItem(type="text", value=None, role="assistant", meta={"labels": torch.tensor([40, 41])}),
        ],
    ]
    # Item 0: [30, 31, 32] -> [31, 32] (2); Item 1: [40, 41] -> [41] (1).
    assert trainer._count_supervised_tokens({"conversation_list": conv}) == 3


def test_validate_loss_reduction_accepts_single_text_decoder_and_rejects_multi_objective():
    """``token_mean`` requires a single ``TextEncoder.decode`` sink and detects ``BagelTextEncoder``."""
    from veomni.models.seed_omni.graphs.training_graph import TrainingGraph
    from veomni.models.seed_omni.modules.bagel.text_encoder.modeling import BagelTextEncoder
    from veomni.models.seed_omni.modules.text_encoder.modeling import TextEncoder

    class _FakeTextEncoder(TextEncoder):
        def __init__(self):
            pass

    class _FakeBagelTextEncoder(BagelTextEncoder):
        def __init__(self):
            pass

    class _FakeImageHead:
        pass

    class _FakeBackbone:
        def __init__(self, *, router_aux: bool = False):
            self._router_aux = router_aux

        def _router_aux_loss_enabled(self) -> bool:
            return self._router_aux

    def _runtime(edges, modules):
        return SimpleNamespace(
            model=SimpleNamespace(training_graph=TrainingGraph(edges)),
            get_module=lambda name: modules[name],
        )

    # Single TextEncoder.decode sink with non-loss-producing backbone -> accepted with per_sample shift.
    trainer = _trainer_with(loss_reduction="token_mean")
    trainer._validate_loss_reduction(
        _runtime(
            [{"from": "llm", "to": "text_enc.decode"}, {"from": "text_enc.decode", "to": "end"}],
            {"llm": _FakeBackbone(router_aux=False), "text_enc": _FakeTextEncoder()},
        )
    )
    assert trainer._text_label_shift_mode == "per_sample"
    assert trainer._token_mean_loss_node == "text_enc.decode"

    # Single BagelTextEncoder.decode sink -> accepted with per_item shift.
    trainer_bagel = _trainer_with(loss_reduction="token_mean")
    trainer_bagel._validate_loss_reduction(
        _runtime(
            [{"from": "mot", "to": "bagel_text.decode"}, {"from": "bagel_text.decode", "to": "end"}],
            {"mot": _FakeBackbone(), "bagel_text": _FakeBagelTextEncoder()},
        )
    )
    assert trainer_bagel._text_label_shift_mode == "per_item"
    assert trainer_bagel._token_mean_loss_node == "bagel_text.decode"

    # Mid-graph MoE LLM with router auxiliary loss enabled -> rejected.
    trainer_moe = _trainer_with(loss_reduction="token_mean")
    with pytest.raises(ValueError, match="also produces a loss"):
        trainer_moe._validate_loss_reduction(
            _runtime(
                [{"from": "llm", "to": "text_enc.decode"}, {"from": "text_enc.decode", "to": "end"}],
                {"llm": _FakeBackbone(router_aux=True), "text_enc": _FakeTextEncoder()},
            )
        )

    # Mid-graph decode node feeding into the text sink -> rejected.
    trainer_mid_decode = _trainer_with(loss_reduction="token_mean")
    with pytest.raises(ValueError, match="also produces a loss"):
        trainer_mid_decode._validate_loss_reduction(
            _runtime(
                [
                    {"from": "llm", "to": "vqvae.decode"},
                    {"from": "vqvae.decode", "to": "text_enc.decode"},
                    {"from": "text_enc.decode", "to": "end"},
                ],
                {"llm": _FakeBackbone(), "vqvae": _FakeImageHead(), "text_enc": _FakeTextEncoder()},
            )
        )

    # Multi-objective graph (e.g. Janus text + VQ image head) -> rejected.
    trainer_multi = _trainer_with(loss_reduction="token_mean")
    with pytest.raises(ValueError, match="single TextEncoder.decode"):
        trainer_multi._validate_loss_reduction(
            _runtime(
                [
                    {"from": "llm", "to": "text_enc.decode"},
                    {"from": "llm", "to": "vqvae.decode"},
                    {"from": "text_enc.decode", "to": "end"},
                    {"from": "vqvae.decode", "to": "end"},
                ],
                {"llm": _FakeBackbone(), "text_enc": _FakeTextEncoder(), "vqvae": _FakeImageHead()},
            )
        )

    # Non-text single sink (e.g. Cosmos3 flow connector) -> rejected.
    trainer_nontext = _trainer_with(loss_reduction="token_mean")
    with pytest.raises(ValueError, match="TextEncoder.decode sink"):
        trainer_nontext._validate_loss_reduction(
            _runtime(
                [{"from": "mot", "to": "flow.decode_velocity"}, {"from": "flow.decode_velocity", "to": "end"}],
                {"mot": _FakeBackbone(), "flow": _FakeImageHead()},
            )
        )


def test_forward_backward_step_rejects_unexpected_extra_loss_under_token_mean():
    """Defense-in-depth: runtime forward emitting a second loss under ``token_mean`` fails before backward."""
    from contextlib import nullcontext

    trainer = _trainer_with(loss_reduction="token_mean")
    trainer._token_mean_loss_node = "text_enc.decode"
    trainer.args.train.enable_batch_invariant_mode = False
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.preforward = lambda mb: mb

    param = torch.nn.Parameter(torch.tensor(1.0))
    trainer.model = SimpleNamespace(
        forward=lambda mb: {
            "loss": param * 2.0,
            "losses": {"text_enc.decode": param * 1.5, "llm.forward": param * 0.5},
        }
    )

    with pytest.raises(ValueError, match="additional loss"):
        trainer.forward_backward_step({}, num_micro_steps=1, backward_scale=1.0)
    assert param.grad is None


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


def test_token_mean_accumulated_gradient_matches_single_step_reference():
    """Real ``forward_backward_step`` with ``backward_scale`` produces exact token-mean gradients."""
    from contextlib import nullcontext

    param_acc = torch.nn.Parameter(torch.tensor([1.5, -0.5]))
    param_ref = torch.nn.Parameter(param_acc.detach().clone())

    # Two micro-batches with unequal supervised token counts: 3 tokens vs 1 token.
    x_long = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    t_long = torch.tensor([1.0, 0.0, 2.0])
    x_short = torch.tensor([[-2.0, 4.0]])
    t_short = torch.tensor([-1.0])

    # Single-step token-mean reference over all 4 tokens concatenated.
    x_all = torch.cat([x_long, x_short], dim=0)
    t_all = torch.cat([t_long, t_short], dim=0)
    ((x_all @ param_ref - t_all) ** 2).mean().backward()

    trainer = _trainer_with(loss_reduction="token_mean")
    trainer.state = SimpleNamespace(global_step=0)
    trainer.args.train.sync_each_train_step = False
    trainer.args.train.enable_batch_invariant_mode = False
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer._reset_async_activation_offload_if_enabled = lambda: None
    trainer.on_step_begin = lambda **kwargs: None
    trainer.sync_before_train_step = lambda: None
    trainer.model_reshard = lambda *a, **k: None
    trainer.preforward = lambda mb: mb
    trainer.on_step_end = lambda **kwargs: None

    grad_at_step: list[torch.Tensor] = []

    def capture_step():
        grad_at_step.append(param_acc.grad.detach().clone())

    def forward(mb):
        loss = ((mb["x"] @ param_acc - mb["t"]) ** 2).mean()
        return {"loss": loss, "losses": {"decode": loss}}

    trainer.model = SimpleNamespace(
        forward=forward,
        clip_grad_norm=lambda: torch.tensor(1.0),
        optimizer=SimpleNamespace(step=capture_step, zero_grad=lambda: None),
        lr_scheduler=SimpleNamespace(step=lambda: None),
    )

    micro_batches = [
        {
            "x": x_long,
            "t": t_long,
            "packed_labels": torch.tensor([IGNORE_INDEX, 1, 2, 3], dtype=torch.long),
        },
        {
            "x": x_short,
            "t": t_short,
            "packed_labels": torch.tensor([IGNORE_INDEX, 4], dtype=torch.long),
        },
    ]
    trainer.train_step(iter([micro_batches]))

    assert len(grad_at_step) == 1
    assert torch.allclose(grad_at_step[0], param_ref.grad, atol=1e-6, rtol=1e-6)


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
