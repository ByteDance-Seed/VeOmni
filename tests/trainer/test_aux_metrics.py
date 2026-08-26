# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ``outputs.aux_metrics`` on the trainer's reporting path.

``loss_dict`` carries a reduction contract a diagnostic does not share: every
entry is pre-weighted by its share of the step's tokens, so ``postforward``
summing it yields the backward scalar and ``train_step`` summing it across micro
batches yields the step's mean loss. Both summations are wrong for a metric — the
first would fold it into the objective, the second would report it
``gradient_accumulation_steps`` times too large — so metrics travel in their own
dict and are averaged instead. These tests pin that separation end to end, the
averaging, and the route to the ``training/<key>`` name that reaches wandb.
"""

import os
import time
from contextlib import nullcontext
from types import SimpleNamespace


os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch

import veomni.trainer.base as base_trainer_module
import veomni.trainer.callbacks.trace_callback as trace_callback_module
import veomni.trainer.text_trainer as text_trainer_module
import veomni.trainer.vlm_trainer as vlm_trainer_module
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.trace_callback import RESERVED_TRAINING_METRIC_NAMES
from veomni.trainer.text_trainer import TextTrainer
from veomni.trainer.vlm_trainer import VLMTrainer


class _Output:
    """A model output that carries aux metrics, as DeepSeek-V4's does."""

    def __init__(self, loss, aux_metrics):
        self.loss = loss
        self.aux_metrics = aux_metrics


class _PlainOutput:
    """A model output with no ``aux_metrics`` attribute — every other model."""

    def __init__(self, loss):
        self.loss = loss


@pytest.fixture
def identity_loss(monkeypatch):
    """Neutralize ``mean_global_loss``: it all-reduces, so it needs a process group.

    The token weighting it applies is not under test here; what is under test is
    what ``postforward`` does with the dict it returns.
    """
    monkeypatch.setattr(
        base_trainer_module,
        "mean_global_loss",
        lambda losses, *token_len_args: {"foundation_loss": losses},
    )


def _bare_trainer() -> BaseTrainer:
    trainer = object.__new__(BaseTrainer)
    trainer.micro_batch_token_len = {}
    trainer.micro_batches_token_len = {}
    return trainer


def test_aux_metrics_are_kept_out_of_the_losses(identity_loss):
    """The metric is returned separately, so no summation of losses can reach it.

    ``postforward`` sums ``loss_dict`` for the backward scalar, and it is not the
    only consumer to do so, so the guarantee has to be structural rather than an
    ordering invariant. 1000.0 against a loss of 2.0 makes a leak unmistakable.
    """
    trainer = _bare_trainer()
    lm_loss = torch.tensor(2.0, requires_grad=True)
    outputs = _Output(lm_loss, {"indexer_kl": torch.tensor(1000.0)})

    loss, loss_dict, aux_metrics = BaseTrainer.postforward(trainer, outputs, {})

    assert loss.item() == pytest.approx(2.0), "aux metrics must not enter the backward scalar"
    assert aux_metrics["indexer_kl"].item() == pytest.approx(1000.0)
    assert list(loss_dict) == ["foundation_loss"], "a metric in loss_dict would be summed as a loss"
    # The invariant that makes the separation worth having: anything that sums the
    # losses -- here or in a future caller -- cannot pick up a diagnostic.
    assert torch.stack(list(loss_dict.values())).sum().item() == pytest.approx(2.0)


def test_reported_aux_metric_carries_no_gradient(identity_loss):
    """A metric that still had a graph must be detached on the way out.

    ``train_step`` only ever calls ``.item()`` on it, but an attached tensor
    parked in a dict keeps its whole graph alive for the step.
    """
    trainer = _bare_trainer()
    attached = (torch.tensor(3.0, requires_grad=True) * 2).sum()
    assert attached.requires_grad, "the fixture tensor must start out attached"
    outputs = _Output(torch.tensor(2.0, requires_grad=True), {"indexer_kl": attached})

    _, _, aux_metrics = BaseTrainer.postforward(trainer, outputs, {})

    assert not aux_metrics["indexer_kl"].requires_grad
    assert aux_metrics["indexer_kl"].grad_fn is None


def test_output_without_aux_metrics_attribute_is_unchanged(identity_loss):
    """The common case: ``postforward`` is shared by every model in the repo."""
    trainer = _bare_trainer()
    outputs = _PlainOutput(torch.tensor(2.0, requires_grad=True))

    loss, loss_dict, aux_metrics = BaseTrainer.postforward(trainer, outputs, {})

    assert loss.item() == pytest.approx(2.0)
    assert list(loss_dict) == ["foundation_loss"]
    assert aux_metrics == {}


@pytest.mark.parametrize("reported", [None, {}], ids=["none", "empty"])
def test_absent_aux_metrics_add_no_keys(identity_loss, reported):
    trainer = _bare_trainer()
    outputs = _Output(torch.tensor(2.0, requires_grad=True), reported)

    loss, loss_dict, aux_metrics = BaseTrainer.postforward(trainer, outputs, {})

    assert loss.item() == pytest.approx(2.0)
    assert list(loss_dict) == ["foundation_loss"]
    assert aux_metrics == {}


def test_aux_metric_colliding_with_a_loss_key_is_rejected(identity_loss):
    """Separate dicts do not separate the ``training/`` namespace they share.

    Nothing diverges — the metric never touches the objective — but
    ``EnvironMeterCallback`` publishes both dicts under the same prefix, so the
    auxiliary value would surface as ``training/foundation_loss``: a loss curve
    silently reading someone else's number.
    """
    trainer = _bare_trainer()
    outputs = _Output(torch.tensor(2.0, requires_grad=True), {"foundation_loss": torch.tensor(1000.0)})

    with pytest.raises(ValueError, match="already reported under"):
        BaseTrainer.postforward(trainer, outputs, {})


@pytest.mark.parametrize("reserved", sorted(RESERVED_TRAINING_METRIC_NAMES))
def test_aux_metric_colliding_with_a_callback_owned_name_is_rejected(identity_loss, reserved):
    """These clash one layer later than a loss key, in the published namespace.

    ``EnvironMeterCallback`` adds them to ``training/`` itself, so they are absent
    from ``loss_dict`` and the loss-key check alone would let them through.
    """
    trainer = _bare_trainer()
    outputs = _Output(torch.tensor(2.0, requires_grad=True), {reserved: torch.tensor(1000.0)})

    with pytest.raises(ValueError, match="already reported under"):
        BaseTrainer.postforward(trainer, outputs, {})


def _accumulating_trainer(outputs, recorded):
    """A ``BaseTrainer`` reduced to the parts ``train_step`` touches.

    Everything stubbed here is off the reporting path — process groups, the
    optimizer, grad clipping, the model. The accumulation loop itself, and the
    ``postforward`` it calls, are the real ones.
    """
    trainer = _bare_trainer()
    trainer.state = TrainerState(global_step=0)
    trainer.model = SimpleNamespace()
    trainer.optimizer = SimpleNamespace(step=lambda: None, zero_grad=lambda: None)
    trainer.lr_scheduler = SimpleNamespace(step=lambda: None)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            optimizer=SimpleNamespace(max_grad_norm=1.0),
            accelerator=SimpleNamespace(
                dp_replicate_size=1,
                fsdp_config=SimpleNamespace(fsdp_mode="fsdp2", reshard_after_backward=True),
            ),
            sync_each_train_step=False,
        )
    )
    trainer._callbacks = []

    remaining = iter(outputs)
    # Stands in for the forward + backward, so the real postforward runs on a
    # real per-micro-batch model output without a model or autograd.
    trainer.forward_backward_step = lambda micro_batch: BaseTrainer.postforward(trainer, next(remaining), micro_batch)
    # ``TextTrainer`` / ``VLMTrainer`` route their ``on_step_end`` through the
    # base trainer, so patching it here captures the step totals for all three.
    trainer.on_step_end = lambda loss=None, loss_dict=None, grad_norm=None, aux_metrics=None: recorded.update(
        loss=loss, loss_dict=loss_dict, aux_metrics=aux_metrics
    )
    return trainer


# ``TextTrainer`` and ``VLMTrainer`` do not reuse ``BaseTrainer.train_step``; each
# keeps its own copy of the accumulation loop, and so has to reduce the metrics
# itself. Running every case through all three is what keeps them from drifting.
_TRAIN_STEP_OWNERS = {
    "base": (base_trainer_module, None),
    "text": (text_trainer_module, TextTrainer),
    "vlm": (vlm_trainer_module, VLMTrainer),
}


@pytest.fixture(params=sorted(_TRAIN_STEP_OWNERS))
def run_train_step(request, monkeypatch):
    """Runs one training step through whichever trainer owns the loop."""
    module, wrapper_cls = _TRAIN_STEP_OWNERS[request.param]
    monkeypatch.setattr(module, "synchronize", lambda: None)
    monkeypatch.setattr(module, "use_parallel_state", lambda name: nullcontext())
    monkeypatch.setattr(module, "veomni_clip_grad_norm", lambda *args, **kwargs: 0.0)
    # Single process: the all-reduce over the step's token denominators is the identity.
    monkeypatch.setattr(
        module, "reduce_global_loss_token", lambda token_len: {key: value.item() for key, value in token_len.items()}
    )
    # VLMTrainer's loop does not mark compile steps.
    monkeypatch.setattr(module, "mark_compile_step_begin", lambda *args, **kwargs: None, raising=False)

    def run(trainer, micro_batches):
        driver = trainer
        if wrapper_cls is not None:
            driver = object.__new__(wrapper_cls)
            driver.base = trainer
        type(driver).train_step(driver, iter([micro_batches]))

    return run


def test_reported_aux_metric_does_not_scale_with_gradient_accumulation(run_train_step, identity_loss):
    """The reported metric is the mean over the step's micro batches, not their sum.

    The two dicts are reduced differently in the same loop: losses are summed,
    because ``mean_global_loss`` has already weighted each by its share of the
    step's tokens, while a metric carries no such weight and is averaged. Summing
    it instead would report it N times too large — plausible-looking, correctly
    signed, and off by a configuration-dependent factor. Three micro batches, so
    the expectation also separates the mean from a hard-coded halving.
    """
    aux_values = [6.0, 9.0, 12.0]
    lm_losses = [1.0, 2.0, 3.0]
    outputs = [_Output(torch.tensor(lm), {"indexer_kl": torch.tensor(aux)}) for lm, aux in zip(lm_losses, aux_values)]
    recorded = {}
    trainer = _accumulating_trainer(outputs, recorded)
    micro_batches = [{"labels": torch.tensor([[1, 2, -100, 4]])} for _ in aux_values]

    run_train_step(trainer, micro_batches)

    # mean(6, 9, 12) == 9.0; the unreduced sum would report 27.0.
    assert recorded["aux_metrics"]["indexer_kl"] == pytest.approx(9.0)
    assert "indexer_kl" not in recorded["loss_dict"], "the metric must not reach the loss channel"
    # The loss terms, by contrast, must still be summed: each already carries
    # its token-share weight, so their sum is the step's mean loss.
    assert recorded["loss_dict"]["foundation_loss"] == pytest.approx(sum(lm_losses))
    assert recorded["loss"] == pytest.approx(sum(lm_losses))


def test_single_micro_batch_reports_the_metric_unscaled(run_train_step, identity_loss):
    """The N == 1 case, which the averaging must leave alone."""
    recorded = {}
    outputs = [_Output(torch.tensor(1.0), {"indexer_kl": torch.tensor(6.0)})]
    trainer = _accumulating_trainer(outputs, recorded)

    run_train_step(trainer, [{"labels": torch.tensor([[1, 2, -100, 4]])}])

    assert recorded["aux_metrics"]["indexer_kl"] == pytest.approx(6.0)


def test_metric_emitted_by_some_micro_batches_is_averaged_over_all_of_them(run_train_step, identity_loss):
    """The documented consequence of dividing by the step's micro-batch count.

    A key only some micro batches emit is still divided by all of them, so a
    partial emission reads low rather than being silently rescaled to look
    complete. Callers are told to emit a key on every micro batch or none.
    """
    recorded = {}
    outputs = [
        _Output(torch.tensor(1.0), {"indexer_kl": torch.tensor(6.0)}),
        _Output(torch.tensor(1.0), {}),
    ]
    trainer = _accumulating_trainer(outputs, recorded)
    micro_batches = [{"labels": torch.tensor([[1, 2, -100, 4]])} for _ in outputs]

    run_train_step(trainer, micro_batches)

    assert recorded["aux_metrics"]["indexer_kl"] == pytest.approx(3.0)


def test_a_step_without_aux_metrics_reports_an_empty_dict(run_train_step, identity_loss):
    """Every other model in the repo takes this path; it must not report a stray key."""
    recorded = {}
    outputs = [_PlainOutput(torch.tensor(1.0))]
    trainer = _accumulating_trainer(outputs, recorded)

    run_train_step(trainer, [{"labels": torch.tensor([[1, 2, -100, 4]])}])

    assert recorded["aux_metrics"] == {}


def _environ_meter_callback(env_metrics=None, lr=None):
    """``EnvironMeterCallback`` reduced to what ``on_step_end`` reads."""
    callback = object.__new__(trace_callback_module.EnvironMeterCallback)
    callback.parallel_state = SimpleNamespace(fsdp_group=None)
    callback.start_time = time.time()
    callback.lora_config = None
    callback.freeze_vit = None
    callback.trainer = SimpleNamespace(
        environ_meter=SimpleNamespace(step=lambda delta_time, global_step, **kwargs: dict(env_metrics or {})),
        lr_scheduler=None if lr is None else SimpleNamespace(get_last_lr=lambda: [lr]),
    )
    return callback


def test_reserved_names_match_what_the_callback_publishes_itself(monkeypatch):
    """Keeps the reserved set from going stale as the callback gains metrics.

    Derives the callback-owned names rather than restating them: every
    ``training/`` name that did not come from ``loss_dict`` was put there by the
    callback or by the environ meter it merges, which is exactly the set an
    auxiliary key must not collide with. Adding a metric to ``on_step_end``
    without extending ``RESERVED_TRAINING_METRIC_NAMES`` fails here.
    """
    monkeypatch.setattr(trace_callback_module, "all_reduce", lambda value, group=None: value)
    # Mirrors the ``training/``-prefixed keys of ``helper.EnvironMeter.step``. The
    # unprefixed ones it also emits (mfu, flops, memory) land in another namespace
    # and cannot collide, so one stands in for all of them.
    env_metrics = {"training/avg_effective_len": 1.0, "training/avg_sample_seq_len": 2.0, "mfu": 0.5}
    callback = _environ_meter_callback(env_metrics=env_metrics, lr=3.0)
    loss_dict = {"foundation_loss": 2.0}

    callback.on_step_end(TrainerState(global_step=1), loss=2.0, loss_dict=loss_dict, grad_norm=0.5)

    published = {
        key.removeprefix("training/")
        for key in (*callback.trainer.step_train_metrics, *callback.trainer.step_env_metrics)
        if key.startswith("training/")
    }
    assert published - set(loss_dict) == set(RESERVED_TRAINING_METRIC_NAMES)


def test_environ_meter_publishes_aux_metric_beside_the_losses(monkeypatch):
    """``training/indexer_kl`` is what reaches wandb, and no consumer needs ``*_loss``.

    ``EnvironMeterCallback.on_step_end`` — not the wandb callback — is what
    prefixes the names, and it treats them as opaque, so a metric arriving in its
    own dict still lands next to the losses. The ``*_loss`` convention is required
    only by ``mean_global_loss``, which never sees an auxiliary key.
    """
    monkeypatch.setattr(trace_callback_module, "all_reduce", lambda value, group=None: value)
    callback = _environ_meter_callback()

    callback.on_step_end(
        TrainerState(global_step=1),
        loss=2.0,
        loss_dict={"foundation_loss": 2.0},
        grad_norm=0.5,
        aux_metrics={"indexer_kl": 9.0},
    )

    assert callback.trainer.step_train_metrics["training/indexer_kl"] == pytest.approx(9.0)
    assert callback.trainer.step_train_metrics["training/foundation_loss"] == pytest.approx(2.0)
    assert callback.trainer.step_env_metrics["training/indexer_kl"] == pytest.approx(9.0)


@pytest.mark.parametrize("aux_metrics", [None, {}], ids=["none", "empty"])
def test_environ_meter_tolerates_a_step_without_aux_metrics(monkeypatch, aux_metrics):
    """Callbacks predate the parameter, and the trainers that never fill it still call it."""
    monkeypatch.setattr(trace_callback_module, "all_reduce", lambda value, group=None: value)
    callback = _environ_meter_callback()

    callback.on_step_end(
        TrainerState(global_step=1),
        loss=2.0,
        loss_dict={"foundation_loss": 2.0},
        grad_norm=0.5,
        aux_metrics=aux_metrics,
    )

    assert callback.trainer.step_train_metrics["training/foundation_loss"] == pytest.approx(2.0)
    assert callback.trainer.step_train_metrics["training/grad_norm"] == pytest.approx(0.5)
