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

``BaseTrainer.postforward`` builds the backward scalar by summing ``loss_dict``,
and ``train_step`` reports the *sum* of every micro batch's ``loss_dict``. Both
summations are load-bearing for losses and both are wrong for a diagnostic: the
first would fold the metric into the objective, the second would report it
``gradient_accumulation_steps`` times too large. These tests pin both, plus the
route from ``loss_dict`` to the ``training/<key>`` name that reaches wandb.
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
        lambda losses, micro_batch_token_len, micro_batches_token_len: {"foundation_loss": losses},
    )


def _bare_trainer() -> BaseTrainer:
    trainer = object.__new__(BaseTrainer)
    trainer.micro_batch_token_len = {}
    trainer.micro_batches_token_len = {}
    return trainer


def test_aux_metrics_are_reported_but_not_backpropagated(identity_loss):
    """The metric is reported without joining the backward scalar.

    ``postforward`` sums ``loss_dict`` to build the scalar it hands to
    ``backward()``, so a metric merged before that sum becomes part of the
    objective. 1000.0 against a loss of 2.0 makes a leak unmistakable.
    """
    trainer = _bare_trainer()
    lm_loss = torch.tensor(2.0, requires_grad=True)
    outputs = _Output(lm_loss, {"indexer_kl": torch.tensor(1000.0)})

    loss, loss_dict = BaseTrainer.postforward(trainer, outputs, {})

    assert loss.item() == pytest.approx(2.0), "aux metrics must not enter the backward scalar"
    assert loss_dict["indexer_kl"].item() == pytest.approx(1000.0)
    assert loss_dict["foundation_loss"].item() == pytest.approx(2.0)


def test_merged_aux_metric_carries_no_gradient(identity_loss):
    """A metric that still had a graph must be detached on the way into ``loss_dict``.

    ``train_step`` only ever calls ``.item()`` on it, but an attached tensor
    parked in a dict keeps its whole graph alive for the step.
    """
    trainer = _bare_trainer()
    attached = (torch.tensor(3.0, requires_grad=True) * 2).sum()
    assert attached.requires_grad, "the fixture tensor must start out attached"
    outputs = _Output(torch.tensor(2.0, requires_grad=True), {"indexer_kl": attached})

    _, loss_dict = BaseTrainer.postforward(trainer, outputs, {})

    assert not loss_dict["indexer_kl"].requires_grad
    assert loss_dict["indexer_kl"].grad_fn is None


def test_output_without_aux_metrics_attribute_is_unchanged(identity_loss):
    """The common case: ``postforward`` is shared by every model in the repo."""
    trainer = _bare_trainer()
    outputs = _PlainOutput(torch.tensor(2.0, requires_grad=True))

    loss, loss_dict = BaseTrainer.postforward(trainer, outputs, {})

    assert loss.item() == pytest.approx(2.0)
    assert list(loss_dict) == ["foundation_loss"]


@pytest.mark.parametrize("aux_metrics", [None, {}], ids=["none", "empty"])
def test_absent_aux_metrics_add_no_keys(identity_loss, aux_metrics):
    trainer = _bare_trainer()
    outputs = _Output(torch.tensor(2.0, requires_grad=True), aux_metrics)

    loss, loss_dict = BaseTrainer.postforward(trainer, outputs, {})

    assert loss.item() == pytest.approx(2.0)
    assert list(loss_dict) == ["foundation_loss"]


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
        )
    )
    trainer._callbacks = []

    remaining = iter(outputs)
    # Stands in for the forward + backward, so the real postforward runs on a
    # real per-micro-batch model output without a model or autograd.
    trainer.forward_backward_step = lambda micro_batch: BaseTrainer.postforward(trainer, next(remaining), micro_batch)
    # ``TextTrainer`` / ``VLMTrainer`` route their ``on_step_end`` through the
    # base trainer, so patching it here captures the step totals for all three.
    trainer.on_step_end = lambda loss=None, loss_dict=None, grad_norm=None: recorded.update(
        loss=loss, loss_dict=loss_dict
    )
    return trainer


# ``TextTrainer`` and ``VLMTrainer`` do not reuse ``BaseTrainer.train_step``; each
# keeps its own copy of the accumulation loop, so each has to publish
# ``num_micro_batches`` for the shared ``postforward`` to normalize against.
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

    ``train_step`` accumulates ``total_loss_dict[k] += v.item()`` over the micro
    batches. That is right for losses, which ``mean_global_loss`` has already
    weighted by their share of the step's tokens, and wrong for an aux metric
    merged after that weighting: summed over N micro batches it would be
    reported N times too large — plausible-looking, correctly signed, and off by
    a configuration-dependent factor. Three micro batches, so the expectation
    also separates the mean from a hard-coded halving.
    """
    aux_values = [6.0, 9.0, 12.0]
    lm_losses = [1.0, 2.0, 3.0]
    outputs = [_Output(torch.tensor(lm), {"indexer_kl": torch.tensor(aux)}) for lm, aux in zip(lm_losses, aux_values)]
    recorded = {}
    trainer = _accumulating_trainer(outputs, recorded)
    micro_batches = [{"labels": torch.tensor([[1, 2, -100, 4]])} for _ in aux_values]

    run_train_step(trainer, micro_batches)

    # mean(6, 9, 12) == 9.0; the unnormalized sum would report 27.0.
    assert recorded["loss_dict"]["indexer_kl"] == pytest.approx(9.0)
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

    assert recorded["loss_dict"]["indexer_kl"] == pytest.approx(6.0)


def test_environ_meter_prefixes_aux_metric_without_a_loss_suffix(monkeypatch):
    """``training/indexer_kl`` is what reaches wandb, and no consumer needs ``*_loss``.

    ``EnvironMeterCallback.on_step_end`` — not the wandb callback — is what
    prefixes ``loss_dict`` keys, and it treats them as opaque names. The
    ``*_loss`` convention is required only by ``mean_global_loss``, which runs
    before the merge and never sees an aux key.
    """
    monkeypatch.setattr(trace_callback_module, "all_reduce", lambda value, group=None: value)
    callback = object.__new__(trace_callback_module.EnvironMeterCallback)
    callback.parallel_state = SimpleNamespace(fsdp_group=None)
    callback.start_time = time.time()
    callback.trainer = SimpleNamespace(
        environ_meter=SimpleNamespace(step=lambda delta_time, global_step: {}),
        lr_scheduler=None,
    )

    callback.on_step_end(
        TrainerState(global_step=1),
        loss=2.0,
        loss_dict={"foundation_loss": 2.0, "indexer_kl": 9.0},
        grad_norm=0.5,
    )

    assert callback.trainer.step_train_metrics["training/indexer_kl"] == pytest.approx(9.0)
    assert callback.trainer.step_train_metrics["training/foundation_loss"] == pytest.approx(2.0)
    assert callback.trainer.step_env_metrics["training/indexer_kl"] == pytest.approx(9.0)
