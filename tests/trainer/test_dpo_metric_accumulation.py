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

"""Tests for how ``TextDPOTrainer.train_step`` reduces its metrics across micro batches.

Summing a ``loss_dict`` across micro batches is only meaningful when each entry
already carries that micro batch's share of the step, which is what
``mean_global_loss`` arranges for the other trainers. DPO builds its ``loss_dict``
by hand out of plain per-micro-batch means, so the sum has to be averaged instead.
"""

import os
from contextlib import nullcontext
from types import SimpleNamespace


os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch

import veomni.trainer.text_dpo_trainer as dpo_trainer_module
from veomni.trainer.text_dpo_trainer import TextDPOTrainer


def _micro_batch_metrics(dpo_loss, accuracy, margin):
    """What ``forward_backward_step`` reports for one micro batch.

    A bounded rate, a loss and a signed margin, each already reduced over that
    micro batch's preference pairs.
    """
    return {
        "dpo_loss": torch.tensor(dpo_loss),
        "reward_accuracy": torch.tensor(accuracy),
        "reward_margin": torch.tensor(margin),
    }


@pytest.fixture
def run_dpo_train_step(monkeypatch):
    """Drives the real ``TextDPOTrainer.train_step`` over stubbed micro batches.

    Everything stubbed is off the reporting path: process groups, grad clipping,
    the optimizer, and the forward/backward itself. The accumulation loop and the
    reduction it applies are the real ones.
    """
    monkeypatch.setattr(dpo_trainer_module, "use_parallel_state", lambda name: nullcontext())
    monkeypatch.setattr(dpo_trainer_module, "veomni_clip_grad_norm", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(dpo_trainer_module, "mark_compile_step_begin", lambda *args, **kwargs: None)

    def run(per_micro_batch_metrics):
        recorded = {}
        driver = object.__new__(TextDPOTrainer)
        driver.base = SimpleNamespace(
            args=SimpleNamespace(train=SimpleNamespace(optimizer=SimpleNamespace(max_grad_norm=1.0))),
            state=SimpleNamespace(global_step=0),
            model=SimpleNamespace(),
            optimizer=SimpleNamespace(step=lambda: None, zero_grad=lambda: None),
            lr_scheduler=SimpleNamespace(step=lambda: None),
            sync_before_train_step=lambda: None,
            model_reshard=lambda micro_step, num_micro_steps: None,
            _configure_hsdp_allreduce=lambda micro_step, num_micro_steps: None,
        )
        driver.on_step_begin = lambda micro_batches=None: None
        driver.on_step_end = lambda loss=None, loss_dict=None, grad_norm=None: recorded.update(
            loss=loss, loss_dict=loss_dict
        )

        # The backward scalar DPO reports is its per-micro-batch mean loss, which
        # is also what it puts in loss_dict under ``dpo_loss``.
        remaining = iter(per_micro_batch_metrics)

        def forward_backward_step(micro_batch):
            metrics = next(remaining)
            return metrics["dpo_loss"], metrics

        driver.forward_backward_step = forward_backward_step

        micro_batches = [{"input_ids": torch.tensor([[1, 2]])} for _ in per_micro_batch_metrics]
        TextDPOTrainer.train_step(driver, iter([micro_batches]))
        return recorded

    return run


def test_reward_accuracy_stays_a_rate_under_gradient_accumulation(run_dpo_train_step):
    """The headline symptom: a bounded rate reported outside its own bounds.

    ``reward_accuracy`` is a fraction of preference pairs, so it cannot exceed 1.
    Summed over four micro batches, a model that is right half the time reports
    2.0 -- impossible for the quantity, and off by a factor that tracks
    ``gradient_accumulation_steps`` rather than anything about training.
    """
    recorded = run_dpo_train_step([_micro_batch_metrics(0.5, 0.5, 0.25) for _ in range(4)])

    assert recorded["loss_dict"]["reward_accuracy"] == pytest.approx(0.5)
    assert 0.0 <= recorded["loss_dict"]["reward_accuracy"] <= 1.0


def test_metrics_are_averaged_not_summed_across_micro_batches(run_dpo_train_step):
    """Distinct values per micro batch, so a mean cannot be confused with a sum.

    Three micro batches also separate a mean from a hard-coded halving.
    """
    recorded = run_dpo_train_step(
        [
            _micro_batch_metrics(1.0, 0.25, -0.5),
            _micro_batch_metrics(2.0, 0.50, 0.0),
            _micro_batch_metrics(3.0, 0.75, 0.5),
        ]
    )

    # mean(1, 2, 3) == 2.0; the unreduced sum would report 6.0.
    assert recorded["loss_dict"]["dpo_loss"] == pytest.approx(2.0)
    assert recorded["loss_dict"]["reward_accuracy"] == pytest.approx(0.5)
    # A signed metric has to keep its sign through the reduction.
    assert recorded["loss_dict"]["reward_margin"] == pytest.approx(0.0)


def test_reported_total_loss_matches_the_averaged_loss(run_dpo_train_step):
    """``training/total_loss`` and ``training/dpo_loss`` must not disagree by N.

    They describe the same quantity, so they have to be reduced the same way.
    """
    recorded = run_dpo_train_step([_micro_batch_metrics(1.0, 0.5, 0.0), _micro_batch_metrics(3.0, 0.5, 0.0)])

    assert recorded["loss"] == pytest.approx(2.0)
    assert recorded["loss"] == pytest.approx(recorded["loss_dict"]["dpo_loss"])


def test_single_micro_batch_is_left_alone(run_dpo_train_step):
    """The N == 1 case, which the averaging must not disturb."""
    recorded = run_dpo_train_step([_micro_batch_metrics(1.5, 0.75, 0.25)])

    assert recorded["loss"] == pytest.approx(1.5)
    assert recorded["loss_dict"]["dpo_loss"] == pytest.approx(1.5)
    assert recorded["loss_dict"]["reward_accuracy"] == pytest.approx(0.75)
