# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Auxiliary-loss-free expert bias update. CPU only, no distributed setup."""

import pytest
import torch
from torch import nn

from veomni.utils.expert_bias import BIAS_ATTR, ExpertBiasUpdater


NUM_EXPERTS = 8
TOP_K = 2


class ToyRouter(nn.Module):
    """Same output contract as the real routers: (logits, weights, indices).

    Routing is fixed rather than learned, so a test can state exactly which
    experts were loaded and check the bias moved the right way.
    """

    def __init__(self, num_experts: int = NUM_EXPERTS):
        super().__init__()
        self.num_experts = num_experts
        self.e_score_correction_bias = nn.Buffer(torch.zeros(num_experts))
        self.next_indices: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor):
        indices = self.next_indices
        logits = torch.zeros(indices.shape[0], self.num_experts)
        weights = torch.ones_like(indices, dtype=torch.float32)
        return logits, weights, indices


class ModelWithRouter(nn.Module):
    def __init__(self, num_routers: int = 1):
        super().__init__()
        self.routers = nn.ModuleList([ToyRouter() for _ in range(num_routers)])

    def route(self, indices: torch.Tensor) -> None:
        for router in self.routers:
            router.next_indices = indices
            router(torch.zeros(indices.shape[0], 4))


def test_attach_finds_routers_by_buffer_not_by_class_name():
    model = ModelWithRouter(num_routers=3)
    # A module without the buffer must not be picked up.
    model.unrelated = nn.Linear(4, 4)

    updater = ExpertBiasUpdater(update_rate=1e-3)
    assert updater.attach(model) == 3


def test_overloaded_expert_bias_goes_down_and_underloaded_goes_up():
    model = ModelWithRouter()
    updater = ExpertBiasUpdater(update_rate=0.1)
    updater.attach(model)

    # Every token to experts 0 and 1; experts 2..7 get nothing.
    indices = torch.zeros(16, TOP_K, dtype=torch.long)
    indices[:, 1] = 1
    model.route(indices)
    updater.step()

    bias = getattr(model.routers[0], BIAS_ATTR)
    assert bias[0] < 0 and bias[1] < 0, "overloaded experts should be pushed down"
    assert torch.all(bias[2:] > 0), "idle experts should be pushed up"
    # Sign-based: every expert moves by exactly the rate, never further.
    assert torch.allclose(bias.abs(), torch.full_like(bias, 0.1))


def test_bias_is_untouched_when_load_is_already_even():
    model = ModelWithRouter()
    updater = ExpertBiasUpdater(update_rate=0.1)
    updater.attach(model)

    # One token per expert: every expert is exactly at the mean, so sign() is 0.
    indices = torch.arange(NUM_EXPERTS, dtype=torch.long).reshape(-1, 1)
    model.route(indices)
    updater.step()

    assert torch.all(getattr(model.routers[0], BIAS_ATTR) == 0)


def test_repeated_steps_reduce_the_gap_between_experts():
    """The point of the mechanism: a persistent imbalance is pushed back."""
    model = ModelWithRouter()
    updater = ExpertBiasUpdater(update_rate=0.05)
    updater.attach(model)

    indices = torch.zeros(32, TOP_K, dtype=torch.long)
    indices[:, 1] = 1
    for _ in range(4):
        model.route(indices)
        updater.step()

    bias = getattr(model.routers[0], BIAS_ATTR)
    # Four steps of 0.05, consistently in the same direction.
    assert torch.allclose(bias[0], torch.tensor(-0.2))
    assert torch.allclose(bias[7], torch.tensor(0.2))
    # The idle experts are now favoured over the hot ones by the selection term.
    assert bias[7] - bias[0] == pytest.approx(0.4)


def test_counts_do_not_leak_across_steps():
    model = ModelWithRouter()
    updater = ExpertBiasUpdater(update_rate=0.1)
    updater.attach(model)

    hot = torch.zeros(8, 1, dtype=torch.long)
    model.route(hot)
    updater.step()

    # Second step with the mirror-image imbalance. If the first step's counts
    # survived, expert 0 would still look loaded and keep sinking.
    cold = torch.full((8, 1), NUM_EXPERTS - 1, dtype=torch.long)
    model.route(cold)
    updater.step()

    bias = getattr(model.routers[0], BIAS_ATTR)
    assert bias[0] == pytest.approx(0.0), "expert 0 should recover once it is idle"
    assert bias[NUM_EXPERTS - 1] == pytest.approx(0.0)


def test_step_without_traffic_is_a_noop():
    model = ModelWithRouter()
    updater = ExpertBiasUpdater(update_rate=0.1)
    updater.attach(model)

    updater.step()
    assert torch.all(getattr(model.routers[0], BIAS_ATTR) == 0)


def test_unrecognised_router_output_is_ignored():
    """An unknown output shape must not be guessed at."""

    class OddRouter(ToyRouter):
        def forward(self, hidden_states):
            return torch.zeros(1)  # not the (logits, weights, indices) contract

    model = nn.Module()
    model.gate = OddRouter()
    updater = ExpertBiasUpdater(update_rate=0.1)
    assert updater.attach(model) == 1

    model.gate.next_indices = torch.zeros(4, 1, dtype=torch.long)
    model.gate(torch.zeros(4, 4))
    updater.step()

    assert torch.all(getattr(model.gate, BIAS_ATTR) == 0)


def test_disable_removes_hooks():
    model = ModelWithRouter()
    updater = ExpertBiasUpdater(update_rate=0.1)
    updater.attach(model)
    updater.disable()

    model.route(torch.zeros(8, 1, dtype=torch.long))
    updater.step()

    assert torch.all(getattr(model.routers[0], BIAS_ATTR) == 0)


def test_rejects_a_nonpositive_rate():
    with pytest.raises(ValueError):
        ExpertBiasUpdater(update_rate=0.0)
