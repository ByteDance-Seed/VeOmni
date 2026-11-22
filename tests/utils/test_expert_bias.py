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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.utils.expert_bias import BIAS_ATTR, ExpertBiasUpdater


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

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


def test_updates_accumulate_in_one_direction():
    """A persistent imbalance keeps pushing the bias the same way."""
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


def test_unrecognised_router_output_raises():
    """A router carrying the buffer but breaking the output contract must fail loudly.

    Silently skipping it would leave balancing switched off for the whole run
    while training looked healthy, which is the failure this feature exists to
    end rather than reproduce.
    """

    class OddRouter(ToyRouter):
        def forward(self, hidden_states):
            return torch.zeros(1)  # not the (logits, weights, indices) contract

    model = nn.Module()
    model.gate = OddRouter()
    updater = ExpertBiasUpdater(update_rate=0.1)
    assert updater.attach(model) == 1

    with pytest.raises(RuntimeError, match="did not return"):
        model.gate(torch.zeros(4, 4))


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


@pytest.mark.parametrize("value, expected", [("1e-3", 1e-3), (1e-3, 1e-3), (0, 0.0), ("0", 0.0)])
def test_rate_config_accepts_yaml_numbers(value, expected):
    from veomni.arguments.arguments_types import TrainingArguments

    assert TrainingArguments(moe_router_bias_update_rate=value).moe_router_bias_update_rate == expected


@pytest.mark.parametrize("value", [-1e-3, "nan", float("nan"), float("inf"), "fast", None, True])
def test_rate_config_rejects_invalid_values(value):
    from veomni.arguments.arguments_types import TrainingArguments

    with pytest.raises(ValueError, match="moe_router_bias_update_rate"):
        TrainingArguments(moe_router_bias_update_rate=value)


def test_attach_is_idempotent():
    """A second attach must not double-count through two hooks."""
    model = ModelWithRouter()
    updater = ExpertBiasUpdater(update_rate=0.1)
    assert updater.attach(model) == 1
    assert updater.attach(model) == 1

    indices = torch.zeros(8, 1, dtype=torch.long)
    model.route(indices)
    updater.step()

    bias = getattr(model.routers[0], BIAS_ATTR)
    assert bias[0] == pytest.approx(-0.1)


def test_real_deepseek_router_satisfies_the_output_contract():
    """Pin the assumption the whole module rests on, against the real router.

    Every other test here uses a toy that returns the contract by construction,
    so none of them would notice if DeepseekV3TopkRouter stopped returning
    (logits, weights, indices) and balancing quietly stopped.
    """
    from veomni.models.transformers.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    cfg = SimpleNamespace(
        num_experts_per_tok=2,
        num_local_experts=NUM_EXPERTS,
        hidden_size=16,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )
    # Seed before the weight init, not after: the init draws from the global
    # RNG, so seeding later leaves the routing this test asserts on random.
    torch.manual_seed(0)
    router = DeepseekV3TopkRouter(cfg)
    torch.nn.init.normal_(router.weight, std=0.5)

    updater = ExpertBiasUpdater(update_rate=0.1)
    assert updater.attach(router) == 1, "attach must find the real router by its buffer"

    router(torch.randn(32, cfg.hidden_size))
    updater.step()

    bias = getattr(router, BIAS_ATTR)
    assert torch.any(bias != 0), "the real router's output must drive the update"
    # Sign-based, so every expert moved by exactly the rate, or by nothing at
    # all if it sat at exactly the average load. Compare elementwise:
    # pytest.approx against a tensor collapses to a single Boolean that is true
    # only when every element matches, so the previous form rejected the valid
    # mixed case where one expert was at average load and the rest had moved.
    moved_by_rate = torch.isclose(bias.abs(), torch.full_like(bias, 0.1))
    assert torch.all(moved_by_rate | (bias == 0))


def test_the_bias_update_is_rejected_together_with_lora():
    """LoRA plus attached routers must fail loudly, not silently lose routing.

    The update writes ``e_score_correction_bias``, a registered buffer, while
    the adapter export collects ``named_parameters()`` only, so the exported
    adapter cannot carry what the run trained.

    Driven through a real ``attach`` rather than a bare call, because the
    decision the callback makes is the pair of them: reject only once routers
    were actually found, so a dense model keeps warning and disabling.
    """
    from veomni.utils.expert_bias import ensure_lora_can_carry_the_bias

    lora = {"lora_rank": 8}

    moe = ModelWithRouter()
    moe_updater = ExpertBiasUpdater(update_rate=0.1)
    attached = moe_updater.attach(moe)
    assert attached > 0, "the fixture must attach, or this asserts nothing"
    with pytest.raises(ValueError, match="cannot be combined with LoRA"):
        ensure_lora_can_carry_the_bias(lora, attached)

    # A model with no such router attaches nothing, so the knob does not apply
    # and the caller disables it with a warning. Rejecting that would fail a
    # LoRA run over a setting with no effect on it.
    dense = nn.Sequential(nn.Linear(4, 4))
    dense_updater = ExpertBiasUpdater(update_rate=0.1)
    assert dense_updater.attach(dense) == 0
    ensure_lora_can_carry_the_bias(lora, 0)

    # And without LoRA the update is free to run.
    ensure_lora_can_carry_the_bias({}, attached)
    ensure_lora_can_carry_the_bias(None, attached)


def test_lora_rejection_leaves_no_hooks_on_the_model():
    from veomni.trainer.callbacks.trace_callback import ExpertBiasCallback

    model = ModelWithRouter()
    args = SimpleNamespace(
        train=SimpleNamespace(moe_router_bias_update_rate=0.1),
        model=SimpleNamespace(lora_config={"lora_rank": 8}),
    )
    callback = ExpertBiasCallback(SimpleNamespace(args=args, model=model))

    with pytest.raises(ValueError, match="cannot be combined with LoRA"):
        callback.on_train_begin(None)

    assert not model.routers[0]._forward_hooks


def test_routers_of_different_widths_update_independently():
    """Each router balances against its own expert count and its own mean."""
    model = nn.Module()
    model.wide = ToyRouter(4)
    model.narrow = ToyRouter(2)

    updater = ExpertBiasUpdater(update_rate=0.1)
    assert updater.attach(model) == 2

    model.wide.next_indices = torch.tensor([[0], [0], [0], [1]])
    model.wide(torch.zeros(4, 4))
    model.narrow.next_indices = torch.tensor([[1], [1]])
    model.narrow(torch.zeros(2, 4))
    updater.step()

    assert torch.allclose(model.wide.e_score_correction_bias, torch.tensor([-0.1, 0.0, 0.1, 0.1]))
    assert torch.allclose(model.narrow.e_score_correction_bias, torch.tensor([0.1, -0.1]))


def test_out_of_range_index_does_not_count_against_the_next_router():
    model = nn.Module()
    model.narrow = ToyRouter(2)
    model.wide = ToyRouter(4)
    updater = ExpertBiasUpdater(update_rate=0.1)
    updater.attach(model)

    model.narrow.next_indices = torch.tensor([[2]])
    with pytest.raises(RuntimeError):
        model.narrow(torch.zeros(1, 4))


def test_dspark_config_with_its_own_expert_count_updates_every_router():
    """The real DSpark layout: four backbone experts, two per DSpark stage."""
    from tests.models.compare import eager_ops_config
    from tests.models.tiny_configs import tiny_deepseek_v4_config
    from veomni.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    torch.manual_seed(0)
    config = tiny_deepseek_v4_config()
    config.dspark_block_size = 5
    config.num_nextn_predict_layers = 1
    config.dspark_target_layer_ids = [1, 2, 3]
    config.dspark_markov_rank = 8
    config.dspark_n_routed_experts = 2
    config.dspark_num_experts_per_tok = 1
    config._set_ops_implementation(eager_ops_config())
    model = DeepseekV4ForCausalLM(config).train()

    dspark_router = model.model.mtp[0].mlp.gate
    backbone_router = model.model.layers[3].mlp.gate
    assert getattr(dspark_router, BIAS_ATTR).numel() == 2
    assert getattr(backbone_router, BIAS_ATTR).numel() == 4

    updater = ExpertBiasUpdater(update_rate=0.1)
    assert updater.attach(model) == 2

    # A large starting bias fixes which experts win, so the expected update
    # does not depend on what the random weights route to.
    getattr(backbone_router, BIAS_ATTR).copy_(torch.tensor([10.0, 10.0, 0.0, 0.0]))
    getattr(dspark_router, BIAS_ATTR).copy_(torch.tensor([10.0, 0.0]))
    with torch.no_grad():
        model.model(input_ids=torch.randint(3, config.vocab_size, (1, 8)), return_mtp_outputs=True, use_cache=False)
    updater.step()

    assert torch.allclose(getattr(backbone_router, BIAS_ATTR), torch.tensor([9.9, 9.9, 0.1, 0.1]))
    assert torch.allclose(getattr(dspark_router, BIAS_ATTR), torch.tensor([9.9, 0.1]))
