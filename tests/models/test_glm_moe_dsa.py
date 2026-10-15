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

"""GLM-MoE-DSA's routed-expert block, built from the toy config on CPU."""

from pathlib import Path

import pytest
import torch

from tests.tools.training_utils import make_eager_ops_config
from veomni.models.auto import build_foundation_model
from veomni.ops.config.singleton import get_ops_config, set_ops_config
from veomni.utils.moe_monitor import MoERouterMonitor, attach_moe_router_monitor, set_active_monitor


_TOY_CONFIG = Path(__file__).resolve().parents[1] / "toy_config" / "glm_moe_dsa_toy"


@pytest.fixture
def build_toy():
    """``build_foundation_model`` installs the ops config process-wide; restore it afterwards."""
    previous = get_ops_config()

    def build(init_device):
        return build_foundation_model(
            config_path=str(_TOY_CONFIG),
            weights_path=None,
            torch_dtype="float32",
            init_device=init_device,
            ops_implementation=make_eager_ops_config(),
        )

    try:
        yield build
    finally:
        set_ops_config(previous)


def _sparse_layers(config):
    return [i for i, layer_type in enumerate(config.mlp_layer_types) if layer_type == "sparse"]


def test_ep_plan_owns_only_the_routed_experts_of_sparse_layers(build_toy):
    """Dense-MLP layers have no experts to shard."""
    model = build_toy("meta")
    sparse_layers = _sparse_layers(model.config)
    assert 0 < len(sparse_layers) < model.config.num_hidden_layers

    owners = model.get_parallel_plan().get_extra_parallel_fsdp_no_shard_info(model, "ep")
    assert sorted(owners) == [f"model.layers.{i}.mlp.experts" for i in sparse_layers]


def test_moe_block_reports_router_indices_to_the_monitor(build_toy):
    model = build_toy("cpu")
    config = model.config
    sparse_layers = _sparse_layers(config)

    monitor = MoERouterMonitor(num_experts=config.n_routed_experts)
    assert attach_moe_router_monitor(model, monitor) == len(sparse_layers)

    moe = model.model.layers[sparse_layers[0]].mlp
    generator = torch.Generator().manual_seed(0)
    moe.load_state_dict({name: torch.randn(t.shape, generator=generator) for name, t in moe.state_dict().items()})

    num_tokens = 5
    set_active_monitor(monitor)
    try:
        moe(torch.randn(1, num_tokens, config.hidden_size, generator=generator))
        counts = monitor._stack_and_reduce()
    finally:
        set_active_monitor(None)

    assert counts.shape == (len(sparse_layers), config.n_routed_experts)
    assert counts[0].sum().item() == num_tokens * config.num_experts_per_tok
