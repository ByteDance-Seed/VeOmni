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

"""Unit tests for the ``muon_expert_zero_comm`` dim-0 divisibility gate."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed._tensor import Shard

from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.torch_parallelize import _check_extra_parallel_dim0_divisibility


EP_PATTERNS = {
    "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
    "model.layers.*.mlp.experts.down_proj": Shard(0),
}


class _Experts(nn.Module):
    def __init__(self, num_local_experts: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(num_local_experts, 8, 4))
        self.down_proj = nn.Parameter(torch.zeros(num_local_experts, 4, 8))


class _Mlp(nn.Module):
    def __init__(self, num_local_experts: int):
        super().__init__()
        self.experts = _Experts(num_local_experts)


class _Layer(nn.Module):
    def __init__(self, num_local_experts: int):
        super().__init__()
        self.mlp = _Mlp(num_local_experts)


class _Inner(nn.Module):
    def __init__(self, num_local_experts: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(num_local_experts) for _ in range(num_layers)])


class _ToyMoe(nn.Module):
    """Parameter names mirror the real plan: ``model.layers.<i>.mlp.experts.*``."""

    def __init__(self, num_local_experts: int, plan: dict | None = None):
        super().__init__()
        self.model = _Inner(num_local_experts)
        self._plan = ParallelPlan(extra_parallel_plan={"ep": dict(EP_PATTERNS if plan is None else plan)})

    def get_parallel_plan(self) -> ParallelPlan:
        return self._plan


def test_wildcard_patterns_are_matched_not_looked_up():
    """Plan keys are FQN patterns; a literal dict lookup never matches them."""
    model = _ToyMoe(num_local_experts=8)

    # A literal lookup finds nothing, which is what used to make this gate
    # return True without inspecting a single parameter.
    assert dict(model.named_parameters()).get("model.layers.*.mlp.experts.gate_up_proj") is None

    assert _check_extra_parallel_dim0_divisibility(model, "ep", ep_fsdp_size=4) is True


def test_indivisible_dim0_is_rejected():
    model = _ToyMoe(num_local_experts=6)
    assert _check_extra_parallel_dim0_divisibility(model, "ep", ep_fsdp_size=4) is False


def test_no_matching_param_is_rejected():
    """An unverifiable plan must keep the safe Shard(1) layout."""
    model = _ToyMoe(num_local_experts=8, plan={"model.layers.*.mlp.nonexistent": Shard(0)})
    assert _check_extra_parallel_dim0_divisibility(model, "ep", ep_fsdp_size=4) is False


def test_unknown_extra_parallel_name_is_rejected():
    model = _ToyMoe(num_local_experts=8)
    assert _check_extra_parallel_dim0_divisibility(model, "emb", ep_fsdp_size=4) is False
