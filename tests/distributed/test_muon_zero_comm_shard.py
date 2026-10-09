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

"""Eligibility checks behind ``muon_expert_zero_comm``'s FSDP ``Shard(0)`` layout."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed._tensor import Shard

from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.torch_parallelize import _can_shard_extra_parallel_dim0


EP_PATTERN = "model.layers.*.mlp.experts.gate_up_proj"


class _Experts(nn.Module):
    def __init__(self, num_local_experts: int, with_bias: bool = False) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(num_local_experts, 4, 6))
        if with_bias:
            # GPT-OSS ships 2D per-expert biases in the same ep plan group.
            self.gate_up_proj_bias = nn.Parameter(torch.zeros(num_local_experts, 6))


class _Mlp(nn.Module):
    def __init__(self, num_local_experts: int, with_bias: bool) -> None:
        super().__init__()
        self.experts = _Experts(num_local_experts, with_bias)


class _Layer(nn.Module):
    def __init__(self, num_local_experts: int, with_bias: bool) -> None:
        super().__init__()
        self.mlp = _Mlp(num_local_experts, with_bias)


class _Inner(nn.Module):
    def __init__(self, num_local_experts: int, with_bias: bool) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([_Layer(num_local_experts, with_bias) for _ in range(2)])


class _ToyMoe(nn.Module):
    """Parameter FQNs mirror Qwen3-MoE: ``model.layers.<i>.mlp.experts.gate_up_proj``."""

    def __init__(self, num_local_experts: int = 8, with_bias: bool = False) -> None:
        super().__init__()
        self.model = _Inner(num_local_experts, with_bias)


def _ep_plan() -> ParallelPlan:
    return ParallelPlan(extra_parallel_plan={"ep": {EP_PATTERN: Shard(0)}})


def test_glob_pattern_resolves_and_accepts_divisible_experts():
    # 8 EP-local experts over an ep_fsdp mesh of 2 -> whole experts per rank.
    assert _can_shard_extra_parallel_dim0(_ToyMoe(num_local_experts=8), _ep_plan(), "ep", 2) is True


def test_glob_pattern_rejects_indivisible_experts():
    # Regression: the plan keys are globs, so an exact-name lookup matched nothing
    # and the guard returned True for every model, enabling Shard(0) unconditionally.
    assert _can_shard_extra_parallel_dim0(_ToyMoe(num_local_experts=3), _ep_plan(), "ep", 2) is False


def test_accepts_expert_stack_alongside_two_dimensional_biases():
    # GPT-OSS mixes 3D weights and 2D per-expert biases in one ep plan. The 2D
    # entries share the expert dim, so they must not veto the whole group.
    plan = ParallelPlan(
        extra_parallel_plan={
            "ep": {EP_PATTERN: Shard(0), f"{EP_PATTERN}_bias": Shard(0)},
        }
    )
    model = _ToyMoe(num_local_experts=8, with_bias=True)
    assert _can_shard_extra_parallel_dim0(model, plan, "ep", 2) is True


def test_rejects_two_dimensional_biases_that_do_not_divide_the_mesh():
    plan = ParallelPlan(
        extra_parallel_plan={
            "ep": {EP_PATTERN: Shard(0), f"{EP_PATTERN}_bias": Shard(0)},
        }
    )
    model = _ToyMoe(num_local_experts=3, with_bias=True)
    assert _can_shard_extra_parallel_dim0(model, plan, "ep", 2) is False


def test_rejects_group_without_any_expert_stack():
    # An ``emb`` weight is 2D: Muon's zero-comm path is 3D-only, so Shard(0) buys
    # nothing and must not be forced on a non-expert ExtraParallel.
    plan = ParallelPlan(extra_parallel_plan={"emb": {"model.embed_tokens.weight": Shard(0)}})
    assert _can_shard_extra_parallel_dim0(_ToyMoe(), plan, "emb", 2) is False


def test_rejects_plan_that_slices_a_non_zero_dim():
    plan = ParallelPlan(extra_parallel_plan={"ep": {EP_PATTERN: Shard(1)}})
    assert _can_shard_extra_parallel_dim0(_ToyMoe(), plan, "ep", 2) is False


def test_rejects_plan_whose_patterns_match_nothing():
    plan = ParallelPlan(extra_parallel_plan={"ep": {"decoder.*.experts.w1": Shard(0)}})
    assert _can_shard_extra_parallel_dim0(_ToyMoe(), plan, "ep", 2) is False


def test_rejects_para_absent_from_the_plan():
    assert _can_shard_extra_parallel_dim0(_ToyMoe(), _ep_plan(), "emb", 2) is False


def test_rejects_missing_plan():
    assert _can_shard_extra_parallel_dim0(_ToyMoe(), None, "ep", 2) is False
