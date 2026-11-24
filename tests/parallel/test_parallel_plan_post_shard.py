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

"""``ParallelPlan.post_shard`` runs before the every-parameter-has-a-spec check."""

import pytest
import torch.nn as nn
from torch.distributed.tensor import Replicate, Shard

from veomni.distributed.parallel_plan import ParallelPlan, SpecInfo


class _SharedStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.decoder = nn.Module()
        self.decoder.proj = self.encoder


class _RetiePlan(ParallelPlan):
    def __init__(self):
        super().__init__({"ep": {"experts.weight": Shard(0)}})
        self.calls = 0

    def post_shard(self, model):
        self.calls += 1
        for fqn, param in model.named_parameters():
            param.spec_info = SpecInfo(para_name="ep", para_fsdp_mesh=None, placement=Replicate(), fqn=fqn)


def test_parameters_without_spec_fail_the_check_by_default():
    with pytest.raises(AssertionError, match="is omitted"):
        ParallelPlan({"ep": {"experts.weight": Shard(0)}}).apply(_SharedStack(), {"ep": None})


def test_post_shard_runs_before_the_check_and_aliases_share_the_spec():
    model = _SharedStack()
    plan = _RetiePlan()

    fqn2spec_info = plan.apply(model, {"ep": None})

    assert plan.calls == 1
    assert set(fqn2spec_info) == {"encoder.weight", "decoder.proj.weight"}
    assert fqn2spec_info["decoder.proj.weight"] is fqn2spec_info["encoder.weight"]
