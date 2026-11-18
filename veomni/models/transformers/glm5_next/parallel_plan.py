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

"""ExtraParallel plans for GLM-5.3-Flash (``glm5_next``)."""

from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_parallel_plan():
    """EP plan for ``Glm5NextForConditionalGeneration`` / ``Glm5NextModel``."""
    ep_plan = {
        "model.language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.language_model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    return ParallelPlan(extra_parallel_plan={"ep": ep_plan})


def get_text_parallel_plan():
    """EP plan for a standalone ``Glm5NextTextModel``."""
    ep_plan = {
        "layers.*.mlp.experts.gate_up_proj": Shard(0),
        "layers.*.mlp.experts.down_proj": Shard(0),
    }
    return ParallelPlan(extra_parallel_plan={"ep": ep_plan})
