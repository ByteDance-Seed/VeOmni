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

"""Training hooks for :class:`LladaMoeOmni`.

Everything the dense module does -- packing the micro-batch, counting the step's
samples from ``on_micro_batches_ready``, dividing by their global total and adding
the router term after that division -- applies unchanged, because LLaDA-MoE takes
the same sample-mean objective over the same packed row. This module therefore
inherits those hooks and adds only what the sparse backbone needs: the Expert
Parallel plan.

The inheritance is also what keeps the two paths from drifting. A second copy of
the sample-count predicate is how the dense module silently dropped
masked-nothing draws; re-deriving the packing or the divisor here would put the
same class of bug one model over.
"""

from __future__ import annotations

from torch.distributed._tensor import Shard

from ......distributed.parallel_plan import ParallelPlan
from ....mixins.base_mixin import BaseMixin
from ....mixins.inference_module_mixin import InferenceModuleMixin
from ...llada.accelerated.accelerated import TrainingMixin as LladaTrainingMixin
from ..modeling import LladaMoeOmni


class TrainingMixin(LladaTrainingMixin):
    """Dense LLaDA's training hooks, plus the Expert Parallel plan."""

    def get_parallel_plan(self) -> ParallelPlan:
        """Shard the fused expert weights over the ``ep`` group.

        The fqn is module-local and keeps the ``model.`` prefix because
        ``LLaDAMoEForCausalLM`` holds the backbone as ``self.model`` (unlike
        Qwen3-MoE, whose module is the bare ``Qwen3MoeModel``). Sharding follows
        the same layout as Qwen3-MoE -- the modeling was seeded from it -- but
        stays declared here so this family has no cross-model dependency.
        """
        ep_plan = {
            "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
            "model.layers.*.mlp.experts.down_proj": Shard(0),
        }
        return ParallelPlan(extra_parallel_plan={"ep": ep_plan})


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceModuleMixin):
    pass


class LladaMoeOmniAccelerated(VeOmniMixin, LladaMoeOmni):
    pass


__all__ = ["LladaMoeOmniAccelerated"]
