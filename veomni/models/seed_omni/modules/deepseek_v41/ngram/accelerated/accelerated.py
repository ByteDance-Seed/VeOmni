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

"""Graph hooks and parallel plan for the DeepSeek-V4.1 n-gram lookup.

The two tables are 196.6B parameters -- ~85% of everything outside the MoE
experts -- so the plan that matters here is the one that splits them. They are
row-split over the ``emb`` extra-parallel group, the same shape the base text
encoder's ``embed_tokens`` and Seedream's over-encoding table use; the lookup
goes through ``AllToAllEmbedding`` in :meth:`DeepseekV41Ngram.lookup`.

No metric meter: the hash is a handful of integer ops per token and the gather
has no FLOPs worth reporting. The cost here is memory and all-to-all bandwidth,
which the backbone's meter already prices in per token.
"""

from torch.distributed._tensor import Shard

from .......distributed.parallel_plan import ParallelPlan
from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin
from ..configuration import DeepseekV41NgramConfig
from ..modeling import DeepseekV41Ngram


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks.

    Training is prefill-only over the whole packed sequence, so the decode
    cache (``history`` / ``start_pos``) is never engaged here -- see
    ``hashing.gather_lookback``. That is why this module needs no pre/post
    forward SP branch either: it emits one row per input position, and the
    backbone slices.
    """

    config: DeepseekV41NgramConfig


class VeOmniMixin(BaseMixin, TrainingMixin):
    """Family assembly plus the table-sharding plan."""

    def get_parallel_plan(self) -> ParallelPlan:
        # fqn is module-local: the tables sit directly under ``self.tables``.
        emb_plan = {f"tables.{index}.weight": Shard(0) for index in range(len(self.config.layer_ids))}
        return ParallelPlan(extra_parallel_plan={"emb": emb_plan})


class DeepseekV41NgramAccelerated(VeOmniMixin, DeepseekV41Ngram):
    pass


__all__ = ["DeepseekV41NgramAccelerated"]
