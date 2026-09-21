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

No FLOPs estimate: the hash is a handful of integer ops per token and the
gather has none worth reporting. The cost here is memory and all-to-all
bandwidth, which the backbone's meter already prices in per token. The meter is
still given this node's token count, so the two agree on what a token is.

The packing is this module's own rather than the shared AR one, and the
difference is the point. ``pack_input_ids`` collects text items only, because
that is what the text encoder embeds. The rows emitted here are indexed
positionally by the backbone against a packed row that *interleaves* image
spans, so they have to span those too -- see :func:`pack_ngram_inputs`. A row
built the text-only way would be shorter than the backbone's, which the
backbone refuses, and would hash n-grams straight across the images an image
span is supposed to interrupt.
"""

from typing import Any, Dict, List, Optional

import torch
from torch.distributed._tensor import Shard

from .......distributed.parallel_plan import ParallelPlan
from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ...quant_scheme import quantized_storage_dtype
from ..configuration import DeepseekV41NgramConfig
from ..modeling import DeepseekV41Ngram
from ..packing import pack_ngram_inputs


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks.

    Training is prefill-only over the whole packed sequence, so the decode
    cache (``history`` / ``start_pos``) is never engaged here -- see
    ``hashing.gather_lookback``. That is why this module needs no pre/post
    forward SP branch either: it emits one row per input position, and the
    backbone slices.

    The hooks pass the carrier through untouched and hand the gathered rows to
    the backbone as ``ngram_embeds``. Nothing is scattered back onto the
    conversation: the rows are not a per-segment property but a per-position
    one, and writing them onto items would make them survive into a later node
    that has no use for ~24 KB/token of memory.
    """

    config: DeepseekV41NgramConfig
    device: torch.device

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Optional[List[List[ConversationItem]]] = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[List[List[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        input_ids, token_mask = pack_ngram_inputs(conversation_list, self.device, self.config.pad_token_id)
        self._conversation_carrier = conversation_list
        self.metric_meter_set_seqlens("forward", [int(input_ids.numel())])
        return dict(input_ids=input_ids, token_mask=token_mask, **kwargs)

    @post_forward("forward")
    def forward_post(self, **outputs: Any) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        return {
            "conversation_list": conversation,
            "ngram_embeds": outputs["ngram_embeds"],
        }


class VeOmniMixin(BaseMixin, TrainingMixin):
    """Family assembly plus the table-sharding plan."""

    def get_parallel_plan(self) -> ParallelPlan:
        # fqn is module-local: the tables sit directly under ``self.tables``.
        #
        #
        # A quantized table is two tensors, both indexed by row, so both shard
        # the same way -- the lookup ships a row and its scales together and
        # dequantizes after the gather. The scale entry is added only when the
        # tables are quantized; a bf16 Embedding has no scale Parameter and a
        # plan entry that resolved to nothing would look like a stale prefix.
        quantized = quantized_storage_dtype(self.config) is not None
        emb_plan = {}
        for index in range(len(self.config.layer_ids)):
            emb_plan[f"tables.{index}.weight"] = Shard(0)
            if quantized:
                emb_plan[f"tables.{index}.weight_scale"] = Shard(0)
        return ParallelPlan(extra_parallel_plan={"emb": emb_plan})


class DeepseekV41NgramAccelerated(VeOmniMixin, DeepseekV41Ngram):
    pass


__all__ = ["DeepseekV41NgramAccelerated", "pack_ngram_inputs"]
