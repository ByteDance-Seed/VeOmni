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

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.distributed._tensor import Shard

from .......distributed.parallel_plan import ParallelPlan
from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem, is_dummy
from ..configuration import DeepseekV41NgramConfig
from ..modeling import DeepseekV41Ngram


def pack_ngram_inputs(
    conversations: List[List[ConversationItem]],
    device: torch.device,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(input_ids, token_mask)`` over the packed row the backbone will see.

    Both are ``[1, total]``, and *total* is the full packed length **including
    image spans** -- not the text-only length the text encoder embeds. That
    distinction is the whole reason this packer exists rather than reusing
    ``pack_input_ids``: the rows this module emits are indexed positionally by
    the backbone, so they have to cover every position the backbone has.

    Image spans contribute ``pad_token_id`` and a ``False`` mask. The id is
    never hashed: the mask turns those positions into ``DEAD``, which stops the
    lookback. Pad rather than the image placeholder token because that is
    already what a blocked lookback slot compresses to, so the two ways a
    position can be unhashable agree instead of merely both working.

    Text ids come from ``meta['input_ids']`` rather than ``value``, because by
    the time this runs the text encoder may already have replaced ``value``
    with embeddings. The template stashes the copy for exactly this.
    """
    ids: List[torch.Tensor] = []
    mask: List[torch.Tensor] = []
    for sample in conversations:
        for item in sample:
            if is_dummy(item):
                continue
            if item.type == "text":
                token_ids = item.meta.get("input_ids")
                if token_ids is None:
                    raise ValueError(
                        "deepseek_v41_ngram needs meta['input_ids'] on every text item: it hashes token ids, "
                        "and the text encoder overwrites `value` with embeddings. DeepseekV41ChatTemplate."
                        "tokenize stashes the copy -- a conversation built by another template will not have it."
                    )
                ids.append(token_ids.reshape(-1).to(device))
                mask.append(torch.ones(token_ids.numel(), dtype=torch.bool, device=device))
            else:
                # The span's length is the number of tokens vision produced for
                # it, which is why this node must run after the vision one.
                length = item.value.size(0)
                ids.append(torch.full((length,), pad_token_id, dtype=torch.long, device=device))
                mask.append(torch.zeros(length, dtype=torch.bool, device=device))
    return torch.cat(ids).unsqueeze(0), torch.cat(mask).unsqueeze(0)


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
        emb_plan = {f"tables.{index}.weight": Shard(0) for index in range(len(self.config.layer_ids))}
        return ParallelPlan(extra_parallel_plan={"emb": emb_plan})


class DeepseekV41NgramAccelerated(VeOmniMixin, DeepseekV41Ngram):
    pass


__all__ = ["DeepseekV41NgramAccelerated", "pack_ngram_inputs"]
