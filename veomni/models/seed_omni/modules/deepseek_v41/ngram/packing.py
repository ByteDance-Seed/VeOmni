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

"""Reading the conversation carrier into the rows the n-gram tables address.

Its own module rather than a function inside ``accelerated/``: this is carrier
packing, not acceleration, and both the training hooks and the FSM ``generate``
need it. Left where it was, the inference mixin importing it would close a cycle
-- ``modeling`` imports the mixin, the mixin imports ``accelerated``, and
``accelerated`` imports ``modeling`` for the class it extends.
"""

from typing import List, Tuple

import torch

from ....utils.conversation import ConversationItem, is_dummy


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


__all__ = ["pack_ngram_inputs"]
