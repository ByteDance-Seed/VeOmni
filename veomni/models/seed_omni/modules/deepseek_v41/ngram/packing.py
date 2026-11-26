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

from ......utils.device import is_nvidia_cuda_device
from ....utils.conversation import ConversationItem
from ...base.llm_packing import item_token_len


def pack_ngram_inputs(
    conversations: List[List[ConversationItem]],
    device: torch.device,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(input_ids, token_mask, position_ids)`` over the packed row the backbone will see.

    All three are ``[1, total]``, and *total* is the full packed length **including
    image spans** -- not the text-only length the text encoder embeds. That
    distinction is the whole reason this packer exists rather than reusing
    ``pack_input_ids``: the rows this module emits are indexed positionally by
    the backbone, so they have to cover every position the backbone has.

    ``position_ids`` restarts at zero for each sample in ``conversations`` so
    n-gram lookback stops at sample boundaries in a packed micro-batch.

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
    positions: List[torch.Tensor] = []
    # Pinned-host staging is only worth it on an NVIDIA CUDA target; for any
    # other backend the row is built straight on the destination device.
    build_on_cpu = is_nvidia_cuda_device(device)
    target_dev = torch.device("cpu") if build_on_cpu else device
    for sample in conversations:
        sample_len = 0
        for item in sample:
            if item.is_dummy:
                continue
            if item.type == "text":
                token_ids = item.meta.get("input_ids")
                if token_ids is None:
                    raise ValueError(
                        "deepseek_v41_ngram needs meta['input_ids'] on every text item: it hashes token ids, "
                        "and the text encoder overwrites `value` with embeddings. DeepseekV41ChatTemplate."
                        "tokenize stashes the copy -- a conversation built by another template will not have it."
                    )
                length = token_ids.numel()
                flat_ids = token_ids.reshape(-1)
                ids.append(flat_ids.cpu() if build_on_cpu and flat_ids.is_cuda else flat_ids.to(target_dev))
                mask.append(torch.ones(length, dtype=torch.bool, device=target_dev))
            else:
                # The span's length is the number of tokens vision produced for
                # it, which is why this node must run after the vision one.
                #
                # Read it from the stamped metadata, not from ``item.value``: under
                # ``skip_sp_gather_between_modules`` the vision tower hands back
                # *this rank's shard*, so ``value.size(0)`` is the window width while
                # the n-gram row is indexed against the full packed sequence. Sizing
                # from the shard would shorten the position ids and the mask for
                # every media span, and the backbone rejects the short row.
                length = item_token_len(item)
                ids.append(torch.full((length,), pad_token_id, dtype=torch.long, device=target_dev))
                mask.append(torch.zeros(length, dtype=torch.bool, device=target_dev))
            positions.append(torch.arange(sample_len, sample_len + length, dtype=torch.long, device=target_dev))
            sample_len += length
    cat_ids = torch.cat(ids).unsqueeze(0)
    cat_mask = torch.cat(mask).unsqueeze(0)
    cat_pos = torch.cat(positions).unsqueeze(0)
    if build_on_cpu:
        cat_ids = cat_ids.pin_memory().to(device, non_blocking=True)
        cat_mask = cat_mask.pin_memory().to(device, non_blocking=True)
        cat_pos = cat_pos.pin_memory().to(device, non_blocking=True)
    return cat_ids, cat_mask, cat_pos


__all__ = ["pack_ngram_inputs"]
