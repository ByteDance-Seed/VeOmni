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
"""Pieces shared by the LLaDA-family mask predictors.

Two things differ from an autoregressive decoder, and both have to hold under
`MainCollator` packing, where several samples share one row:

- Attention is bidirectional **within** a sample and absent **across** samples.
  The varlen flash backends get that from the ``cu_seq_lens`` the collator
  precomputes; sdpa / eager need an explicit block-diagonal mask.
- The loss is the in-place masked CE ``sum_i w_i * CE_i`` over masked positions
  (no label shift), left un-normalized for `DllmTrainer` to divide by the global
  sample count.
"""

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from transformers.masking_utils import create_bidirectional_mask

from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel.loss import SumOverShards
from ....utils.constants import IGNORE_INDEX
from ...loss_utils.dllm import dllm_masked_cross_entropy


VARLEN_ATTENTION_TYPES = (
    "flash_attention_2",
    "flash_attention_2_hub",
    "flash_attention_3",
    "flash_attention_3_hub",
    "flash_attention_4",
    "veomni_flash_attention_2",
    "veomni_flash_attention_2_hub",
    "veomni_flash_attention_3",
    "veomni_flash_attention_3_hub",
    "veomni_flash_attention_4",
)


def bidirectional_attention_mask(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    past_key_values=None,
) -> Optional[torch.Tensor]:
    """Attention mask for a mask predictor, packed or not.

    ``None`` on the varlen backends: they take the sample boundaries from
    ``cu_seq_lens_*`` and a dense mask would be ignored anyway. Otherwise a packed
    row (``position_ids`` restarting at 0 mid-row) gets a 4D additive mask that
    only lets a token see its own sample, and an unpacked batch falls through to
    `create_bidirectional_mask`, which handles padding.
    """
    if config._attn_implementation in VARLEN_ATTENTION_TYPES:
        return None
    if position_ids is not None and position_ids.dim() == 3:
        position_ids = position_ids[:, 0, :]
    if position_ids is not None and position_ids.shape == inputs_embeds.shape[:2] and past_key_values is None:
        starts = position_ids == 0
        starts[:, 0] = True
        segment_ids = starts.cumsum(dim=-1)
        if bool((segment_ids[:, -1] > 1).any()):
            allowed = segment_ids[:, :, None] == segment_ids[:, None, :]
            if attention_mask is not None and attention_mask.dim() == 2:
                allowed = allowed & attention_mask[:, None, :].bool()
            mask = torch.zeros(allowed.shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            mask.masked_fill_(~allowed, torch.finfo(inputs_embeds.dtype).min)
            return mask[:, None, :, :]
    return create_bidirectional_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
    )


def dllm_lm_head_loss(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    loss_weight: torch.Tensor,
) -> torch.Tensor:
    """Local ``sum_i w_i * CE_i`` over the masked positions only.

    The LM head runs on the masked positions rather than the whole row: the
    loss reads nothing else, and a full ``[tokens, vocab]`` float32 logit tensor
    is several GiB at LLaDA's 126k vocabulary.
    """
    if labels.shape != hidden_states.shape[:-1] or loss_weight.shape != labels.shape:
        raise ValueError(
            f"`labels` {tuple(labels.shape)} and `loss_weight` {tuple(loss_weight.shape)} must match the "
            f"hidden states' token axes {tuple(hidden_states.shape[:-1])}; a mask predictor is supervised in place."
        )
    selected = labels != IGNORE_INDEX
    logits = lm_head(hidden_states[selected])
    return dllm_masked_cross_entropy(
        logits,
        labels[selected],
        loss_weight[selected],
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


def ulysses_local_image_slots(
    image_mask: torch.Tensor, image_rows: torch.Tensor, local_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's part of a full-length ``image_mask`` and of the rows that fill it.

    ``image_rows`` fill the ``True`` slots of ``image_mask`` in row-major order. Under
    Ulysses the collator slices the text sequence but keeps ``image_mask`` and the
    pixels whole, so every rank encodes every image and keeps the rows whose slots
    land in its shard; an image cut by a shard boundary contributes to both sides.
    Gradient reaches the vision side only through the kept rows, which is the same
    per-shard contribution the text tokens make. Without Ulysses only the mask is cast to bool.
    """
    state = get_parallel_state()
    if not state.ulysses_enabled:
        return image_mask.bool(), image_rows
    if image_mask.shape[-1] != local_len * state.ulysses_size:
        raise ValueError(
            f"Under Ulysses `image_mask` must keep the full padded sequence ({local_len} x {state.ulysses_size} "
            f"tokens), got length {image_mask.shape[-1]}; it must not be sequence-sliced."
        )
    start = state.ulysses_rank * local_len
    slots = image_mask.bool()
    row_index = slots.flatten().long().cumsum(0).view_as(slots) - 1
    local_slots = slots[:, start : start + local_len]
    return local_slots, image_rows[row_index[:, start : start + local_len][local_slots]]


def ulysses_load_balancing_loss(
    gate_logits: tuple[torch.Tensor, ...],
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Switch load-balancing loss ``E * sum_e f_e * P_e`` over the whole Ulysses group's tokens.

    Each rank routes only its sequence shard, and the loss is a product of two
    token means, so averaging per-shard losses is not the unsharded loss. The
    statistics are summed over the group instead: the expert counts and token
    count carry no gradient and are plainly all-reduced, while the router
    probability sums go through `SumOverShards`. Every peer then reports the
    value ``sp_size = 1`` would, and FSDP's average over the group recovers its
    gradient.

    No padding mask, as on the packed flash path without SP: the at most
    ``ulysses_size - 1`` tokens of SP padding route like any other token.
    """
    state = get_parallel_state()
    device = gate_logits[0].device
    counts = torch.zeros(num_experts + 1, device=device, dtype=torch.float32)
    router_prob_sum = torch.zeros(num_experts, device=device, dtype=torch.float32)
    for layer_logits in gate_logits:
        probs = torch.softmax(layer_logits.to(device).float(), dim=-1)
        selected = torch.topk(probs, top_k, dim=-1).indices.reshape(-1)
        counts.scatter_add_(0, selected, torch.ones_like(selected, dtype=torch.float32))
        counts[num_experts] += layer_logits.shape[0]
        router_prob_sum = router_prob_sum + probs.sum(dim=0)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=state.ulysses_group)
    router_prob_sum = SumOverShards.apply(router_prob_sum, state.ulysses_group, state.ulysses_size)
    expert_count, num_tokens = counts[:num_experts], counts[num_experts]
    return torch.dot(expert_count, router_prob_sum) * (num_experts / (num_tokens * num_tokens))
