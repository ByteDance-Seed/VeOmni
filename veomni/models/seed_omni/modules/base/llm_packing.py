"""Shared conversation packing for standard AR LLM backbones.

Applies to backbones that consume pre-embedded ``conversation_list`` segments
with flat 1-D ``position_ids`` and per-token ``attention_mask`` — e.g. Janus
LLaMA and Qwen3 dense/MoE.  Vision-language backbones (Qwen3-VL M-RoPE) and
packed MoT layouts (BAGEL Qwen2-MoT) keep family-specific packers in their own
``modulemixin.py`` files.

``scatter_llm_hidden_states`` is the inverse scatter step shared by all of the
above families (including Qwen3-VL) after ``forward_post`` unflattens backbone
outputs back onto ``conversation_list`` segments.
"""

from __future__ import annotations

import torch

from veomni.utils.tensor_utils import naflatten

from ...utils.conversation import ConversationItem, is_dummy


def pack_llm_conversations_for_forward(
    conversations: list[list[ConversationItem]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack conversation embed segments for AR backbone forward (training + inference).

    Returns ``(inputs_embeds, attention_mask, position_ids, inputs_embeds_shape)``
    with batch dim 0 when the packed sample is a single varlen sequence.
    """
    inputs_embeds_list = []
    attention_mask = []
    position_ids = []
    for sample in conversations:
        sample_lengths = 0
        for item in sample:
            if item.role == "dummy":
                continue
            embeds = item.value
            embeds_length = embeds.size(0)
            chunk_attention_mask = item.meta.pop("attention_mask", None)
            if chunk_attention_mask is None:
                chunk_attention_mask = torch.ones(embeds_length, dtype=torch.long, device=device)
            inputs_embeds_list.append(embeds.to(device))
            attention_mask.append(chunk_attention_mask.to(device))
            sample_lengths += embeds_length
        sample_position_ids = torch.arange(sample_lengths, dtype=torch.long, device=device)
        position_ids.append(sample_position_ids)
    inputs_embeds, inputs_embeds_shape = naflatten(inputs_embeds_list)
    position_ids = torch.cat(position_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    if inputs_embeds.dim() == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    return inputs_embeds, attention_mask, position_ids, inputs_embeds_shape


def scatter_llm_hidden_states(
    conversation_list: list[list[ConversationItem]],
    hidden_states_list: list[torch.Tensor],
) -> None:
    """Write unflattened backbone hidden states back onto non-dummy conversation segments."""
    hidden_states_list_iter = iter(hidden_states_list)
    for sample in conversation_list:
        for part in sample:
            if is_dummy(part):
                continue
            part.value = next(hidden_states_list_iter)
    if next(hidden_states_list_iter, None) is not None:
        raise RuntimeError("scatter_llm_hidden_states: segment count exceeds non-dummy conversation items.")
