"""Shared conversation packing for standard AR LLM backbones.

Applies to backbones that consume pre-embedded ``conversation_list`` segments
with flat 1-D ``position_ids`` and per-token ``attention_mask`` — e.g. Janus
LLaMA and Qwen3 dense/MoE.  Vision-language backbones (Qwen3-VL M-RoPE) and
packed MoT layouts (BAGEL Qwen2-MoT) keep family-specific packers in their own
``modulemixin.py`` files.

``scatter_llm_hidden_states`` is the inverse scatter step shared by all of the
above families (including Qwen3-VL) after ``forward_post`` unflattens backbone
outputs back onto ``conversation_list`` segments.

``SimpleArGenerationMixin`` is the shared no-CFG, single-KV-cache
``generate()`` used by the dense and MoE Qwen3 backbones (identical decode
loop; only ``forward`` differs between the two). Janus LLaMA keeps its own
CFG-aware ``generate()`` natively since it needs a second (unconditional)
cache.
"""

from __future__ import annotations

from typing import Any

import torch

from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids
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


class SimpleArGenerationMixin:
    """No-CFG, single-KV-cache ``generate()`` for Qwen3-style AR backbones.

    Subclasses must call ``self._past_key_values = None`` in ``__init__`` and
    provide ``self.forward(inputs_embeds=..., attention_mask=..., past_key_values=...,
    use_cache=..., **kwargs)`` returning ``{"hidden_states": ..., "past_key_values": ...}``.
    """

    _past_key_values: Any

    def reset_local_inference_state(self) -> None:
        return

    def reset_global_inference_state(self) -> None:
        self.reset_local_inference_state()
        self._past_key_values = None

    def generate(
        self,
        conversation_list: list[ConversationItem] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if self._past_key_values is None:
            inputs_embeds, attention_mask, position_ids, _ = pack_llm_conversations_for_forward(
                [conversation_list], self.device
            )
            (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
                position_ids
            )

            outputs = self.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=self._past_key_values,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                use_cache=True,
            )
            self._past_key_values = outputs["past_key_values"]

            hidden_states = outputs["hidden_states"]
            conversation_list.append(
                ConversationItem(
                    type="output",
                    value=self._tail_hidden_from_forward(hidden_states),
                    role="assistant",
                )
            )
            return {"conversation_list": conversation_list}

        tail_part = conversation_list[-1]
        assert tail_part.type == "output"

        inputs_embeds: torch.Tensor = tail_part.value[-1:].to(self.device)
        inputs_embeds = inputs_embeds.unsqueeze(0)

        outputs = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=self._past_key_values,
            use_cache=True,
        )
        self._past_key_values = outputs["past_key_values"]
        hidden_states = outputs["hidden_states"]

        conversation_list.append(
            ConversationItem(
                type="output",
                value=self._tail_hidden_from_forward(hidden_states),
                role="assistant",
            )
        )
        return {"conversation_list": conversation_list}

    @staticmethod
    def _tail_hidden_from_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
            return hidden_states[-1:].contiguous()
        return hidden_states[:, -1:, :].contiguous()
