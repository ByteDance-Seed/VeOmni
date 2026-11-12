"""Shared conversation packing for standard AR LLM backbones.

Applies to backbones that consume pre-embedded ``conversation_list`` segments
with flat 1-D ``position_ids`` and per-token ``attention_mask`` — e.g. Janus
LLaMA and Qwen3 dense/MoE.  Vision-language backbones (Qwen3-VL M-RoPE) and
packed MoT layouts (BAGEL Qwen2-MoT) keep family-specific packers in their own
``modeling.py`` files.

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

from dataclasses import dataclass, field
from typing import Any

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids
from veomni.utils.tensor_utils import naflatten

from ...utils.conversation import _OMNI_TOKEN_LEN, ConversationItem


_SP_SHARDED_EMBEDS_KEY = "_veomni_sp_sharded_embeds"
_SP_SHARDED_HIDDEN_KEY = "_veomni_sp_sharded_hidden"


@dataclass
class SpShardedHiddenStates:
    """Sequence-sharded backbone hidden states passed directly to ``text_encoder.decode``."""

    hidden_states: torch.Tensor
    start: int
    end: int
    valid_len: int
    full_len: int
    extra_rows: dict[str, torch.Tensor] = field(default_factory=dict)


def _carrier_anchor_meta(conversations: list[list[ConversationItem]] | None) -> dict[str, Any] | None:
    if not conversations:
        return None
    for sample in conversations:
        for item in sample:
            return item.meta
    return None


def is_text_only_conversation(conversations: list[list[ConversationItem]] | None) -> bool:
    if not conversations:
        return False
    saw_text = False
    for sample in conversations:
        for item in sample:
            if item.is_dummy:
                continue
            if item.type != "text":
                return False
            saw_text = True
    return saw_text


def stash_sp_sharded_embeds(
    conversations: list[list[ConversationItem]] | None,
    inputs_embeds: torch.Tensor,
    valid_len: int,
) -> None:
    meta = _carrier_anchor_meta(conversations)
    if meta is not None:
        meta[_SP_SHARDED_EMBEDS_KEY] = (inputs_embeds, int(valid_len))


def pop_sp_sharded_embeds(
    conversations: list[list[ConversationItem]] | None,
) -> tuple[torch.Tensor, int] | None:
    meta = _carrier_anchor_meta(conversations)
    if meta is None:
        return None
    return meta.pop(_SP_SHARDED_EMBEDS_KEY, None)


def stash_sp_sharded_hidden_states(
    conversations: list[list[ConversationItem]] | None,
    hidden_states: torch.Tensor,
    *,
    start: int,
    end: int,
    valid_len: int,
    full_len: int,
    extra_rows: dict[str, torch.Tensor] | None = None,
) -> None:
    meta = _carrier_anchor_meta(conversations)
    if meta is not None:
        meta[_SP_SHARDED_HIDDEN_KEY] = SpShardedHiddenStates(
            hidden_states=hidden_states,
            start=int(start),
            end=int(end),
            valid_len=int(valid_len),
            full_len=int(full_len),
            extra_rows=dict(extra_rows) if extra_rows else {},
        )


def pop_sp_sharded_hidden_states(
    conversations: list[list[ConversationItem]] | None,
) -> SpShardedHiddenStates | None:
    meta = _carrier_anchor_meta(conversations)
    if meta is None:
        return None
    return meta.pop(_SP_SHARDED_HIDDEN_KEY, None)


def pack_llm_metadata_from_conversations(
    conversations: list[list[ConversationItem]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[tuple[int, int]]]:
    """Build ``(position_ids, attention_mask, sample_seqlens, seq_slices)`` without touching ``item.value`` embeds."""
    build_on_cpu = device.type != "cpu"
    target_dev = torch.device("cpu") if build_on_cpu else device
    attention_mask_chunks: list[torch.Tensor] = []
    position_ids_chunks: list[torch.Tensor] = []
    sample_seqlens: list[int] = []
    seq_slices: list[tuple[int, int]] = []
    offset = 0
    for sample in conversations:
        sample_length = 0
        for item in sample:
            if item.is_dummy:
                continue
            if isinstance(item.value, torch.Tensor):
                val = item.value.squeeze(0) if item.value.dim() == 3 else item.value
                seg_len = int(val.size(0))
            else:
                seg_len = int(item.meta.get(_OMNI_TOKEN_LEN, 0))
            chunk_mask = item.meta.get("attention_mask")
            if chunk_mask is None:
                chunk_mask = torch.ones(seg_len, dtype=torch.long, device=target_dev)
            else:
                chunk_mask = chunk_mask.to(target_dev)
            attention_mask_chunks.append(chunk_mask)
            sample_length += seg_len
        if sample_length > 0:
            sample_seqlens.append(sample_length)
            seq_slices.append((offset, offset + sample_length))
            offset += sample_length
            position_ids_chunks.append(torch.arange(sample_length, dtype=torch.long, device=target_dev))
    position_ids = torch.cat(position_ids_chunks, dim=0).unsqueeze(0)
    attention_mask = torch.cat(attention_mask_chunks, dim=0).unsqueeze(0)
    if build_on_cpu:
        position_ids = position_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
    return position_ids, attention_mask, sample_seqlens, seq_slices


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
            if item.is_dummy:
                continue
            embeds = item.value
            embeds_length = embeds.size(0)
            # Read, don't consume: packing is not the item's last reader.
            # scatter_llm_hidden_states writes hidden states back onto these same
            # items, and any second forward over one conversation (a re-forward,
            # a reference pass) would otherwise pack it with a fabricated
            # all-ones mask the second time, silently.
            chunk_attention_mask = item.meta.get("attention_mask")
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
            if part.is_dummy:
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
    # Prompt mask carried across decode steps: the cached keys include masked
    # prompt positions, so dropping it would make those keys visible again.
    _generation_attention_mask: torch.Tensor | None = None

    def reset_local_inference_state(self) -> None:
        return

    def reset_global_inference_state(self) -> None:
        self.reset_local_inference_state()
        self._past_key_values = None
        self._generation_attention_mask = None

    def generate(
        self,
        conversation_list: list[ConversationItem] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if get_parallel_state().sp_size > 1:
            raise ValueError(
                "SimpleArGenerationMixin.generate feeds the full prompt to the backbone, which conflicts "
                "with Ulysses sequence parallelism (attention expects this rank's shard); run inference "
                "with sp_size=1."
            )
        if self._past_key_values is None:
            inputs_embeds, attention_mask, position_ids, _ = pack_llm_conversations_for_forward(
                [conversation_list], self.device
            )
            (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
                position_ids
            )

            self._generation_attention_mask = attention_mask
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
            attention_mask=self._extend_generation_attention_mask(),
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

    def _extend_generation_attention_mask(self) -> torch.Tensor | None:
        """Extend the prompt mask by one valid position for each decoded token."""
        mask = self._generation_attention_mask
        if mask is None:
            return None
        mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
        self._generation_attention_mask = mask
        return mask

    @staticmethod
    def _tail_hidden_from_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
            return hidden_states[-1:].contiguous()
        return hidden_states[:, -1:, :].contiguous()
