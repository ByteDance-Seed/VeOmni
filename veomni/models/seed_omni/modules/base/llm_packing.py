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

from collections.abc import Sequence
from typing import Any

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids
from veomni.utils.tensor_utils import naflatten

from ...utils.conversation import _OMNI_TOKEN_LEN, ConversationItem, iter_desired_items


def item_token_len(item: ConversationItem) -> int:
    """Return the unsharded sequence-token length of ``item``."""
    if _OMNI_TOKEN_LEN in item.meta:
        return int(item.meta[_OMNI_TOKEN_LEN])
    if item.type == "text" and "labels" in item.meta:
        return int(item.meta["labels"].numel())
    if isinstance(item.value, torch.Tensor):
        val = item.value.squeeze(0) if item.value.dim() == 3 else item.value
        return int(val.size(0))
    return 1


def mask_sp_pad_tail(
    local_tensor: torch.Tensor,
    valid_len: int,
    sp_rank: int,
    sp_size: int,
) -> torch.Tensor:
    """Zero-mask any SP-padded tail rows in ``local_tensor`` while keeping autograd connected."""
    full_len = valid_len + (-valid_len % sp_size)
    local_len = full_len // sp_size
    start = sp_rank * local_len
    if start + local_len <= valid_len:
        return local_tensor
    valid_in_shard = max(0, valid_len - start)
    if valid_in_shard == 0:
        return local_tensor * 0.0
    return torch.cat(
        [
            local_tensor[:valid_in_shard],
            local_tensor[valid_in_shard:] * 0.0,
        ],
        dim=0,
    )


def split_sp_sharded_tensor(
    local_tensor: torch.Tensor,
    lengths: Sequence[int],
    sp_rank: int,
    sp_size: int,
) -> list[torch.Tensor]:
    """Split this rank's ``[local_len, ...]`` SP shard into per-segment slices matching ``lengths``.

    Any SP-padding tail rows at ``[sum(lengths), full_len)`` stay attached to
    the last segment so concatenating the returned slices on any rank
    reconstructs ``local_tensor`` verbatim.
    """
    if not lengths:
        return []
    valid_len = sum(int(x) for x in lengths)
    full_len = valid_len + (-valid_len % sp_size)
    local_len = full_len // sp_size
    if local_tensor.size(0) != local_len:
        raise ValueError(
            f"split_sp_sharded_tensor: expected local_tensor dim 0 == {local_len} "
            f"(valid_len={valid_len}, sp_size={sp_size}), got {local_tensor.size(0)}."
        )
    shard_start = sp_rank * local_len
    out: list[torch.Tensor] = []
    seg_start = 0
    num_segs = len(lengths)
    for idx, length in enumerate(lengths):
        seg_end = full_len if idx == num_segs - 1 else seg_start + int(length)
        lo = min(max(seg_start - shard_start, 0), local_len)
        hi = min(max(seg_end - shard_start, 0), local_len)
        out.append(local_tensor[lo:hi])
        seg_start += int(length)
    return out


def gather_sp_sharded_items(
    items: Sequence[ConversationItem],
    sp_group: Any,
    *,
    meta_list_keys: Sequence[str] = (),
) -> None:
    """All-gather SP-sharded ``item.value`` tensors across ``sp_group`` and restore full per-item slices."""
    if not items:
        return
    lengths = [item_token_len(it) for it in items]
    valid_len = sum(lengths)
    local_cat = torch.cat(
        [it.value.squeeze(0) if it.value.dim() == 3 else it.value for it in items],
        dim=0,
    )
    ps = get_parallel_state()
    local_len = (valid_len + (-valid_len % ps.sp_size)) // ps.sp_size if ps.sp_size > 1 else valid_len
    if local_cat.size(0) != local_len:
        return
    full_cat = gather_outputs(local_cat, gather_dim=0, group=sp_group).narrow(0, 0, valid_len)
    for it, chunk in zip(items, torch.split(full_cat, lengths, dim=0), strict=True):
        it.value = chunk

    for key in meta_list_keys:
        first_list = items[0].meta.get(key)
        if not first_list:
            continue
        num_layers = len(first_list)
        per_layer_splits = []
        for layer_idx in range(num_layers):
            local_layer = torch.cat([it.meta[key][layer_idx] for it in items], dim=0)
            full_layer = gather_outputs(local_layer, gather_dim=0, group=sp_group).narrow(0, 0, valid_len)
            per_layer_splits.append(torch.split(full_layer, lengths, dim=0))
        for idx, it in enumerate(items):
            it.meta[key] = [per_layer_splits[layer_idx][idx] for layer_idx in range(num_layers)]


def gather_sp_sharded_modalities_for_interleaving(
    conversations: list[list[ConversationItem]] | None,
    sp_group: Any,
) -> None:
    """All-gather per-modality SP-sharded ``item.value``s so a multimodal backbone can interleave them."""
    if not conversations:
        return
    text_items = [item for sample in conversations for item in sample if not item.is_dummy and item.type == "text"]
    if text_items and all(isinstance(it.value, torch.Tensor) and it.value.dim() >= 2 for it in text_items):
        gather_sp_sharded_items(text_items, sp_group)

    vis_items = [
        it
        for it in (
            *iter_desired_items(conversations, types=["image"], roles=["user"]),
            *iter_desired_items(conversations, types=["video"], roles=["user"]),
        )
        if not it.is_dummy
    ]
    if vis_items and all(isinstance(it.value, torch.Tensor) and it.value.dim() >= 2 for it in vis_items):
        gather_sp_sharded_items(vis_items, sp_group, meta_list_keys=("deepstack",))


def can_use_sp_sharded_embeds_directly(conversations: list[list[ConversationItem]] | None) -> bool:
    """True when non-dummy ``item.value`` shards already partition the packed backbone sequence without interleaving."""
    if not conversations:
        return False
    ps = get_parallel_state()
    if ps.sp_size <= 1 or not ps.skip_sp_gather_between_modules:
        return False
    non_dummy_items = [item for sample in conversations for item in sample if not item.is_dummy]
    if not non_dummy_items:
        return False
    non_dummy_types = {item.type for item in non_dummy_items}
    if non_dummy_types not in ({"text"}, {"image"}, {"video"}):
        return False
    if not all(isinstance(it.value, torch.Tensor) and it.value.dim() >= 2 for it in non_dummy_items):
        return False
    valid_len = sum(item_token_len(it) for it in non_dummy_items)
    local_len = (valid_len + (-valid_len % ps.sp_size)) // ps.sp_size
    actual_rows = sum(
        int((it.value.squeeze(0) if it.value.dim() == 3 else it.value).size(0)) for it in non_dummy_items
    )
    return actual_rows == local_len


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


def pack_llm_metadata_from_conversations(
    conversations: list[list[ConversationItem]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[tuple[int, int]]]:
    """Build ``(position_ids, attention_mask, sample_seqlens, seq_slices)`` from unsharded item lengths."""
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
            seg_len = item_token_len(item)
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


def pack_sp_sharded_conversations_for_forward(
    conversations: list[list[ConversationItem]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, list[int]]:
    """Pack SP-sharded ``item.value`` segments directly for single-modality AR backbone forward."""
    local_embeds_list: list[torch.Tensor] = []
    prefix_rows: list[torch.Tensor] = []
    for sample in conversations:
        for item in sample:
            if item.is_dummy:
                continue
            val = item.value.squeeze(0) if item.value.dim() == 3 else item.value
            local_embeds_list.append(val.to(device))
            prefix_rows.append(torch.tensor([item_token_len(item)], dtype=torch.long, device="cpu"))

    local_embeds = torch.cat(local_embeds_list, dim=0).unsqueeze(0)
    inputs_embeds_shape = torch.stack(prefix_rows)
    position_ids, attention_mask, sample_seqlens, _ = pack_llm_metadata_from_conversations(conversations, device)
    return local_embeds, attention_mask, position_ids, inputs_embeds_shape, sample_seqlens


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
