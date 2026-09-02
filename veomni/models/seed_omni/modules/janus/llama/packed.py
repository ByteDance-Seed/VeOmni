"""Janus LLaMA packed training hooks — packed features in, packed hidden out."""

from typing import Any, Dict, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids, valid_seqlens_from_cu_seqlens

from ....mixins.training_module_mixin import post_forward, pre_forward
from ..packing import PACKED_HIDDEN, ensure_packed_batch_dim


class PackedTrainingMixin:
    """Packed graph node: ``pack_forward`` (AR backbone over packed embeddings)."""

    device: torch.device
    training: bool

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pack_sp_own_len: Optional[int] = None

    @pre_forward("pack_forward")
    def pack_forward_pre(
        self,
        packed_features: Optional[torch.Tensor] = None,
        packed_attention_mask: Optional[torch.Tensor] = None,
        packed_position_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if packed_features is None or packed_position_ids is None:
            raise ValueError("Janus LLaMA pack_forward: packed_features and packed_position_ids are required.")
        inputs_embeds = ensure_packed_batch_dim(packed_features).to(device=self.device, non_blocking=True)
        position_ids = packed_position_ids.to(device=self.device, non_blocking=True)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        attention_mask = packed_attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        else:
            attention_mask = attention_mask.to(device=self.device, non_blocking=True)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

        (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
            position_ids
        )
        self.metric_meter_set_seqlens(
            "pack_forward", [int(s) for s in valid_seqlens_from_cu_seqlens(cu_seq_lens_q).tolist()]
        )

        if get_parallel_state().sp_size > 1:
            group = get_parallel_state().sp_group
            self._pack_sp_own_len = inputs_embeds.size(1)
            embeds = sp_pad(inputs_embeds, dim=1, pad_value=0)
            mask = sp_pad(attention_mask, dim=1, pad_value=1)
            pids = sp_pad(position_ids, dim=1, pad_value=0)
            (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(pids)
            embeds = slice_input_tensor(embeds, dim=1, padding=False, group=group)
            pids = slice_input_tensor(pids, dim=1, padding=False, group=group)
            return dict(
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=pids,
                cu_seq_lens_q=cu_q,
                cu_seq_lens_k=cu_k,
                max_length_q=max_q,
                max_length_k=max_k,
            )
        return dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
        )

    def pack_forward(self, **kwargs: Any) -> Dict[str, Any]:
        return self.forward(**kwargs)

    @post_forward("pack_forward")
    def pack_forward_post(self, **outputs: Any) -> Dict[str, Any]:
        hidden_states = outputs.get("hidden_states")
        if get_parallel_state().sp_size > 1:
            hidden_states = gather_outputs(hidden_states, gather_dim=1, group=get_parallel_state().sp_group)
            hidden_states = hidden_states.narrow(1, 0, self._pack_sp_own_len)
        return {PACKED_HIDDEN: ensure_packed_batch_dim(hidden_states)}


__all__ = ["PackedTrainingMixin"]
