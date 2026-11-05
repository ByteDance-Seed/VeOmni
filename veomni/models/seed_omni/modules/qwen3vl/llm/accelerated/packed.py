"""Qwen3-VL LLM packed training hooks — packed features in, packed hidden out."""

from typing import Any, Dict, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad

from .....mixins.training_module_mixin import post_forward, pre_forward
from ...packing import PACKED_HIDDEN, ensure_packed_batch_dim


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
        packed_position_ids: Optional[torch.Tensor] = None,
        packed_cu_seqlens: Optional[torch.Tensor] = None,
        packed_max_length: Optional[int] = None,
        visual_pos_mask: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if packed_features is None or packed_position_ids is None or packed_cu_seqlens is None:
            raise ValueError(
                "Qwen3-VL LLM pack_forward: packed_features, packed_position_ids, packed_cu_seqlens are required."
            )
        inputs_embeds = ensure_packed_batch_dim(packed_features).to(device=self.device, non_blocking=True)
        position_ids = packed_position_ids.to(device=self.device, non_blocking=True)
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(1)
        cu = packed_cu_seqlens.to(device=self.device, non_blocking=True)
        max_length = int(packed_max_length) if packed_max_length is not None else int((cu[1:] - cu[:-1]).max().item())
        if visual_pos_mask is None:
            visual_pos_masks = torch.zeros(1, inputs_embeds.size(1), dtype=torch.bool, device=self.device)
            deepstack = inputs_embeds.new_zeros(0, 0, inputs_embeds.size(-1))
        else:
            visual_pos_masks = visual_pos_mask.to(device=self.device, non_blocking=True)
            if visual_pos_masks.dim() == 1:
                visual_pos_masks = visual_pos_masks.unsqueeze(0)
            deepstack = deepstack_visual_embeds
            if deepstack is None:
                deepstack = inputs_embeds.new_zeros(0, 0, inputs_embeds.size(-1))
            else:
                deepstack = deepstack.to(device=self.device, non_blocking=True)

        if get_parallel_state().sp_size > 1:
            ps = get_parallel_state()
            group, sp_size, sp_rank = ps.sp_group, ps.sp_size, ps.sp_rank
            self._pack_sp_own_len = inputs_embeds.size(1)
            embeds = sp_pad(inputs_embeds, dim=1, pad_value=0)
            pids = sp_pad(position_ids, dim=2, pad_value=0)
            mask = sp_pad(visual_pos_masks, dim=1, pad_value=0)
            padded_len = embeds.size(1)
            pad_len = padded_len - self._pack_sp_own_len
            if pad_len > 0:
                tail = torch.tensor([padded_len], dtype=cu.dtype, device=cu.device)
                cu = torch.cat([cu, tail], dim=0)
                max_length = max(max_length, pad_len)
            unit = padded_len // sp_size
            n_before = int(mask[:, : unit * sp_rank].sum().item())
            embeds = slice_input_tensor(embeds, dim=1, padding=False, group=group)
            pids = slice_input_tensor(pids, dim=2, padding=False, group=group)
            mask = slice_input_tensor(mask, dim=1, padding=False, group=group)
            if deepstack is not None and deepstack.numel() > 0:
                k_local = int(mask.sum().item())
                deepstack = deepstack[:, n_before : n_before + k_local, :]
            return dict(
                inputs_embeds=embeds,
                position_ids=pids,
                visual_pos_masks=mask,
                deepstack_visual_embeds=deepstack,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=max_length,
                max_length_k=max_length,
            )
        return dict(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack,
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=max_length,
            max_length_k=max_length,
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
