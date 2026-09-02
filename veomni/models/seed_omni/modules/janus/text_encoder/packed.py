"""Janus text-encoder packed training hooks (wte + causal CE).

Parallel to the conversation ``encode`` / ``decode`` hooks in
:mod:`...base.text_encoder.accelerated`. Janus-only: do not put this mixin on
the shared text-encoder base.
"""

from typing import Any, Dict, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad

from ....mixins.training_module_mixin import post_forward, pre_forward
from ..packing import (
    PACKED_FEATURES,
    ensure_packed_batch_dim,
    shift_packed_labels,
)


class PackedTrainingMixin:
    """Packed graph nodes: ``pack_encode`` (wte) and ``pack_decode`` (lm-head CE)."""

    device: torch.device

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pack_sp_own_len: Optional[int] = None

    @pre_forward("pack_encode")
    def pack_encode_pre(
        self,
        packed_input_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if packed_input_ids is None:
            raise ValueError("Janus text encoder pack_encode: packed_input_ids is required.")
        input_ids = packed_input_ids.reshape(-1).to(device=self.device, non_blocking=True)
        self.metric_meter_set_seqlens("pack_encode", [int(input_ids.numel())])
        if get_parallel_state().sp_size > 1:
            self._pack_sp_own_len = int(input_ids.size(0))
            input_ids = sp_pad(input_ids, dim=0, pad_value=0)
            input_ids = slice_input_tensor(input_ids, dim=0, padding=False, group=get_parallel_state().sp_group)
        return {"input_ids": input_ids}

    def pack_encode(self, input_ids: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
        del kwargs
        out = self.encode(input_ids=input_ids)
        embeds = out["inputs_embeds"]
        return {PACKED_FEATURES: ensure_packed_batch_dim(embeds)}

    @post_forward("pack_encode")
    def pack_encode_post(self, packed_features: torch.Tensor, **outputs: Any) -> Dict[str, Any]:
        del outputs
        if get_parallel_state().sp_size > 1:
            packed_features = gather_outputs(packed_features, gather_dim=1, group=get_parallel_state().sp_group)
            packed_features = packed_features.narrow(1, 0, self._pack_sp_own_len)
        packed_features = ensure_packed_batch_dim(packed_features)
        return {PACKED_FEATURES: packed_features}

    @pre_forward("pack_decode")
    def pack_decode_pre(
        self,
        packed_hidden: Optional[torch.Tensor] = None,
        packed_labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if packed_hidden is None or packed_labels is None:
            raise ValueError("Janus text encoder pack_decode: packed_hidden and packed_labels are required.")
        hidden = ensure_packed_batch_dim(packed_hidden)
        if hidden.dim() == 3 and hidden.size(0) == 1:
            hidden = hidden.squeeze(0)
        labels = packed_labels.reshape(-1) if packed_labels.dim() > 1 else packed_labels
        shift_labels = shift_packed_labels(labels.unsqueeze(0)).reshape(-1)
        shift_labels = shift_labels.to(device=hidden.device, non_blocking=True)
        return {"hidden_states": hidden, "shift_labels": shift_labels}

    def pack_decode(
        self,
        hidden_states: torch.Tensor,
        shift_labels: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        return self.decode(hidden_states=hidden_states, shift_labels=shift_labels)

    @post_forward("pack_decode")
    def pack_decode_post(
        self, loss: torch.Tensor, logits: torch.Tensor | None = None, **outputs: Any
    ) -> Dict[str, Any]:
        del logits, outputs
        if loss is not None:
            return {"_loss": loss}
        return {}


__all__ = ["PackedTrainingMixin"]
