"""Janus VQVAE packed training hooks — encode gen pixels, scatter, VQ CE via mask."""

from typing import Any, Dict, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor
from veomni.utils.constants import IGNORE_INDEX

from ....mixins.training_module_mixin import post_forward, pre_forward
from ..packing import (
    PACKED_FEATURES,
    VQ_TOKEN_IDS,
    fold_dummy_anchor,
    masked_scatter_embeds,
    teacher_force_vq_hidden,
)


class PackedTrainingMixin:
    """Packed graph nodes: ``pack_encode`` (VQ embeds) and ``pack_decode`` (VQ CE)."""

    device: torch.device
    dtype: torch.dtype

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._packed_features: Optional[torch.Tensor] = None
        self._gen_image_mask: Optional[torch.Tensor] = None
        self._gen_num_real: int = 0
        self._pack_sp_own_len: Optional[int] = None

    def _metric_meter_stash_pack_tokens(self, num_images: int) -> None:
        self.metric_meter_set_seqlens("pack_encode", [int(self._image_processor.num_image_tokens)] * num_images)

    @pre_forward("pack_encode")
    def pack_encode_pre(
        self,
        packed_features: Optional[torch.Tensor] = None,
        pixel_values_gen: Optional[torch.Tensor] = None,
        gen_image_mask: Optional[torch.Tensor] = None,
        gen_num_real: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if packed_features is None or pixel_values_gen is None or gen_image_mask is None:
            raise ValueError("Janus VQVAE pack_encode: packed_features, pixel_values_gen, gen_image_mask required.")
        self._packed_features = packed_features
        self._gen_image_mask = gen_image_mask
        self._gen_num_real = int(gen_num_real)
        pixel_values = pixel_values_gen.to(device=self.device, dtype=self.dtype, non_blocking=True)
        self._metric_meter_stash_pack_tokens(int(pixel_values.shape[0]))
        if get_parallel_state().sp_size > 1:
            self._pack_sp_own_len = pixel_values.size(0)
            pixel_values = slice_input_tensor(pixel_values, dim=0, padding=True, group=get_parallel_state().sp_group)
            return {"pixel_values": pixel_values}
        return {"pixel_values": pixel_values, "is_dummy": self._gen_num_real == 0}

    def pack_encode(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        is_dummy: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        return self.encode(pixel_values=pixel_values, is_dummy=is_dummy)

    @post_forward("pack_encode")
    def pack_encode_post(
        self,
        image_embeds: torch.Tensor,
        vq_token_ids: torch.Tensor,
        **outputs: Any,
    ) -> Dict[str, Any]:
        del outputs
        if get_parallel_state().sp_size > 1:
            image_embeds = gather_outputs(image_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            vq_token_ids = gather_outputs(vq_token_ids, gather_dim=0, group=get_parallel_state().sp_group)
            image_embeds = image_embeds.narrow(0, 0, self._pack_sp_own_len)
            vq_token_ids = vq_token_ids.narrow(0, 0, self._pack_sp_own_len)
        packed = self._packed_features
        mask = self._gen_image_mask
        n_real = self._gen_num_real
        self._packed_features = None
        self._gen_image_mask = None
        dummy_tail = image_embeds[n_real:] if image_embeds.size(0) > n_real else None
        if n_real > 0:
            packed = masked_scatter_embeds(packed, mask, image_embeds[:n_real])
            vq_token_ids = vq_token_ids[:n_real]
        if dummy_tail is not None and dummy_tail.numel() > 0:
            packed = fold_dummy_anchor(packed, dummy_tail)
        elif n_real == 0:
            packed = fold_dummy_anchor(packed, image_embeds)
            token_width = vq_token_ids.size(-1) if vq_token_ids.dim() > 1 else 0
            vq_token_ids = vq_token_ids.new_empty(0, token_width)
        return {PACKED_FEATURES: packed, VQ_TOKEN_IDS: vq_token_ids.to(dtype=torch.long)}

    @pre_forward("pack_decode")
    def pack_decode_pre(
        self,
        packed_hidden: Optional[torch.Tensor] = None,
        vq_token_ids: Optional[torch.Tensor] = None,
        gen_image_mask: Optional[torch.Tensor] = None,
        gen_num_real: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if packed_hidden is None or gen_image_mask is None:
            raise ValueError("Janus VQVAE pack_decode: packed_hidden and gen_image_mask are required.")
        n_real = int(gen_num_real)
        hidden = packed_hidden
        if n_real > 0:
            if vq_token_ids is None:
                raise ValueError("Janus VQVAE pack_decode: vq_token_ids required when gen_num_real > 0.")
            selected = teacher_force_vq_hidden(hidden, gen_image_mask)
            labels = vq_token_ids.reshape(-1).to(device=selected.device, dtype=torch.long)
            return {"hidden_states": selected, "labels": labels, "is_dummy": False}

        # All-dummy span: still run generation_head (FSDP anchor) with -100 labels.
        dummy_hidden = hidden.reshape(-1, hidden.size(-1))[:1]
        dummy_hidden = fold_dummy_anchor(dummy_hidden, dummy_hidden)
        labels = torch.full((dummy_hidden.size(0),), IGNORE_INDEX, dtype=torch.long, device=dummy_hidden.device)
        return {"hidden_states": dummy_hidden.unsqueeze(0), "labels": labels.unsqueeze(0), "is_dummy": True}

    def pack_decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_dummy: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        return self.decode(hidden_states=hidden_states, labels=labels, is_dummy=is_dummy)

    @post_forward("pack_decode")
    def pack_decode_post(self, **outputs: Any) -> Dict[str, Any]:
        loss = outputs.pop("loss", None)
        if loss is not None:
            outputs["_loss"] = loss
        return outputs


__all__ = ["PackedTrainingMixin"]
