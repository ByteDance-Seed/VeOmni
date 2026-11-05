"""Janus SigLIP packed training hooks — encode und pixels and scatter onto packed features."""

from typing import Any, Dict, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor

from .....mixins.training_module_mixin import post_forward, pre_forward
from ...packing import PACKED_FEATURES, fold_dummy_anchor, masked_scatter_embeds


class PackedTrainingMixin:
    """Packed graph node: ``pack_encode`` (SigLIP und-image embeds → mask scatter)."""

    device: torch.device
    dtype: torch.dtype

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._packed_features: Optional[torch.Tensor] = None
        self._und_image_mask: Optional[torch.Tensor] = None
        self._und_num_real: int = 0
        self._pack_sp_own_len: Optional[int] = None

    def _metric_meter_stash_pack_tokens(self, num_images: int) -> None:
        cfg = self.config.vision_config
        patches = (cfg.image_size // cfg.patch_size) ** 2
        self.metric_meter_set_seqlens("pack_encode", [patches] * num_images)

    @pre_forward("pack_encode")
    def pack_encode_pre(
        self,
        packed_features: Optional[torch.Tensor] = None,
        pixel_values_und: Optional[torch.Tensor] = None,
        und_image_mask: Optional[torch.Tensor] = None,
        und_num_real: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if packed_features is None or pixel_values_und is None or und_image_mask is None:
            raise ValueError("Janus SigLIP pack_encode: packed_features, pixel_values_und, und_image_mask required.")
        self._packed_features = packed_features
        self._und_image_mask = und_image_mask
        self._und_num_real = int(und_num_real)
        pixel_values = pixel_values_und.to(device=self.device, dtype=self.dtype, non_blocking=True)
        self._metric_meter_stash_pack_tokens(int(pixel_values.shape[0]))
        if get_parallel_state().sp_size > 1:
            self._pack_sp_own_len = pixel_values.size(0)
            pixel_values = slice_input_tensor(pixel_values, dim=0, padding=True, group=get_parallel_state().sp_group)
            return {"pixel_values": pixel_values}
        return {"pixel_values": pixel_values, "is_dummy": self._und_num_real == 0}

    def pack_encode(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        is_dummy: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        return self.forward(pixel_values=pixel_values, is_dummy=is_dummy)

    @post_forward("pack_encode")
    def pack_encode_post(self, image_embeds: torch.Tensor, **outputs: Any) -> Dict[str, Any]:
        del outputs
        if get_parallel_state().sp_size > 1:
            image_embeds = gather_outputs(image_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            image_embeds = image_embeds.narrow(0, 0, self._pack_sp_own_len)
        packed = self._packed_features
        mask = self._und_image_mask
        n_real = self._und_num_real
        self._packed_features = None
        self._und_image_mask = None
        dummy_tail = image_embeds[n_real:] if image_embeds.size(0) > n_real else None
        if n_real > 0:
            packed = masked_scatter_embeds(packed, mask, image_embeds[:n_real])
        if dummy_tail is not None and dummy_tail.numel() > 0:
            packed = fold_dummy_anchor(packed, dummy_tail)
        elif n_real == 0:
            packed = fold_dummy_anchor(packed, image_embeds)
        return {PACKED_FEATURES: packed}


__all__ = ["PackedTrainingMixin"]
