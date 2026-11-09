"""Qwen3-VL vision packed training hooks — encode patches and scatter onto packed features."""

from typing import Any, Dict, List, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, sp_pad_and_slice

from .....mixins.training_module_mixin import post_forward, pre_forward
from ...packing import (
    DEEPSTACK_VISUAL_EMBEDS,
    PACKED_FEATURES,
    fold_dummy_anchor,
    masked_scatter_embeds,
    visual_row_mask,
    visual_token_count,
)
from ..modeling import build_qwen3vl_vit_metadata


class PackedTrainingMixin:
    """Packed graph node: ``pack_encode`` (ViT embeds + DeepStack → mask scatter)."""

    device: torch.device
    dtype: torch.dtype

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._packed_features: Optional[torch.Tensor] = None
        self._visual_pos_mask: Optional[torch.Tensor] = None
        self._visual_is_real: Optional[torch.Tensor] = None
        self._image_grid_thw: Optional[torch.Tensor] = None
        self._pack_sp_own_len: Optional[int] = None

    @pre_forward("pack_encode")
    def pack_encode_pre(
        self,
        packed_features: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        visual_pos_mask: Optional[torch.Tensor] = None,
        visual_is_real: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if (
            packed_features is None
            or pixel_values is None
            or image_grid_thw is None
            or visual_pos_mask is None
            or visual_is_real is None
        ):
            raise ValueError(
                "Qwen3-VL vision pack_encode: packed_features, pixel_values, image_grid_thw, "
                "visual_pos_mask and visual_is_real required."
            )
        # Metering: merged tokens for the real visuals in this packed batch, stashed
        # BEFORE the SP slice below.
        merge_area = int(self.config.vision_config.spatial_merge_size) ** 2
        real_grids = image_grid_thw.reshape(-1, 3)[visual_is_real.reshape(-1).bool()]
        self.metric_meter_set_seqlens(
            "pack_encode",
            [int(row.prod().item()) // merge_area for row in real_grids],
        )
        self._packed_features = packed_features
        self._visual_pos_mask = visual_pos_mask
        self._visual_is_real = visual_is_real
        self._image_grid_thw = image_grid_thw
        pixels = pixel_values.to(device=self.device, dtype=self.dtype, non_blocking=True)
        grid = image_grid_thw.to(device=self.device, non_blocking=True)
        merge = self.config.vision_config.spatial_merge_size
        vit_metadata = build_qwen3vl_vit_metadata(grid.tolist(), merge)
        if get_parallel_state().sp_size > 1:
            merge_area = merge**2
            assert merge_area == 4, (
                "qwen3vl_vision packed SP requires spatial_merge_size**2 == 4 to match "
                f"the patchgen ViT pad_scale, got {merge_area}."
            )
            self._pack_sp_own_len = int(grid.prod(dim=1).sum().item()) // merge_area
            pixels = sp_pad_and_slice(pixels, dim=0, pad_value=0, pad_scale=merge_area)
            return {
                "pixel_values": pixels,
                "image_grid_thw": grid,
                "vit_metadata": vit_metadata,
            }
        return {
            "pixel_values": pixels,
            "image_grid_thw": grid,
            "vit_metadata": vit_metadata,
            "is_dummy": not bool(visual_is_real.any()),
        }

    def pack_encode(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        vit_metadata: Optional[Dict[str, Any]] = None,
        is_dummy: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        return self.forward(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vit_metadata=vit_metadata,
            is_dummy=is_dummy,
        )

    @post_forward("pack_encode")
    def pack_encode_post(
        self,
        image_embeds: torch.Tensor,
        deepstack_features: List[torch.Tensor],
        image_grid_thw: torch.Tensor,
        **outputs: Any,
    ) -> Dict[str, Any]:
        del outputs
        if get_parallel_state().sp_size > 1:
            group = get_parallel_state().sp_group

            def _gather(t: torch.Tensor) -> torch.Tensor:
                t = gather_outputs(t, gather_dim=0, group=group)
                return t.narrow(0, 0, self._pack_sp_own_len)

            image_embeds = _gather(image_embeds)
            deepstack_features = [_gather(layer) for layer in deepstack_features]

        packed = self._packed_features
        mask = self._visual_pos_mask
        is_real = self._visual_is_real
        grid = self._image_grid_thw if self._image_grid_thw is not None else image_grid_thw
        merge = self.config.vision_config.spatial_merge_size
        n_real_tokens = visual_token_count(grid, merge, is_real)
        self._packed_features = None
        self._visual_pos_mask = None
        self._visual_is_real = None
        self._image_grid_thw = None

        # Select by the per-row flag, not by position: a packed batch holds every
        # sample's visuals in whatever order the collator joined them.
        real_tokens = visual_row_mask(grid, merge, is_real)
        dummy_tokens = ~real_tokens
        real_embeds = image_embeds[real_tokens]
        dummy_embeds = image_embeds[dummy_tokens]

        if n_real_tokens > 0:
            packed = masked_scatter_embeds(packed, mask, real_embeds)
        if dummy_embeds.numel() > 0:
            packed = fold_dummy_anchor(packed, dummy_embeds)
        elif n_real_tokens == 0:
            packed = fold_dummy_anchor(packed, image_embeds)

        if deepstack_features:
            stacked = torch.stack([layer[real_tokens] for layer in deepstack_features], dim=0)
            dummy_deepstack = [layer[dummy_tokens] for layer in deepstack_features]
            if dummy_deepstack[0].numel() > 0:
                packed = fold_dummy_anchor(packed, torch.stack(dummy_deepstack, dim=0))
        else:
            stacked = packed.new_zeros(0, 0, packed.size(-1))

        return {PACKED_FEATURES: packed, DEEPSTACK_VISUAL_EMBEDS: stacked}


__all__ = ["PackedTrainingMixin"]
