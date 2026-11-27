"""Training-graph hooks for the Qwen3-Omni vision tower.

``generate()`` lives natively on :class:`~..modeling.Qwen3OmniVisionEncoder`;
this file carries the SP-aware training pre/forward/post hooks plus the FSDP
dummy-forward patch.
"""

from typing import Any, Dict, List, Optional

import torch

from .......distributed.parallel_state import get_parallel_state
from .......distributed.sequence_parallel import gather_outputs, sp_pad_and_slice
from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem, iter_desired_items
from ..configuration import Qwen3OmniVisionEncoderConfig
from ..modeling import (
    Qwen3OmniVisionEncoder,
    _VisualOutputSlot,
    build_qwen3omni_vit_metadata,
    process_qwen3omni_visual_items,
    scatter_qwen3omni_visual_embeds,
)


class TrainingMixin(TrainingModuleMixin):
    """Graph hooks for the Qwen3-Omni vision tower (training path).

    Image and video user items are encoded by the same ViT in a single forward
    (patches concatenated), so this FSDP unit runs exactly one visual forward
    per step regardless of which modality a micro-batch holds. Each item's
    merged patch tokens are written back onto its ``value``; per-item
    ``grid_thw`` (for the backbone's TMRoPE) and ``deepstack`` features (for
    interior-layer injection) ride on ``item.meta``.
    """

    config: Qwen3OmniVisionEncoderConfig
    device: torch.device
    dtype: torch.dtype

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        self._visual_output_slots: Optional[List[_VisualOutputSlot]] = None
        # Active sample's real MERGED-token count, stashed by ``forward_pre`` so
        # ``forward_post`` can drop the sp-pad tail from the all-gathered tokens.
        self._sp_own_len: Optional[int] = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # The graph hands a node the whole batch; this hook owns only the carrier.
        # Swallowing the rest keeps framework metadata and sibling keys from
        # failing the call.
        del kwargs
        self._conversation_carrier = conversation_list
        image_items = list(iter_desired_items(conversation_list, types=["image"], roles=["user"]))
        video_items = list(iter_desired_items(conversation_list, types=["video"], roles=["user"]))
        real_image_items = [it for it in image_items if not it.is_dummy]
        dummy = not (real_image_items or video_items)
        if not dummy:
            image_items = real_image_items
        # No real visual input: the worker appended one dummy placeholder per
        # sample, carrying patches + ``_OMNI_GRID`` like a real image. They are
        # already in ``image_items`` (same ``type`` / ``role`` tags), so the path
        # below covers them. Under FSDP they run the ViT (gradient anchor);
        # without FSDP ``modeling.forward`` emits real-shaped zeros, so the batch
        # stays uniform either way.
        merge = self.config.vision_config.spatial_merge_size
        pixel_values, grid_thw, output_slots = process_qwen3omni_visual_items(
            image_items,
            video_items,
            device=self.device,
            dtype=self.dtype,
            spatial_merge_size=merge,
        )
        self._visual_output_slots = output_slots
        # ``build_qwen3omni_vit_metadata`` appends the sp-pad tail segment itself
        # when SP is on, so it matches the sliced patches below.
        vit_metadata = build_qwen3omni_vit_metadata(grid_thw.tolist(), merge) if grid_thw is not None else None

        if get_parallel_state().sp_size > 1:
            # SP input-slice: hand this rank its Ulysses shard of the sample's
            # flat patch sequence. Every SP rank already holds the same patches +
            # ``grid_thw`` (the dataloader replicates each shard). Pad+slice with
            # ``pad_scale = spatial_merge_size**2`` so each chunk stays aligned to
            # merge-group boundaries; the ViT slices its own pos-embeds / cos /
            # sin internally and runs the sp-pad tail segment. ``is_dummy`` is
            # dropped: the SP path always runs the ViT (the FSDP grad anchor).
            merge_area = merge**2
            # ``Qwen3OmniMoeVisionEncoder`` hardcodes its internal SP pad_scale to 4
            # (pos-embed / cos / sin slicing). Keep the two in lockstep so a future
            # variant with a different merge size fails loudly instead of mis-slicing.
            if merge_area != 4:
                raise AssertionError(
                    "qwen3omni_vision SP requires spatial_merge_size**2 == 4 to match "
                    f"Qwen3OmniMoeVisionEncoder pad_scale, got {merge_area}."
                )
            self._sp_own_len = int(grid_thw.prod(dim=1).sum().item()) // merge_area
            pixel_values = sp_pad_and_slice(pixel_values, dim=0, pad_value=0, pad_scale=merge_area)
            return {
                "pixel_values": pixel_values,
                "image_grid_thw": grid_thw,
                "vit_metadata": vit_metadata,
            }
        # ``is_dummy`` stays for the plain (non-SP) forward: a non-SP
        # inference/eval short-circuit that skips the ViT for an all-dummy batch
        # with no anchor to produce.
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "vit_metadata": vit_metadata,
            "is_dummy": dummy,
        }

    @post_forward("forward")
    def forward_post(
        self,
        image_embeds: torch.Tensor,
        deepstack_features: List[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del image_grid_thw, kwargs
        ps = get_parallel_state()
        if ps.sp_size > 1 and ps.skip_sp_gather_between_modules:
            # Shard handoff: the merged tokens are this rank's window, so each
            # item takes its intersection with that window and the backbone
            # concatenates them straight into its own row. Nothing is gathered.
            carrier = self._conversation_carrier
            output_slots = self._visual_output_slots
            self._conversation_carrier = None
            self._visual_output_slots = None
            self._sp_own_len = None
            scatter_qwen3omni_visual_embeds(
                output_slots,
                image_embeds,
                deepstack_features,
                sp_rank=ps.sp_rank,
                sp_size=ps.sp_size,
            )
            return {"conversation_list": carrier}
        if ps.sp_size > 1:
            # SP output-gather: all-gather Ulysses shards of merged tokens (+
            # deepstack) back to the full sequence (autograd-aware; backward sums
            # grads across the SP group), then drop the sp-pad tail.
            group = ps.sp_group

            def _gather(tensor: torch.Tensor) -> torch.Tensor:
                tensor = gather_outputs(tensor, gather_dim=0, group=group)
                return tensor.narrow(0, 0, self._sp_own_len)

            image_embeds = _gather(image_embeds)
            deepstack_features = [_gather(layer) for layer in deepstack_features]
            self._sp_own_len = None

        carrier = self._conversation_carrier
        output_slots = self._visual_output_slots
        self._conversation_carrier = None
        self._visual_output_slots = None
        scatter_qwen3omni_visual_embeds(output_slots, image_embeds, deepstack_features)
        return {"conversation_list": carrier}


class VeOmniMixin(BaseMixin, TrainingMixin):
    """``generate()`` already lives on the native :class:`Qwen3OmniVisionEncoder`."""


class Qwen3OmniVisionEncoderAccelerated(VeOmniMixin, Qwen3OmniVisionEncoder):
    """Training/runtime Qwen3-Omni vision — adds the FSDP dummy-forward patch."""

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        vit_metadata: Optional[Dict[str, Any]] = None,
        is_dummy: bool = False,
    ) -> Dict[str, Any]:
        if is_dummy and not (self.training and get_parallel_state().fsdp_enabled):
            image_embeds, deepstack_features = self._dummy_outputs(pixel_values, image_grid_thw)
        else:
            image_embeds, deepstack_features = self._encode(pixel_values, image_grid_thw, vit_metadata)
        return {
            "image_embeds": image_embeds,
            "deepstack_features": deepstack_features,
            "image_grid_thw": image_grid_thw,
        }


__all__ = ["Qwen3OmniVisionEncoderAccelerated", "TrainingMixin", "VeOmniMixin"]
