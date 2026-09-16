"""VeOmni-accelerated Qwen3VLVisionEncoder — training-graph hooks only.

``generate()`` lives natively on :class:`~.modeling.Qwen3VLVisionEncoder` —
this file only carries the SP-aware training pre/forward/post hooks + the
FSDP dummy-forward patch.
"""

from typing import Any, Dict, List, Optional

import torch

from .......distributed.parallel_state import get_parallel_state
from .......distributed.sequence_parallel import gather_outputs, sp_pad_and_slice
from .....mixins.base_mixin import BaseMixin
from .....mixins.data_balance_mixin import DataBalanceMixin, DataBalanceSpec, ItemTensorField, ModuleStructure
from .....mixins.metric_meter_mixin import MetricMeterMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem, iter_desired_items
from ..configuration import Qwen3VLVisionEncoderConfig
from ..modeling import (
    Qwen3VLVisionEncoder,
    _VisualOutputSlot,
    build_qwen3vl_vit_metadata,
    process_qwen3vl_visual_items,
    scatter_qwen3vl_visual_embeds,
)
from ..processing import _SOURCE
from .packed import PackedTrainingMixin


class TrainingMixin(TrainingModuleMixin):
    """Graph hooks for the Qwen3-VL vision tower (training path).

    Both ``image`` and ``video`` user items are encoded by the **same** ViT in a
    single forward (image patches + video patches concatenated) so the FSDP unit
    runs exactly one visual forward per step regardless of which modality a
    micro-batch holds.  Each item's merged patch tokens are written back onto its
    ``value``; per-item ``grid_thw`` (for backbone M-RoPE) and ``deepstack``
    features (for interior-layer injection) ride on ``item.meta``.

    Video frames come pre-decoded as a :class:`VideoInputs` bundle (``item.value``)
    and go through the dedicated ``Qwen3VLVideoProcessor`` (temporal patchify);
    images use the ``Qwen2VLImageProcessor``.      Qwen3-VL has no audio modality.
    """

    config: Qwen3VLVisionEncoderConfig
    device: torch.device
    dtype: torch.dtype

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        self._visual_output_slots: Optional[List[_VisualOutputSlot]] = None
        # Active sample's real MERGED-token count, stashed by ``forward_sp_pre``
        # so ``forward_sp_post`` can drop the sp-pad tail from the all-gathered
        # merged tokens.
        self._sp_own_len: Optional[int] = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        image_items = list(iter_desired_items(conversation_list, types=["image"], roles=["user"]))
        video_items = list(iter_desired_items(conversation_list, types=["video"], roles=["user"]))
        dummy = not (image_items or video_items)
        if dummy:
            # No real visual input: feed the worker-built dummy placeholders (one
            # per sample) through the same path as real images — they carry
            # patches + ``_OMNI_GRID`` like real items. Under FSDP these run the
            # ViT (gradient anchor); without FSDP modeling.forward emits
            # real-shaped zeros, so the batch stays uniform either way.
            image_items = list(
                iter_desired_items(conversation_list, types=["image"], roles=["dummy"], sources=[_SOURCE])
            )
        merge = self.config.vision_config.spatial_merge_size
        pixel_values, grid_thw, output_slots = process_qwen3vl_visual_items(
            image_items,
            video_items,
            device=self.device,
            dtype=self.dtype,
            spatial_merge_size=merge,
        )
        self._visual_output_slots = output_slots
        if grid_thw is not None:
            pixel_values, grid_thw, balanced = self._balance_vision_inputs("forward", pixel_values, grid_thw)
            # A dummy owner can receive real items; encode the routed contents,
            # never short-circuit based on the original owner's dummy flag.
            dummy = dummy and not balanced
        # ViT metadata (cu_seqlens). ``build_qwen3vl_vit_metadata`` itself appends the sp-pad
        # tail segment when SP is enabled, so it matches the sliced patches below.
        vit_metadata = build_qwen3vl_vit_metadata(grid_thw.tolist(), merge) if grid_thw is not None else None

        if get_parallel_state().sp_size > 1:
            # SP input-slice: hand this rank its Ulysses shard of the sample's flat
            # patch sequence. Every SP rank already holds the same patches +
            # ``grid_thw`` (the dataloader replicates each shard). Pad+slice the
            # patches with ``pad_scale = spatial_merge_size**2`` (keeps each chunk
            # aligned to merge-group boundaries); the ViT slices its own pos-embeds /
            # cos / sin internally and runs the sp-pad tail segment. ``is_dummy`` is
            # dropped: the SP path is a training path that always runs the ViT (the
            # FSDP grad anchor). ``forward_post`` all-gathers the merged tokens.
            merge_area = self.config.vision_config.spatial_merge_size**2
            # The generated Qwen3-VL ViT hardcodes its internal SP pad_scale to 4
            # (pos-embed / cos / sin slicing). Keep the two in lockstep so a future
            # variant with a different merge size fails loudly instead of mis-slicing.
            assert merge_area == 4, (
                "qwen3vl_vision SP requires spatial_merge_size**2 == 4 to match the patchgen ViT pad_scale, "
                f"got {merge_area}."
            )
            # Active sample's real merged-token count (drop the sp-pad tail in the gather).
            self._sp_own_len = int(grid_thw.prod(dim=1).sum().item()) // merge_area
            pixel_values = sp_pad_and_slice(pixel_values, dim=0, pad_value=0, pad_scale=merge_area)
            return {
                "pixel_values": pixel_values,
                "image_grid_thw": grid_thw,
                "vit_metadata": vit_metadata,
            }
        # ``is_dummy`` stays for the plain (non-SP) forward: a non-SP inference/eval
        # short-circuit that skips the ViT for an all-dummy batch with no anchor.
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
        image_grid_thw: torch.Tensor,
    ) -> Dict[str, Any]:
        if get_parallel_state().sp_size > 1:
            # SP output-gather: all-gather Ulysses shards of merged tokens (+
            # deepstack) back to the full sequence (autograd-aware; backward sums
            # grads across the SP group), then drop the sp-pad tail.
            group = get_parallel_state().sp_group

            def _gather(t: torch.Tensor) -> torch.Tensor:
                t = gather_outputs(t, gather_dim=0, group=group)
                return t.narrow(0, 0, self._sp_own_len)

            image_embeds = _gather(image_embeds)
            deepstack_features = [_gather(layer) for layer in deepstack_features]

        restored = self.data_balance_post(
            "forward", {"image_embeds": image_embeds, "deepstack_features": deepstack_features}
        )
        image_embeds, deepstack_features = restored["image_embeds"], restored["deepstack_features"]
        conversation = self._conversation_carrier
        output_slots = self._visual_output_slots
        self._conversation_carrier = None
        self._visual_output_slots = None
        # forward returns merged tokens in slot order (real or dummy alike); scatter
        # them back onto the originating items.
        scatter_qwen3vl_visual_embeds(output_slots, image_embeds, deepstack_features)
        return {"conversation_list": conversation}

    def dummy_inputs(self) -> Dict[str, Any]:
        cfg = self.config.vision_config
        merge = cfg.spatial_merge_size
        t, h, w = 1, 2 * merge, 2 * merge
        pixel_row = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        pixel_values = torch.zeros(t * h * w, pixel_row, device=self.device, dtype=self.dtype)
        image_grid_thw = torch.tensor([[t, h, w]], dtype=torch.long, device=self.device)
        vit_metadata = build_qwen3vl_vit_metadata([[t, h, w]], merge)
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "vit_metadata": vit_metadata}


class BalanceMixin(DataBalanceMixin):
    """Qwen owns only its item schema; routing and inverse live on the shared mixin."""

    _vision_balance_spec = DataBalanceSpec(
        fields=(ItemTensorField("pixel_values", "patch_lengths"), ItemTensorField("image_grid_thw", "item_lengths")),
        output_names=("image_embeds", "deepstack_features"),
        output_lengths_key="merged_lengths",
        cost_lengths_key="attention_costs",
        structure=ModuleStructure(global_attention=True, non_overlapping_patchify=True),
        cost_exponent=1,
    )
    data_balance_specs = {"forward": _vision_balance_spec, "pack_encode": _vision_balance_spec}

    def _balance_vision_inputs(self, method: str, pixels: torch.Tensor, grid: torch.Tensor) -> tuple:
        # Worker-built geometry is the authoritative cost signal. Stash original
        # full lengths for token accounting before routing or SP padding/slicing.
        grid_list = grid.tolist()
        patch_lengths = [t * h * w for t, h, w in grid_list]
        merged_lengths = [n // self.config.vision_config.spatial_merge_size**2 for n in patch_lengths]
        merge_area = self.config.vision_config.spatial_merge_size**2
        # The ViT attends within each frame, not across the whole video clip.
        # Keep frame lengths for FLOPs and clip output lengths for inverse splits.
        self.metric_meter_set_seqlens(method, [h * w // merge_area for t, h, w in grid_list for _ in range(t)])
        inputs, balanced = self.data_balance_pre(
            method,
            {"pixel_values": pixels, "image_grid_thw": grid},
            {
                "patch_lengths": patch_lengths,
                "item_lengths": [1] * len(grid_list),
                "merged_lengths": merged_lengths,
                "attention_costs": [t * (h * w // merge_area) ** 2 for t, h, w in grid_list],
            },
        )
        return inputs["pixel_values"], inputs["image_grid_thw"], balanced


class MeterMixin(MetricMeterMixin):
    """Merged-token accounting and this vision tower's theoretical FLOPs only."""

    def estimate_flops(self, seqlens: list[int]) -> float:
        cfg = self.config.vision_config
        merge_area = cfg.spatial_merge_size**2
        dim = cfg.hidden_size
        patch_params = dim * cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2
        block_params = 4 * dim**2 + 2 * dim * cfg.intermediate_size
        merger_dim = dim * merge_area
        merger_params = merger_dim * (merger_dim + cfg.out_hidden_size)
        merger_count = 1 + len(cfg.deepstack_visual_indexes)
        tokens = sum(seqlens)
        # Patch/block work runs on raw tokens; mergers run on merged tokens.
        # freeze_model freezes the tower but leaves the final merger trainable.
        frozen = self.config.freeze
        tower_factor = 2 if frozen else 6
        merger_flops = merger_params * tokens * (6 + tower_factor * (merger_count - 1))
        dense = tower_factor * (patch_params + cfg.depth * block_params) * tokens * merge_area
        attention = (4 if frozen else 12) * dim * cfg.depth * sum((n * merge_area) ** 2 for n in seqlens)
        return (dense + attention + merger_flops) / 1e12


class VeOmniMixin(BaseMixin, PackedTrainingMixin, TrainingMixin, BalanceMixin, MeterMixin):
    """``generate()`` already lives on the native :class:`~.modeling.Qwen3VLVisionEncoder`
    (via its own :class:`~.modeling.InferenceMixin`), so no ``InferenceMixin`` is needed here.
    """


class Qwen3VLVisionEncoderAccelerated(VeOmniMixin, Qwen3VLVisionEncoder):
    """Training/runtime Qwen3-VL vision — FSDP dummy forward patch."""

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


__all__ = ["Qwen3VLVisionEncoderAccelerated"]
