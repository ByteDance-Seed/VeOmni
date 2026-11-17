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

    def _merged_tokens_per_item(self, item: ConversationItem) -> int:
        """Merged visual tokens one item contributes: ``prod(grid) / merge²``.

        The tower's token domain is its *output* rows, not patch rows — the
        merger collapses ``merge²`` patches into one. Reads the grid the
        preprocessor stamped on the item, so it works for image and video alike.
        """
        grid = (item.meta or {}).get("_omni_grid") or (item.meta or {}).get("grid_thw")
        if grid is None:
            return 0
        values = grid.tolist() if isinstance(grid, torch.Tensor) else list(grid)
        t, h, w = (int(x) for x in values)
        merge_area = int(self.config.vision_config.spatial_merge_size) ** 2
        return int(t * h * w) // merge_area

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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # The graph hands a node the whole batch; this hook owns only the carrier.
        # Swallowing the rest keeps framework metadata (`ds_idx`, …) and any
        # sibling module's keys from failing the call.
        del kwargs
        self._conversation_carrier = conversation_list
        image_items = list(iter_desired_items(conversation_list, types=["image"], roles=["user"]))
        video_items = list(iter_desired_items(conversation_list, types=["video"], roles=["user"]))
        real_image_items = [it for it in image_items if not it.is_dummy]
        # Metering: this rank's own merged visual tokens, stashed BEFORE any SP
        # slice so the count does not depend on `sp_size`.
        self.metric_meter_set_seqlens(
            "forward",
            [self._merged_tokens_per_item(it) for it in (*real_image_items, *video_items) if not it.is_dummy],
        )
        dummy = not (real_image_items or video_items)
        if not dummy:
            image_items = real_image_items
        # Otherwise ``image_items`` holds only the worker-built dummy placeholders,
        # fed through the same path as real images — they carry patches +
        # ``_OMNI_GRID`` like real items. Under FSDP these run the ViT (gradient
        # anchor); without FSDP modeling.forward emits real-shaped zeros, so the
        # batch stays uniform either way.
        merge = self.config.vision_config.spatial_merge_size
        pixel_values, grid_thw, output_slots = process_qwen3vl_visual_items(
            image_items,
            video_items,
            device=self.device,
            dtype=self.dtype,
            spatial_merge_size=merge,
        )
        self._visual_output_slots = output_slots
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


class VisionMeterMixin(MetricMeterMixin):
    """Per-module training meter for the Qwen3-VL vision tower.

    `seqlens` carries **merged visual tokens** (the tower's output rows, what the
    backbone sees), but the tower's own compute runs on **raw patch rows**: the
    merger collapses `merge²` patches into one *after* the blocks. So the block
    and patch-embed terms are scaled back up by `merge²`, and the merger terms are
    the two projections that do the collapsing.
    """

    config: Qwen3VLVisionEncoderConfig

    def estimate_flops(self, seqlens: List[int]) -> float:
        cfg = self.config.vision_config
        hidden = cfg.hidden_size
        depth = cfg.depth
        merge_area = int(cfg.spatial_merge_size) ** 2
        num_heads = cfg.num_heads
        head_dim = hidden // num_heads

        # The blocks see raw patches, `merge²` per merged token.
        raw_seqlens = [max(0, int(s)) * merge_area for s in seqlens]
        raw_tokens = sum(raw_seqlens)
        raw_sq = sum(s * s for s in raw_seqlens)
        merged_tokens = sum(max(0, int(s)) for s in seqlens)

        # The vision block's MLP is a plain two-layer GELU MLP — `linear_fc1`
        # (hidden -> intermediate) then `linear_fc2` (intermediate -> hidden) —
        # not the SwiGLU of the language stack, so it carries no third
        # projection.
        mlp_n = hidden * cfg.intermediate_size * 2
        attn_linear_n = hidden * (3 * hidden + hidden)  # qkv + o
        dense_n = (mlp_n + attn_linear_n) * depth
        dense_flops = 6 * dense_n * raw_tokens
        # Quadratic term: per layer, per token pair, `head_dim * num_heads` MACs.
        attn_flops = 12 * raw_sq * head_dim * num_heads * depth

        # patch_embed (Conv3d over each patch) — one pass per raw patch.
        patch_n = hidden * cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2
        patch_flops = 6 * patch_n * raw_tokens

        # Mergers: the patch merger widens to `hidden * merge²` and keeps that
        # width, so `linear_fc1` maps `hidden*merge² -> hidden*merge²` and
        # `linear_fc2` maps `hidden*merge² -> out_hidden_size`. Both run once
        # per merged token, by the spatial merger plus one per DeepStack layer.
        merger_count = 1 + len(cfg.deepstack_visual_indexes)
        merged_width = hidden * merge_area
        fc1_n = merged_width * merged_width
        fc2_n = merged_width * cfg.out_hidden_size
        merger_flops = 6 * (fc1_n + fc2_n) * merged_tokens * merger_count

        return (dense_flops + attn_flops + patch_flops + merger_flops) / 1e12


class VeOmniMixin(BaseMixin, PackedTrainingMixin, TrainingMixin, VisionMeterMixin):
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
