"""Qwen3-Omni thinker vision tower (ViT + patch merger + deepstack mergers).

``forward`` returns two payloads the backbone consumes:

* ``image_embeds`` — merged patch tokens ``(sum_merged_tokens, hidden)`` that
  fill the ``<|image_pad|>`` / ``<|video_pad|>`` placeholder positions.
* ``deepstack_features`` — one tensor per ``deepstack_visual_indexes`` layer
  (``[8, 16, 24]`` for Qwen3-Omni); the backbone adds each into the matching
  interior decoder layer (DeepStack, https://arxiv.org/abs/2406.04334).

Image and video items go through the **same** ViT in one forward, so the FSDP
unit runs exactly one visual forward per step regardless of which modality a
micro-batch holds. What separates the two is timeline bookkeeping, not weights:
video items additionally carry ``second_per_grid`` (set by the CPU
preprocessor) because the backbone's TMRoPE scales its temporal row by it.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch

from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder

from ......distributed.parallel_state import get_parallel_state
from ....utils.conversation import ConversationItem
from ...module_modeling_base import PretrainedOmniModule
from .configuration import Qwen3OmniVisionEncoderConfig
from .processing import (
    _OMNI_GRID,
    Qwen3OmniVisionImageProcessor,
    Qwen3OmniVisionPreprocessor,
    Qwen3OmniVisionVideoProcessor,
)


class _VisualOutputSlot(NamedTuple):
    """How one encoded image/video is scattered back onto the carrier.

    The ViT returns every item's merged patch tokens as a single flat
    ``(sum_merged, hidden)`` tensor, so one slot per encoded item — in the ViT's
    concat order — records how to split and place that item's slice.
    """

    item: ConversationItem
    grid: torch.Tensor
    num_merged: int


def process_qwen3omni_visual_items(
    image_items: list,
    video_items: list,
    *,
    device: torch.device,
    dtype: torch.dtype,
    spatial_merge_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[_VisualOutputSlot]]:
    """Build one ViT batch from preprocessed image/video items (training + inference)."""
    if not image_items and not video_items:
        return None, None, []

    pixel_value_list: list[torch.Tensor] = []
    grid_rows: list[list[int]] = []
    output_slots: List[_VisualOutputSlot] = []
    merge_area = spatial_merge_size**2

    def _add(items: list) -> None:
        pixel_value_list.append(torch.cat([item.value for item in items], dim=0))
        for item in items:
            grid = item.meta.pop(_OMNI_GRID)
            grid_rows.append(grid)
            output_slots.append(
                _VisualOutputSlot(
                    item=item,
                    grid=torch.tensor(grid, dtype=torch.long, device=device),
                    num_merged=int(grid[0] * grid[1] * grid[2]) // merge_area,
                )
            )

    if image_items:
        _add(image_items)
    if video_items:
        _add(video_items)

    pixel_values = torch.cat(pixel_value_list, dim=0).to(device=device, dtype=dtype)
    grid_thw = torch.tensor(grid_rows, dtype=torch.long, device=device)
    return pixel_values, grid_thw, output_slots


def build_qwen3omni_vit_metadata(
    grid_thw_list: list[list[int]],
    spatial_merge_size: int,
) -> Dict[str, Any]:
    """Host-side ViT ``cu_seqlens`` metadata (training + inference).

    One attention segment per video frame, so a clip's frames do not attend
    across each other. Under sequence parallelism an extra tail segment covers
    the padding the ViT's own input slice introduces.
    """
    cu: list[int] = [0]
    for t, h, w in grid_thw_list:
        frame_len = h * w
        for _ in range(t):
            cu.append(cu[-1] + frame_len)
    max_seqlen = max((end - start for start, end in zip(cu, cu[1:])), default=0)
    parallel_state = get_parallel_state()
    if parallel_state.sp_enabled:
        merge_area = spatial_merge_size**2
        total = cu[-1]
        scale = parallel_state.sp_size * merge_area
        padded_total = ((total + scale - 1) // scale) * scale
        sp_pad_seq_len = padded_total - total
        if sp_pad_seq_len > 0:
            cu.append(padded_total)
            max_seqlen = max(max_seqlen, sp_pad_seq_len)
    return {
        "grid_thw_list": grid_thw_list,
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device="cpu"),
        "max_seqlen": max_seqlen,
    }


def scatter_qwen3omni_visual_embeds(
    output_slots: List[_VisualOutputSlot],
    embeds: torch.Tensor,
    deepstack_features: List[torch.Tensor],
    *,
    sp_rank: int = 0,
    sp_size: int = 1,
) -> None:
    """Scatter merged ViT tokens back onto conversation items.

    ``second_per_grid`` is deliberately left untouched: the CPU preprocessor set
    it and the backbone still needs it after this.

    With ``sp_size > 1`` the tensors hold this rank's window only, so each item
    receives its intersection with that window (:func:`split_sp_sharded_tensor`)
    instead of a whole span. That is the ``skip_sp_gather_between_modules``
    handoff: the backbone concatenates these per-item slices straight into its
    own row, so nothing is gathered on the way in.
    """
    sizes = [slot.num_merged for slot in output_slots]
    if sp_size > 1:
        from ...base.sp_shard import split_sp_sharded_tensor

        embeds_split = split_sp_sharded_tensor(embeds, sizes, sp_rank, sp_size)
        deepstack_split = [split_sp_sharded_tensor(layer, sizes, sp_rank, sp_size) for layer in deepstack_features]
    else:
        embeds_split = torch.split(embeds, sizes, dim=0)
        deepstack_split = [torch.split(layer, sizes, dim=0) for layer in deepstack_features]
    for index, slot in enumerate(output_slots):
        slot.item.value = embeds_split[index]
        slot.item.meta["grid_thw"] = slot.grid
        slot.item.meta["deepstack"] = [layer[index] for layer in deepstack_split]


class InferenceMixin:
    """FSM ``generate`` — runs the ViT and scatters merged tokens + DeepStack features.

    Listed *before* :class:`~...module_modeling_base.PretrainedOmniModule` in
    :class:`Qwen3OmniVisionEncoder`'s bases so MRO resolves this concrete
    ``generate``. Nothing to reset between FSM steps — this module is stateless.
    """

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        image_items = [item for item in conversation_list if item.type == "image" and item.role == "user"]
        video_items = [item for item in conversation_list if item.type == "video" and item.role == "user"]
        if not image_items and not video_items:
            return {"conversation_list": conversation_list}

        # The inference CPU preprocessor already patchified these items before
        # the FSM (patches on ``value``, grid on ``meta``, exactly as in
        # training), so this only reads them back and runs the tower.
        merge = self.config.vision_config.spatial_merge_size
        pixel_values, grid_thw, output_slots = process_qwen3omni_visual_items(
            image_items,
            video_items,
            device=self.device,
            dtype=self.dtype,
            spatial_merge_size=merge,
        )
        vit_metadata = build_qwen3omni_vit_metadata(grid_thw.tolist(), merge)
        image_embeds, deepstack_features = self._encode(pixel_values, grid_thw, vit_metadata)
        scatter_qwen3omni_visual_embeds(output_slots, image_embeds, deepstack_features)
        return {"conversation_list": conversation_list}


class Qwen3OmniVisionEncoder(InferenceMixin, PretrainedOmniModule):
    """Qwen3-Omni thinker vision tower for image and video understanding."""

    config_class = Qwen3OmniVisionEncoderConfig
    image_processor_class = Qwen3OmniVisionImageProcessor
    video_processor_class = Qwen3OmniVisionVideoProcessor
    preprocessor_class = Qwen3OmniVisionPreprocessor
    base_model_prefix = "qwen3omni_vision"
    main_input_name = "pixel_values"
    # Merged tokens are a plain per-item concatenation of the carrier, so
    # ``forward_post`` can write this rank's window back item by item — the same
    # contract the qwen3-vl tower implements.
    supports_sp_sharded_handoff = True
    _no_split_modules = ["Qwen3OmniMoeVisionBlock"]
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3OmniVisionEncoderConfig):
        super().__init__(config)
        self.config = config
        self.visual = Qwen3OmniMoeVisionEncoder._from_config(self.config.vision_config)
        self._image_processor: Optional[Any] = None
        self._video_processor: Optional[Any] = None
        self.post_init()

    def freeze_model(self) -> None:
        """When ``config.freeze``, freeze the ViT but keep the patch merger trainable.

        The merger is the projection into the thinker's embedding space, so
        freezing it too would leave the visual pathway unable to adapt at all.
        The deepstack mergers stay frozen with the rest of the tower: they feed
        interior decoder layers that the backbone can adapt on its own side.

        ``freeze_merger`` takes even that away, which is what a round training
        something *downstream* of a frozen thinker needs: one trainable
        parameter here makes the backbone's ``inputs_embeds`` require grad, and
        a 30B backbone then retains a full activation graph nobody backpropagates.
        """
        if self.config.freeze:
            self.visual.requires_grad_(False)
            if not self.config.freeze_merger:
                self.visual.merger.requires_grad_(True)

    def _encode(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        vit_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        out = self.visual(
            hidden_states=pixel_values.type(self.visual.dtype),
            grid_thw=image_grid_thw,
            vit_metadata=vit_metadata,
        )
        return out.pooler_output, list(out.deepstack_features)

    def _dummy_outputs(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Zero stand-ins shaped exactly like :meth:`_encode`'s output.

        For the non-FSDP dummy, whose ViT forward is skipped (no gradient anchor
        is needed without FSDP). Emitting real-shaped zeros rather than ``None``
        keeps the pre/post hooks branch-free.
        """
        cfg = self.config.vision_config
        merge_area = cfg.spatial_merge_size**2
        num_merged = int(image_grid_thw.prod(dim=1).sum().item()) // merge_area
        image_embeds = pixel_values.new_zeros(num_merged, cfg.out_hidden_size)
        deepstack_features = [
            pixel_values.new_zeros(num_merged, cfg.out_hidden_size) for _ in cfg.deepstack_visual_indexes
        ]
        return image_embeds, deepstack_features

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        vit_metadata: Optional[Dict[str, Any]] = None,
        is_dummy: bool = False,
    ) -> Dict[str, Any]:
        del is_dummy
        image_embeds, deepstack_features = self._encode(pixel_values, image_grid_thw, vit_metadata)
        return {
            "image_embeds": image_embeds,
            "deepstack_features": deepstack_features,
            "image_grid_thw": image_grid_thw,
        }


__all__ = [
    "Qwen3OmniVisionEncoder",
    "_VisualOutputSlot",
    "build_qwen3omni_vit_metadata",
    "process_qwen3omni_visual_items",
    "scatter_qwen3omni_visual_embeds",
]
