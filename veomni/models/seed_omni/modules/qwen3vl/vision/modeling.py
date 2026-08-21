"""Qwen3-VL vision tower (ViT + patch merger + deepstack mergers).

``Qwen3VLVisionEncoder(InferenceMixin, OmniPreTrainedModel)`` — HF
vision stack in this file; graph hooks in ``accelerated.py``.

The ``forward`` returns two payloads consumed by the backbone:

* ``image_embeds`` — merged patch tokens ``(sum_merged_tokens, hidden)`` that
  fill the ``<|image_pad|>`` placeholder positions in the LLM sequence.
* ``deepstack_features`` — one ``(sum_merged_tokens, hidden)`` tensor per
  ``deepstack_visual_indexes`` layer; the backbone adds each into the matching
  interior decoder layer (DeepStack, https://arxiv.org/abs/2406.04334). Empty
  when ``config.disable_deepstack`` (a plain LLM backbone ignores them).

Bootstrapping a different-sized LLM (e.g. Qwen3-0.6B, hidden 1024) onto this
2048-d ViT is done **without an extra projector**: the config retargets the patch
merger's ``out_hidden_size`` directly, so the merger's own ``linear_fc2`` is the
trainable projection. Since a stock checkpoint's ``linear_fc2`` then has the wrong
shape, :class:`_MergerProjectionConverter` drops it at load so it's re-initialised.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch

from veomni.utils import logging
from veomni.utils.device import IS_NPU_AVAILABLE

from ......distributed.parallel_state import get_parallel_state
from ......models.checkpoint_tensor_loading import ConvertedCheckpointTensor
from ....omni_pretrained_model import OmniPreTrainedModel
from ....utils.conversation import ConversationItem
from .configuration import Qwen3VLVisionEncoderConfig
from .processing import (
    _OMNI_GRID,
    _SOURCE,
    Qwen3VLVisionImageProcessor,
    Qwen3VLVisionPreprocessor,
    Qwen3VLVisionVideoProcessor,
)


class _VisualOutputSlot(NamedTuple):
    """How ``forward_post`` scatters one encoded image/video back onto the carrier.

    The ViT returns all items' merged patch tokens as one flat ``(sum_merged,
    hidden)`` tensor. One slot per encoded item, in the ViT's concat order, records
    how to split + place that item's slice — the qwen3vl analog of the text
    backbone's ``_pack_inputs_embeds_shape`` (which unflattens the packed hidden
    states back into per-item segments):

    * ``item``        — the ``ConversationItem`` to receive its merged tokens.
    * ``grid``        — its ``(t, h, w)`` grid, restashed on ``item.meta`` for the
                        backbone's M-RoPE.
    * ``num_merged``  — its merged-token count, i.e. the split size along dim 0.
    """

    item: ConversationItem
    grid: torch.Tensor
    num_merged: int


def pixels_and_grid_from_items(items: list) -> Tuple[torch.Tensor, list[list[int]]]:
    """Concat CPU-preprocessed patch tensors and pop stashed grid metadata."""
    pixel_values = torch.cat([it.value for it in items], dim=0)
    grids = [it.meta.pop(_OMNI_GRID) for it in items]
    return pixel_values, grids


def process_qwen3vl_visual_items(
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

    pv_list: list[torch.Tensor] = []
    grid_rows: list[list[int]] = []
    output_slots: List[_VisualOutputSlot] = []
    merge_area = spatial_merge_size**2

    def _add(items: list) -> None:
        pixel_values, grids = pixels_and_grid_from_items(items)
        pv_list.append(pixel_values)
        for it, g in zip(items, grids):
            grid_rows.append(g)
            output_slots.append(
                _VisualOutputSlot(
                    item=it,
                    grid=torch.tensor(g, dtype=torch.long, device=device),
                    num_merged=int(g[0] * g[1] * g[2]) // merge_area,
                )
            )

    if image_items:
        _add(image_items)
    if video_items:
        _add(video_items)

    pixel_values = torch.cat(pv_list, dim=0).to(device=device, dtype=dtype)
    grid_thw = torch.tensor(grid_rows, dtype=torch.long, device=device)
    return pixel_values, grid_thw, output_slots


def build_qwen3vl_vit_metadata(
    grid_thw_list: list[list[int]],
    spatial_merge_size: int,
) -> Dict[str, Any]:
    """Host-side ViT cu_seqlens metadata (training + inference)."""
    cu: list[int] = [0]
    for t, h, w in grid_thw_list:
        frame_len = h * w
        for _ in range(t):
            cu.append(cu[-1] + frame_len)
    max_seqlen = max((c2 - c1 for c1, c2 in zip(cu, cu[1:])), default=0)
    ps = get_parallel_state()
    if ps.sp_enabled:
        merge_area = spatial_merge_size**2
        total = cu[-1]
        scale = ps.sp_size * merge_area
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


def scatter_qwen3vl_visual_embeds(
    output_slots: List[_VisualOutputSlot],
    embeds: torch.Tensor,
    deepstack_features: List[torch.Tensor],
) -> None:
    """Scatter merged ViT tokens back onto conversation items."""
    sizes = [slot.num_merged for slot in output_slots]
    embeds_split = torch.split(embeds, sizes, dim=0)
    deepstack_split = [torch.split(layer, sizes, dim=0) for layer in deepstack_features]
    for idx, slot in enumerate(output_slots):
        slot.item.value = embeds_split[idx]
        slot.item.source = _SOURCE
        slot.item.meta["grid_thw"] = slot.grid
        slot.item.meta["deepstack"] = [layer[idx] for layer in deepstack_split]


if IS_NPU_AVAILABLE:
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_npu import Qwen3VLVisionModel
else:
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import Qwen3VLVisionModel


logger = logging.get_logger(__name__)


class _MergerProjectionConverter:
    """Drop the patch-merger output projection (``merger.linear_fc2``) from the
    checkpoint when it was retargeted to a different ``out_hidden_size``.

    Its shape no longer matches the stock checkpoint, so it must be
    re-initialised (and trained) rather than loaded. Non-matching keys (and the
    deepstack mergers' ``linear_fc2``) pass through untouched; when shapes do
    match (standard Qwen3-VL load) nothing is dropped.
    """

    def __init__(self, model: "OmniPreTrainedModel"):
        self._model = model

    def can_handle(self, name: str) -> bool:
        return name.endswith(("merger.linear_fc2.weight", "merger.linear_fc2.bias")) and "deepstack" not in name

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional["ConvertedCheckpointTensor"]:
        try:
            param = self._model.get_parameter(name)
        except AttributeError:
            return ConvertedCheckpointTensor(name=name, tensor=tensor)
        if tuple(param.shape) != tuple(tensor.shape):
            logger.warning_rank0(
                f"qwen3vl_vision: re-initialising '{name}' "
                f"(checkpoint {tuple(tensor.shape)} != model {tuple(param.shape)} — retargeted merger)."
            )
            return None
        return ConvertedCheckpointTensor(name=name, tensor=tensor)

    def finalize(self) -> List["ConvertedCheckpointTensor"]:
        return []


class InferenceMixin:
    """FSM ``generate`` — reads patchified image/video items, runs the ViT, scatters
    merged tokens + DeepStack features back onto the carrier.

    Listed *before* :class:`~....omni_pretrained_model.OmniPreTrainedModel` in
    :class:`Qwen3VLVisionEncoder`'s bases so MRO resolves this concrete
    ``generate`` (there are no inference-state resets to worry about shadowing
    here — this module is stateless across FSM steps).
    """

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        image_items = [p for p in conversation_list if p.type == "image" and p.role == "user"]
        video_items = [p for p in conversation_list if p.type == "video" and p.role == "user"]
        if not image_items and not video_items:
            return {"conversation_list": conversation_list}

        # Items were patchified by the inference CPU preprocessor before the FSM
        # (patches on ``value`` + grid on ``meta``, exactly as in training), so this
        # only reads them back, runs the ViT, and scatters the merged tokens.
        merge = self.config.vision_config.spatial_merge_size
        pixel_values, grid_thw, output_slots = process_qwen3vl_visual_items(
            image_items,
            video_items,
            device=self.device,
            dtype=self.dtype,
            spatial_merge_size=merge,
        )
        vit_metadata = build_qwen3vl_vit_metadata(grid_thw.tolist(), merge)
        image_embeds, deepstack_features = self._encode(pixel_values, grid_thw, vit_metadata)
        scatter_qwen3vl_visual_embeds(output_slots, image_embeds, deepstack_features)
        return {"conversation_list": conversation_list}


class Qwen3VLVisionEncoder(InferenceMixin, OmniPreTrainedModel):
    """Qwen3-VL vision tower for image understanding."""

    config_class = Qwen3VLVisionEncoderConfig
    image_processor_class = Qwen3VLVisionImageProcessor
    video_processor_class = Qwen3VLVisionVideoProcessor
    preprocessor_class = Qwen3VLVisionPreprocessor
    base_model_prefix = "qwen3vl_vision"
    main_input_name = "pixel_values"
    _no_split_modules = ["Qwen3VLVisionBlock"]
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3VLVisionEncoderConfig):
        super().__init__(config)
        self.config = config
        # `model_config` overrides reach us through HF `from_pretrained`, which
        # applies them via post-construction `setattr` on the top-level config —
        # bypassing the vision_config derivation in Qwen3VLVisionEncoderConfig
        # `__init__`. Reconcile here (idempotent) so the merger is built at the
        # retargeted `out_hidden_size` and with deepstack disabled.
        if config.out_hidden_size is not None:
            config.vision_config.out_hidden_size = config.out_hidden_size
        if config.disable_deepstack:
            config.vision_config.deepstack_visual_indexes = []
        self.visual = Qwen3VLVisionModel._from_config(self.config.vision_config)
        self._image_processor: Optional[Any] = None
        self._video_processor: Optional[Any] = None
        self.post_init()

    @staticmethod
    def _create_checkpoint_tensor_converter(model: "OmniPreTrainedModel") -> _MergerProjectionConverter:
        return _MergerProjectionConverter(model)

    def freeze_model(self) -> None:
        """When ``config.freeze``, freeze the ViT but keep the patch merger
        trainable — the merger (with its retargeted ``linear_fc2``) is the
        projection into the LLM embedding space."""
        if self.config.freeze:
            self.visual.requires_grad_(False)
            self.visual.merger.requires_grad_(True)

    def _encode(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        vit_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        out = self.visual(pixel_values.type(self.visual.dtype), grid_thw=image_grid_thw, vit_metadata=vit_metadata)
        return out.pooler_output, list(out.deepstack_features)

    def _dummy_outputs(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Zero stand-ins shaped exactly like :meth:`_encode`' output (merged-token
        count + per-deepstack-layer features) for the non-FSDP dummy, whose ViT
        forward is skipped (no gradient anchor needed without FSDP). Emitting
        real-shaped zeros instead of ``None`` keeps the pre/post hooks branch-free."""
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
