"""Qwen3-VL vision tower (ViT + patch merger + deepstack mergers).

``Qwen3VLVisionEncoder(Qwen3VLVisionEncoderModuleMixin, PreTrainedModel)`` — HF
vision stack in this file; graph hooks in ``modulemixin.py``.

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

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel

from veomni.utils import logging
from veomni.utils.device import IS_NPU_AVAILABLE

from ......distributed.parallel_state import get_parallel_state
from ......models.checkpoint_tensor_loading import ConvertedCheckpointTensor
from .configuration import Qwen3VLVisionEncoderConfig
from .modulemixin import Qwen3VLVisionEncoderModuleMixin
from .processing import Qwen3VLVisionImageProcessor, Qwen3VLVisionVideoProcessor


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

    def __init__(self, model: "PreTrainedModel"):
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


class Qwen3VLVisionEncoder(Qwen3VLVisionEncoderModuleMixin, PreTrainedModel):
    """Qwen3-VL vision tower for image understanding."""

    config_class = Qwen3VLVisionEncoderConfig
    image_processor_class = Qwen3VLVisionImageProcessor
    video_processor_class = Qwen3VLVisionVideoProcessor
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
    def _create_checkpoint_tensor_converter(model: "PreTrainedModel") -> _MergerProjectionConverter:
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

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        vit_metadata: Optional[Dict[str, Any]] = None,
        is_dummy: Optional[bool] = None,
    ) -> Dict[str, Any]:
        # ``is_dummy`` comes from the (worker-aware) pre_forward: True with
        # worker-built dummy zeros, False with real patches. ``None`` = eager /
        # no-worker path → derive from the absence of pixels (FSDP only).
        if is_dummy is None:
            is_dummy = pixel_values is None and get_parallel_state().fsdp_enabled
        if is_dummy and pixel_values is None:
            dummy = self.dummy_inputs()
            pixel_values, image_grid_thw, vit_metadata = (
                dummy["pixel_values"],
                dummy["image_grid_thw"],
                dummy["vit_metadata"],
            )
        image_embeds, deepstack_features = self._encode(pixel_values, image_grid_thw, vit_metadata)
        return {
            "image_embeds": image_embeds,
            "deepstack_features": deepstack_features,
            "image_grid_thw": image_grid_thw,
            "is_dummy": is_dummy,
        }
