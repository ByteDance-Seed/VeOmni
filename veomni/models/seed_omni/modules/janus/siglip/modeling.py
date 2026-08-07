"""Janus SigLIP vision tower + aligner — HF-native :class:`OmniPreTrainedModel`."""

from typing import Any, Dict, List, Optional

import torch
from transformers.models.janus.modeling_janus import JanusVisionAlignerMLP, JanusVisionModel

from ....omni_pretrained_model import OmniPreTrainedModel
from ....utils.conversation import ConversationItem
from .configuration import JanusSiglipConfig
from .processing import JanusSiglipPreprocessor, JanusSiglipProcessor


class InferenceMixin:
    """FSM ``generate`` — HF ``GenerationMixin`` analog.

    Listed *before* :class:`~....omni_pretrained_model.OmniPreTrainedModel` in
    :class:`JanusSiglip`'s bases for consistency with every other module's
    native / accelerated split (this module has no reset/finalize override
    to worry about shadowing — see the sibling ``janus/llama`` and
    ``janus/vqvae`` ``modeling.py`` docstrings for the MRO rationale where it
    does matter).
    """

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        pending = [part for part in conversation_list if part.type == "image_output"]
        if pending:
            pixel_values = self._pixels_from_raw_images([part.value for part in pending])
        else:
            pending = [part for part in conversation_list if part.type == "image" and part.role == "user"]
            if not pending:
                return {"conversation_list": conversation_list}
            pixel_values = torch.stack([part.value for part in pending], dim=0).to(self.device, self.dtype)

        embeds = self._encode_pixel_values(pixel_values)
        for part, emb in zip(pending, embeds, strict=True):
            part.value = emb if emb.dim() == 2 else emb.squeeze(0)
            if part.type == "image_output":
                part.type = "image"
                assert part.role == "assistant"

        return {"conversation_list": conversation_list}

    def _pixels_from_raw_images(self, raw_images: list[Any]) -> Optional[torch.Tensor]:
        if not raw_images:
            return None
        return self._image_processor(images=raw_images, return_tensors="pt")["pixel_values"].to(
            device=self.device, dtype=self.dtype
        )


class JanusSiglip(InferenceMixin, OmniPreTrainedModel):
    """SigLIP vision tower + MLP aligner for image understanding."""

    config_class = JanusSiglipConfig
    image_processor_class = JanusSiglipProcessor
    preprocessor_class = JanusSiglipPreprocessor
    base_model_prefix = "janus_siglip"
    main_input_name = "pixel_values"
    _no_split_modules = ["JanusVisionEncoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: JanusSiglipConfig):
        super().__init__(config)
        self.config = config
        self.vision_model = JanusVisionModel(self.config.vision_config)
        self.aligner = JanusVisionAlignerMLP(self.config.vision_config)

        self._image_processor: Optional[Any] = None
        self.post_init()

    def _encode_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_out = self.vision_model(pixel_values, return_dict=True)
        return self.aligner(vision_out.last_hidden_state)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        return {"image_embeds": self._encode_pixel_values(pixel_values)}


__all__ = ["InferenceMixin", "JanusSiglip"]
