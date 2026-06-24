"""Janus SigLIP vision tower + MLP aligner.

``JanusSiglip(JanusSiglipModuleMixin, PreTrainedModel)`` — HF vision stack in
this file; graph hooks in ``modulemixin.py``.
"""

from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel
from transformers.models.janus.modeling_janus import JanusVisionAlignerMLP, JanusVisionModel

from ......distributed.parallel_state import get_parallel_state
from .configuration import JanusSiglipConfig
from .modulemixin import JanusSiglipModuleMixin, JanusSiglipTraceMixin
from .processing import JanusSiglipProcessor


class JanusSiglip(JanusSiglipModuleMixin, JanusSiglipTraceMixin, PreTrainedModel):
    """SigLIP vision tower + MLP aligner for image understanding.

    Composes HF :class:`JanusVisionModel` + :class:`JanusVisionAlignerMLP`
    (weights split from ``JanusForConditionalGeneration`` by
    ``scripts/convert_model.py``).
    """

    config_class = JanusSiglipConfig
    image_processor_class = JanusSiglipProcessor
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

    def _dummy_image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Zero stand-in shaped exactly like :meth:`_encode_pixel_values`' output
        (same batch + patch count) for the non-FSDP dummy, whose ViT forward is
        skipped (no gradient anchor needed without FSDP). Emitting real-shaped
        zeros instead of ``None`` keeps the pre/post hooks branch-free."""
        cfg = self.config.vision_config
        b, _, h, w = pixel_values.shape
        num_patches = (h // cfg.patch_size) * (w // cfg.patch_size)
        return pixel_values.new_zeros(b, num_patches, cfg.projection_dim)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor],
        is_dummy: bool = False,
    ) -> Dict[str, Any]:
        # Real pixels → always encode. A worker-built dummy is only the training
        # FSDP gradient anchor, so it runs the ViT solely under training + FSDP;
        # otherwise (inference, or no FSDP) there is nothing to anchor, so skip the
        # forward but still emit real-shaped zeros so the output is uniform with a
        # real encode (pre/post never branch on dummy).
        if is_dummy and not (self.training and get_parallel_state().fsdp_enabled):
            image_embeds = self._dummy_image_embeds(pixel_values)
        else:
            image_embeds = self._encode_pixel_values(pixel_values)
        return {"image_embeds": image_embeds}
