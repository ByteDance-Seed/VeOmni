"""
JanusSiglip — Janus' SigLIP vision tower + MLP aligner as one OmniModule.

Mixin form: ``class JanusSiglip(OmniModule, PreTrainedModel)``.  The vision
stack reuses HuggingFace :class:`~transformers.JanusVisionModel` and
:class:`~transformers.JanusVisionAlignerMLP` — the same pair wired inside
``JanusForConditionalGeneration`` (``aligner(vision_model(pixel_values))``).

Call-site split (V2)
--------------------
* :meth:`pre_forward` — pull raw ``(C,H,W)`` images from ``conversation_list``
  → ``pixel_values``; stash the carrier for :meth:`post_forward`.
* :meth:`forward` — pure encoder: ``pixel_values`` → ``image_embeds`` (HF path).
* :meth:`post_forward` — write ``image_embeds`` back onto ``conversation_list``
  ``type="image"`` items in place.
"""

from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel
from transformers.models.janus.modeling_janus import JanusVisionAlignerMLP, JanusVisionModel

from ......distributed.parallel_state import get_parallel_state
from .configuration import JanusSiglipConfig
from .modulemixin import JanusSiglipModuleMixin
from .processing import JanusSiglipProcessor


class JanusSiglip(JanusSiglipModuleMixin, PreTrainedModel):
    """SigLIP vision tower + MLP aligner for image understanding.

    Composes HF :class:`JanusVisionModel` + :class:`JanusVisionAlignerMLP`
    (weights split from ``JanusForConditionalGeneration`` by
    ``scripts/split_janus.py``).
    """

    config_class = JanusSiglipConfig
    processor_class = JanusSiglipProcessor
    base_model_prefix = "janus_siglip"
    main_input_name = "pixel_values"
    _no_split_modules = ["JanusVisionEncoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: JanusSiglipConfig):
        super().__init__(config)
        self.config = config
        self.vision_model = JanusVisionModel(self.config.vision_config)
        self.aligner = JanusVisionAlignerMLP(self.config.vision_config)

        self._processor: Optional[Any] = None
        self.post_init()

    def _encode_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_out = self.vision_model(pixel_values, return_dict=True)
        return self.aligner(vision_out.last_hidden_state)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if pixel_values is None and get_parallel_state().fsdp_enabled:
            return {"image_embeds": self._encode_pixel_values(**self.dummy_inputs()), "is_dummy": True}
        return {"image_embeds": self._encode_pixel_values(pixel_values)}
