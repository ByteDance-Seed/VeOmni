"""
JanusVisionEncoder — OmniModule wrapping the Janus SigLIP tower + aligner.

Corresponds to JanusModel.vision_model + JanusModel.aligner in the original
JanusForConditionalGeneration checkpoint.

Connection outputs
------------------
``image_embeds``
    Float tensor of shape ``(batch, num_patches, llm_hidden_size)`` ready to
    be injected into the AR-LLM as understanding image embeddings.

Batch inputs (read from raw batch)
-----------------------------------
``pixel_values``
    Float tensor of shape ``(batch, 3, H, W)`` (normalised to SigLIP range).
``und_image_mask`` (optional)
    Bool tensor indicating which positions in ``input_ids`` are image tokens
    (used by the LLM to scatter-insert embeds).  Produced by the data
    pipeline; if absent the LLM falls back to its own placeholder detection.
"""

from typing import Any, Dict, Optional

import torch
from transformers import PretrainedConfig
from transformers.models.janus.modeling_janus import JanusVisionAlignerMLP, JanusVisionModel

from ...module import OmniModule


class JanusVisionEncoderConfig(PretrainedConfig):
    model_type = "janus_vision_encoder"

    def __init__(
        self,
        vision_config: Optional[Dict] = None,
        **kwargs,
    ):
        self.vision_config = vision_config or {}
        super().__init__(**kwargs)


class JanusVisionEncoder(OmniModule):
    """SigLIP vision tower + MLP aligner for image understanding.

    Loaded from the ``model.vision_model`` and ``model.aligner`` sub-modules
    of a ``JanusForConditionalGeneration`` checkpoint.
    """

    config_class = JanusVisionEncoderConfig
    _no_split_modules = ["JanusVisionEncoderLayer"]

    def __init__(self, config: JanusVisionEncoderConfig):
        super().__init__()
        self.config = config

        from transformers.models.janus.configuration_janus import JanusVisionConfig

        vision_cfg = JanusVisionConfig(**config.vision_config) if config.vision_config else JanusVisionConfig()
        self.vision_model = JanusVisionModel._from_config(vision_cfg)
        self.aligner = JanusVisionAlignerMLP(vision_cfg)

    # ── OmniModule interface ───────────────────────────────────────────────────

    def forward(self, pixel_values: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """Encode understanding image patches to LLM-space embeddings.

        If ``pixel_values`` is ``None`` (text-only batch), returns an empty dict
        so downstream modules can skip image injection.
        """
        if pixel_values is None:
            return {}

        # vision_model returns pooler_output of shape (B*N_images, num_patches, vision_hidden)
        vision_out = self.vision_model(pixel_values, return_dict=True)
        # pooler_output: (B, num_patches, vision_hidden)
        feats = vision_out.pooler_output if hasattr(vision_out, "pooler_output") else vision_out.last_hidden_state
        image_embeds = self.aligner(feats)  # (B, num_patches, llm_hidden)
        return {"image_embeds": image_embeds}

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    @classmethod
    def from_janus(cls, janus_model) -> "JanusVisionEncoder":
        """Extract vision encoder weights from a loaded JanusForConditionalGeneration."""

        vision_cfg = janus_model.config.vision_config
        cfg = JanusVisionEncoderConfig(vision_config=vision_cfg.to_dict() if hasattr(vision_cfg, "to_dict") else vars(vision_cfg))
        enc = cls(cfg)
        enc.vision_model.load_state_dict(janus_model.model.vision_model.state_dict())
        enc.aligner.load_state_dict(janus_model.model.aligner.state_dict())
        return enc
