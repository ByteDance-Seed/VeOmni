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

    def pre_forward(self, pixel_values: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """Packing: flatten per-sample multi-image tensors.

        Accepts ``pixel_values`` in two shapes:

        * ``(B, C, H, W)`` — one image per sample (already flat), pass through.
        * ``(B, N_images, C, H, W)`` — N images per sample; flatten to
          ``(B * N_images, C, H, W)`` so the vision backbone sees a single
          batch dimension.  The original ``(B, N_images)`` shape is stored in
          ``_pv_shape`` for potential use in ``post_forward``.

        All other kwargs are passed through unchanged.
        """
        if pixel_values is not None and pixel_values.ndim == 5:
            # (B, N_images, C, H, W) → (B*N_images, C, H, W)
            b, n = pixel_values.shape[:2]
            pixel_values = pixel_values.reshape(b * n, *pixel_values.shape[2:])
            kwargs["_pv_batch_n_images"] = (b, n)
        return dict(pixel_values=pixel_values, **kwargs)

    def forward(self, pixel_values: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """Encode understanding image patches to LLM-space embeddings.

        If ``pixel_values`` is ``None`` (text-only batch), returns an empty dict
        so downstream modules can skip image injection.
        """
        if pixel_values is None:
            return {}

        # pixel_values: (B_flat, C, H, W) — already flattened by pre_forward
        vision_out = self.vision_model(pixel_values, return_dict=True)
        # last_hidden_state: (B_flat, num_patches, vision_hidden) — per-patch features
        feats = vision_out.last_hidden_state
        # feats: (B_flat, num_patches, vision_hidden)
        image_embeds = self.aligner(feats)  # (B_flat, num_patches, llm_hidden)

        # Reshape back to (B, N_images * num_patches, llm_hidden) if multi-image packing was used
        b_n = kwargs.pop("_pv_batch_n_images", None)
        if b_n is not None:
            b, n = b_n
            p = image_embeds.size(1)
            image_embeds = image_embeds.reshape(b, n * p, image_embeds.size(2))

        return {"image_embeds": image_embeds}

    # ── Build lifecycle ───────────────────────────────────────────────────────

    @classmethod
    def _build_nn_module(cls, cfg: Dict[str, Any], init_device: str = "cpu") -> "JanusVisionEncoder":
        """Construct JanusVisionEncoder from a raw config dict.

        Allocates the module on *init_device* (use ``"meta"`` for lazy
        allocation when weights will be loaded by ``build_parallelize_model``).
        """
        config = JanusVisionEncoderConfig(
            vision_config=cfg.get("vision_config"),
        )
        with torch.device(init_device):
            return cls(config)

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    @classmethod
    def from_janus(cls, janus_model) -> "JanusVisionEncoder":
        """Extract vision encoder weights from a loaded JanusForConditionalGeneration."""

        vision_cfg = janus_model.config.vision_config
        cfg = JanusVisionEncoderConfig(
            vision_config=vision_cfg.to_dict() if hasattr(vision_cfg, "to_dict") else vars(vision_cfg)
        )
        enc = cls(cfg)
        enc.vision_model.load_state_dict(janus_model.model.vision_model.state_dict())
        enc.aligner.load_state_dict(janus_model.model.aligner.state_dict())
        return enc
