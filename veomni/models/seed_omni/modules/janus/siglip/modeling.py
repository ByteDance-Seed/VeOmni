"""
JanusSiglip — Janus' SigLIP vision tower + MLP aligner as one OmniModule.

Mixin form: ``class JanusSiglip(OmniModule, PreTrainedModel)``.  HuggingFace
``from_pretrained`` / ``save_pretrained`` work natively against
``<weights_path>/{config.json, model.safetensors[, preprocessor_config.
json]}``; the SeedOmni V2 graph runtime (:class:`OmniModel`) calls
:meth:`forward` / :meth:`pre_forward` / :meth:`post_forward` per
``OmniModule`` mixin protocol.

Connection outputs
------------------
``image_embeds``
    Float tensor of shape ``(batch, num_patches, llm_hidden_size)`` ready
    to be injected into the AR-LLM as understanding image embeddings.

Batch inputs (read from raw batch)
----------------------------------
``pixel_values``
    Float tensor of shape ``(B, 3, H, W)`` — single image per sample, or
    ``(B, N_images, 3, H, W)`` — multiple images per sample.  The latter
    shape is flattened by :meth:`pre_forward`.
"""

from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel
from transformers.models.janus.configuration_janus import JanusVisionConfig
from transformers.models.janus.modeling_janus import JanusVisionAlignerMLP, JanusVisionModel

from ....module import OmniModule
from .configuration import JanusSiglipConfig


class JanusSiglip(OmniModule, PreTrainedModel):
    """SigLIP vision tower + MLP aligner for image understanding.

    Multi-inherits :class:`OmniModule` (V2 mixin) and
    :class:`PreTrainedModel` so HF lifecycle methods work natively.
    Loaded from the ``model.vision_model`` and ``model.aligner`` sub-
    modules of the original ``JanusForConditionalGeneration`` checkpoint
    (split into a standalone folder by ``scripts/split_janus.py``).
    """

    config_class = JanusSiglipConfig
    base_model_prefix = "janus_siglip"
    main_input_name = "pixel_values"
    _no_split_modules = ["JanusVisionEncoderLayer"]

    def __init__(self, config: JanusSiglipConfig):
        super().__init__(config)

        vision_cfg = JanusVisionConfig(**config.vision_config) if config.vision_config else JanusVisionConfig()
        self.vision_model = JanusVisionModel._from_config(vision_cfg)
        self.aligner = JanusVisionAlignerMLP(vision_cfg)

        self.post_init()

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Defer to the inner SigLIP modules' own initialisers."""
        if hasattr(module, "_init_weights"):
            return
        std = 0.02
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(self, pixel_values: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """Flatten ``(B, N_images, C, H, W)`` → ``(B*N_images, C, H, W)``.

        The original ``(B, N_images)`` shape is stored in
        ``_pv_batch_n_images`` so :meth:`forward` can reshape the output
        back.  ``(B, C, H, W)`` shapes pass through.
        """
        if pixel_values is not None and pixel_values.ndim == 5:
            b, n = pixel_values.shape[:2]
            pixel_values = pixel_values.reshape(b * n, *pixel_values.shape[2:])
            kwargs["_pv_batch_n_images"] = (b, n)
        return dict(pixel_values=pixel_values, **kwargs)

    def forward(self, pixel_values: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        """Encode understanding image patches to LLM-space embeddings.

        Returns ``{}`` for text-only batches (``pixel_values is None``) —
        this is the *inference* fast path.  In training the trainer fills
        :meth:`dummy_inputs` so this branch is never reached.
        """
        if pixel_values is None:
            return {}

        vision_out = self.vision_model(pixel_values, return_dict=True)
        feats = vision_out.last_hidden_state
        image_embeds = self.aligner(feats)

        b_n = kwargs.pop("_pv_batch_n_images", None)
        if b_n is not None:
            b, n = b_n
            p = image_embeds.size(1)
            image_embeds = image_embeds.reshape(b, n * p, image_embeds.size(2))

        return {"image_embeds": image_embeds}

    # ── Training-side dummy forward ────────────────────────────────────────────

    def dummy_inputs(self, *, batch_size: int, device: Any, dtype: Any) -> Dict[str, Any]:
        """Return zero ``pixel_values`` so the full vision path forwards.

        Used by the trainer for micro-batches that have no understanding
        images — keeps the FSDP graph aligned across DP/SP ranks.  The
        zero output flows through the LLM's ``masked_scatter`` as a
        no-op (mask is all-False) and contributes no gradient; the LLM's
        ``pre_forward`` adds an ``image_embeds.sum() * 0.0`` anchor so
        the upstream params still receive a (zero) gradient and FSDP
        sync stays consistent.
        """
        cfg = self.config.vision_config or {}
        h = cfg.get("image_size", 384) if isinstance(cfg, dict) else getattr(cfg, "image_size", 384)
        c = cfg.get("num_channels", 3) if isinstance(cfg, dict) else getattr(cfg, "num_channels", 3)
        return {
            "pixel_values": torch.zeros(batch_size, c, h, h, device=device, dtype=dtype),
        }
