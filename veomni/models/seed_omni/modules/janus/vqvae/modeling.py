"""
JanusVqvae — Janus' VQVAE + generation projection head as one OmniModule.

Mixin form: ``class JanusVqvae(OmniModule, PreTrainedModel)``.

Call-site methods (graph entry points)
--------------------------------------
This module exposes three call-site methods that map 1-to-1 to YAML
``nodes:`` entries of the form ``module: <name>.<method>``:

  ``encode``       — VQVAE encode (training + inference setup):
                     ``gen_image_patches`` → ``gen_embeds`` + ``vq_token_ids``.
  ``decode``       — Unified VQ head; one node covers training loss and
                     inference sampling+feedback:

                       * Training (``hidden_states`` + ``gt_token_ids``):
                         ``_loss`` (token-mean CE via ``generation_head``).
                       * Inference sample (``hidden_states`` only): project
                         the last position via ``generation_head``, sample
                         → ``vq_token_id``, codebook lookup → ``embed``
                         (next-step ``inputs_embeds``).
                       * Inference lookup (``token_id`` only): codebook
                         lookup → ``embed`` (when sampling lives outside
                         the FSM body).

  ``decode_pixels``— Image rendering (post-FSM): ``vq_token_ids`` → pixels.

``forward`` aliases :meth:`encode` (the most common training default).

Single-loss protocol
--------------------
``decode`` returns a token-mean ``_loss`` scalar in training (the only
allowed loss key per the V2 ``OmniModel`` contract).  Inference paths
return ``vq_token_id`` / ``embed`` and never set ``_loss``.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.janus.configuration_janus import JanusVQVAEConfig
from transformers.models.janus.modeling_janus import (
    JanusVQVAE,
    JanusVQVAEAlignerMLP,
    JanusVQVAEHead,
)

from ....module import OmniModule
from .configuration import JanusVqvaeConfig


class JanusVqvae(OmniModule, PreTrainedModel):
    """VQVAE + generation head for Janus VQ image generation.

    The VQVAE encoder/decoder is frozen by default (matching the Janus
    paper).  Only the generation projection layers
    (``generation_embeddings``, ``generation_aligner``,
    ``generation_head``) are trainable.
    """

    config_class = JanusVqvaeConfig
    base_model_prefix = "janus_vqvae"
    main_input_name = "gen_image_patches"
    _no_split_modules: list = []

    def __init__(self, config: JanusVqvaeConfig):
        super().__init__(config)

        vq_cfg = JanusVQVAEConfig(**config.vq_config) if config.vq_config else JanusVQVAEConfig()
        self._vq_cfg = vq_cfg
        self.vqmodel = JanusVQVAE._from_config(vq_cfg)
        self.generation_embeddings = nn.Embedding(vq_cfg.num_embeddings, vq_cfg.embed_dim)
        self.generation_aligner = JanusVQVAEAlignerMLP(vq_cfg)
        self.generation_head = JanusVQVAEHead(vq_cfg)

        if config.freeze_vqvae:
            self.vqmodel.requires_grad_(False)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if hasattr(module, "_init_weights"):
            return
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(self, gen_image_patches: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """Flatten ``(B, N_images, 3, H, W)`` → ``(B*N_images, 3, H, W)``."""
        if gen_image_patches is not None and gen_image_patches.ndim == 5:
            b, n = gen_image_patches.shape[:2]
            gen_image_patches = gen_image_patches.reshape(b * n, *gen_image_patches.shape[2:])
            kwargs["_gpatch_batch_n_images"] = (b, n)
        return dict(gen_image_patches=gen_image_patches, **kwargs)

    def forward(self, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        """Default forward — alias for :meth:`encode`."""
        return self.encode(**kwargs)

    def encode(self, gen_image_patches: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """VQVAE encode pass: pixels → ground-truth tokens + teacher-forcing embeds.

        Returns ``{}`` for text-only batches.  When ``self.config.freeze_vqvae``
        is ``True`` the VQVAE is wrapped in ``torch.no_grad()`` to avoid
        tracking gradients on frozen parameters.
        """
        if gen_image_patches is None:
            return {}

        b_n = kwargs.pop("_gpatch_batch_n_images", None)

        with torch.no_grad() if self.config.freeze_vqvae else torch.enable_grad():
            vq_out = self.vqmodel.encode(gen_image_patches)
        vq_token_ids = vq_out.image_tokens

        gen_embeds_raw = self.generation_embeddings(vq_token_ids)
        gen_embeds = self.generation_aligner(gen_embeds_raw)

        if b_n is not None:
            b, n = b_n
            p = vq_token_ids.size(1)
            vq_token_ids = vq_token_ids.reshape(b, n * p)
            gen_embeds = gen_embeds.reshape(b, n * p, gen_embeds.size(2))

        return {"gen_embeds": gen_embeds, "vq_token_ids": vq_token_ids}

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        gt_token_ids: Optional[torch.Tensor] = None,
        token_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Unified VQ head — training loss + inference sample + lookup.

        Three input-driven dispatch paths:

          * **Training** (``hidden_states`` + ``gt_token_ids``):
              ``logits = generation_head(hidden_states)``, then token-mean
              CE w.r.t. ``gt_token_ids``.  Returns ``{"_loss"}``.

          * **Inference sample** (``hidden_states`` alone, no
            ``gt_token_ids``): project the last position, sample with
            multinomial, codebook lookup + ``generation_aligner`` for the
            next embed.  Returns ``{"vq_token_id", "embed"}``.

          * **Inference lookup** (``token_id``): codebook lookup +
            ``generation_aligner``.  Returns ``{"embed"}``.

        Branches are mutually exclusive at runtime — one set of inputs
        present means one path runs.
        """
        out: Dict[str, Any] = {}

        if hidden_states is not None and gt_token_ids is not None:
            logits = self.generation_head(hidden_states)
            out["_loss"] = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                gt_token_ids.reshape(-1).long(),
                ignore_index=-100,
            )
            return out

        if hidden_states is not None:
            last_logits = self.generation_head(hidden_states[:, -1:, :])
            probs = torch.softmax(last_logits.squeeze(1), dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            embed_raw = self.generation_embeddings(sampled)
            out["vq_token_id"] = sampled.squeeze(-1)
            out["embed"] = self.generation_aligner(embed_raw)
            return out

        if token_id is not None:
            if token_id.ndim == 1:
                token_id = token_id.unsqueeze(1)
            embed_raw = self.generation_embeddings(token_id)
            out["embed"] = self.generation_aligner(embed_raw)
            return out

        return out

    def decode_pixels(self, vq_token_ids: torch.Tensor) -> torch.Tensor:
        """Decode a sequence of VQ token IDs to pixel values ``(B, H, W, 3)``."""
        pixel_values = self.vqmodel.decode(vq_token_ids)
        return pixel_values.permute(0, 2, 3, 1)

    # ── Training-side dummy forward ────────────────────────────────────────────

    def dummy_inputs(self, *, batch_size: int, device: Any, dtype: Any) -> Dict[str, Any]:
        """Return zero ``gen_image_patches`` so the VQVAE encode path runs.

        Used by the trainer for micro-batches that don't carry any image
        for VQ-generation — keeps the FSDP graph aligned across DP/SP
        ranks.  See module-doc "Training vs. inference no input
        semantics" in :mod:`veomni.models.seed_omni.module`.
        """
        cfg = self.config.vq_config or {}
        h = cfg.get("resolution", 384) if isinstance(cfg, dict) else getattr(cfg, "resolution", 384)
        c = cfg.get("in_channels", 3) if isinstance(cfg, dict) else getattr(cfg, "in_channels", 3)
        return {
            "gen_image_patches": torch.zeros(batch_size, c, h, h, device=device, dtype=dtype),
        }
