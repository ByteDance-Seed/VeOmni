"""
JanusVQDecoder — OmniModule wrapping the Janus VQVAE + generation head.

Corresponds to JanusModel.vqmodel, JanusModel.generation_embeddings,
JanusModel.generation_aligner and JanusModel.generation_head in the original
JanusForConditionalGeneration checkpoint.

Call-site methods (graph entry points)
--------------------------------------
This module exposes three call-site methods that map 1-to-1 to YAML
``nodes:`` entries of the form ``module: vq_decoder.<method>``:

  ``encode``       — VQVAE encode (training + inference setup):
                     ``gen_image_patches`` → ``gen_embeds`` + ``vq_token_ids`` (+ ``vq_loss``).
  ``decode``       — Unified VQ head; one node covers both training loss and
                     inference sampling+feedback.  Three paths dispatched by
                     inputs:
                       * Training (``hidden_states`` + ``gt_token_ids``):
                         ``gen_loss`` (via ``generation_head`` + CE).
                       * Inference sample (``hidden_states`` only, no
                         ``gt_token_ids``): project last position via
                         ``generation_head``, sample → ``vq_token_id``,
                         then codebook lookup → ``embed`` (next-step
                         ``inputs_embeds``).  This is the path the FSM
                         ``image_vq`` state takes — sampling lives here
                         (not in ar_llm) because the VQ vocab is local to
                         this module.
                       * Inference lookup (``token_id`` only): pure codebook
                         lookup → ``embed``.  Useful when the caller
                         pre-sampled the VQ id from another source.
                     Paths are mutually exclusive at runtime — present-input
                     → run, absent-input → skip.
  ``decode_pixels``— image rendering (post-FSM): ``vq_token_ids`` → pixels.

``forward`` aliases ``encode`` (the most common training default).

VQ generation loop (inside the FSM ``image_vq`` state)
------------------------------------------------------
The AR backbone produces hidden states; sampling and embedding lookup both
happen inside this module (the AR LLM has no lm_head).  For each step:

  1. Node ``run_ar`` (``ar_llm.generate_step``) → ``hidden_states``.
  2. Edge ``ar_to_vq_decode``: routes ``hidden_states`` → ``hidden_states``.
  3. Node ``vq_decode`` (``vq_decoder.decode(hidden_states=...)``)
       • projects the last position via ``generation_head`` and samples
         → ``vq_token_id``
       • looks up the codebook embedding for the sampled id and projects
         to LLM hidden size via ``generation_aligner``
       • returns ``embed`` (shape: ``(B, 1, llm_hidden)``) plus
         ``vq_token_id``.
  4. Edge ``vq_decode_to_ar``: routes ``embed`` → ``inputs_embeds``.
  5. ``ar_llm`` uses ``inputs_embeds`` as its next-step input.

After 576 VQ steps the FSM transitions back to ``text_ar``.

Training
--------
During SFT the decoder is used to provide the *generation embeddings* for
image-generation sequences:

  • ``gen_image_patches`` (float, pre-encoded by VQVAE offline or online) → VQ
    encode → get token IDs and embeddings → feed as teacher-forcing inputs to
    the LLM.
  • The generation head produces per-position VQ-token logits for the cross-
    entropy loss.

Frozen by default (``vqmodel`` is frozen as in the original Janus paper;
only ``generation_embeddings``, ``generation_aligner``, ``generation_head``
are trained).

Connection outputs
------------------
``encode``:
  ``gen_embeds``      Projected generation embeddings (teacher-forcing inputs).
  ``vq_token_ids``    Ground-truth VQ token IDs (shift targets for LLM loss).
  ``vq_loss``         VQVAE embedding / commitment loss (when ``vqmodel`` unfrozen).

``decode``:
  ``gen_loss``        Image-generation CE (training; when ``hidden_states`` +
                      ``gt_token_ids`` are routed in).
  ``vq_token_id``     Sampled VQ id, shape ``(B,)`` (inference sample path,
                      when ``hidden_states`` alone is routed in).
  ``embed``           Shape ``(B, 1, llm_hidden)`` — next-step input for the
                      LLM (inference: emitted by both the sample and the
                      pre-sampled ``token_id`` paths).
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.janus.modeling_janus import (
    JanusVQVAE,
    JanusVQVAEAlignerMLP,
    JanusVQVAEHead,
)

from ...module import OmniModule


class JanusVQDecoderConfig(PretrainedConfig):
    model_type = "janus_vq_decoder"

    def __init__(
        self,
        vq_config: Optional[Dict] = None,
        freeze_vqvae: bool = True,
        **kwargs,
    ):
        self.vq_config = vq_config or {}
        self.freeze_vqvae = freeze_vqvae
        super().__init__(**kwargs)


class JanusVQDecoder(OmniModule):
    """VQVAE + generation head for Janus VQ image generation.

    The VQVAE encoder/decoder is frozen by default (matching the Janus paper).
    Only the generation projection layers (``generation_embeddings``,
    ``generation_aligner``, ``generation_head``) are trainable.
    """

    config_class = JanusVQDecoderConfig
    _no_split_modules: list = []

    def __init__(self, config: JanusVQDecoderConfig):
        super().__init__()
        self.config = config

        from transformers.models.janus.configuration_janus import JanusVQVAEConfig

        vq_cfg = JanusVQVAEConfig(**config.vq_config) if config.vq_config else JanusVQVAEConfig()
        self.vqmodel = JanusVQVAE._from_config(vq_cfg)
        self.generation_embeddings = nn.Embedding(vq_cfg.num_embeddings, vq_cfg.embed_dim)
        self.generation_aligner = JanusVQVAEAlignerMLP(vq_cfg)
        self.generation_head = JanusVQVAEHead(vq_cfg)

        if config.freeze_vqvae:
            self.vqmodel.requires_grad_(False)

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(
        self,
        gen_image_patches: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Packing: flatten per-sample multi-image patch tensors.

        Accepts ``gen_image_patches`` in two shapes:

        * ``(B, 3, H, W)`` — one image per sample, pass through.
        * ``(B, N_images, 3, H, W)`` — N images; flatten to
          ``(B * N_images, 3, H, W)``.
        """
        if gen_image_patches is not None and gen_image_patches.ndim == 5:
            b, n = gen_image_patches.shape[:2]
            gen_image_patches = gen_image_patches.reshape(b * n, *gen_image_patches.shape[2:])
            kwargs["_gpatch_batch_n_images"] = (b, n)
        return dict(gen_image_patches=gen_image_patches, **kwargs)

    # ── Call-site methods ─────────────────────────────────────────────────────
    # Each YAML node of the form ``module: vq_decoder.<method>`` invokes the
    # matching method below.  ``forward`` is the OmniModule abstract entry and
    # aliases to :meth:`encode` (the most common default).

    def forward(self, **kwargs) -> Dict[str, Any]:
        """Default forward — alias for :meth:`encode`."""
        return self.encode(**kwargs)

    def encode(
        self,
        gen_image_patches: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """VQVAE encode pass: pixels → ground-truth tokens + teacher-forcing embeds.

        Used by the ``vq_encode`` call-site during training.  If
        ``gen_image_patches`` (float, shape ``(B, 3, H, W)``) is present,
        encode them with the VQVAE to get token IDs + generation embeddings.
        Otherwise returns an empty dict (text-only batch).
        """
        if gen_image_patches is None:
            return {}

        # gen_image_patches: (B_flat, 3, H, W) after pre_forward flattening
        b_n = kwargs.pop("_gpatch_batch_n_images", None)

        with torch.no_grad() if self.config.freeze_vqvae else torch.enable_grad():
            vq_out = self.vqmodel.encode(gen_image_patches)
        vq_token_ids = vq_out.image_tokens  # (B_flat, N_patches)
        vq_loss = vq_out.embedding_loss if hasattr(vq_out, "embedding_loss") else None

        # Generation embeddings: lookup + project to LLM hidden size
        gen_embeds_raw = self.generation_embeddings(vq_token_ids)  # (B, N, embed_dim)
        gen_embeds = self.generation_aligner(gen_embeds_raw)  # (B, N, llm_hidden)

        # Reshape back to (B, N_images * N_patches, ...) if multi-image packing was used
        if b_n is not None:
            b, n = b_n
            p = vq_token_ids.size(1)
            vq_token_ids = vq_token_ids.reshape(b, n * p)
            gen_embeds = gen_embeds.reshape(b, n * p, gen_embeds.size(2))

        out = {
            "gen_embeds": gen_embeds,
            "vq_token_ids": vq_token_ids,
        }
        if vq_loss is not None:
            out["vq_loss"] = vq_loss
        return out

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        gt_token_ids: Optional[torch.Tensor] = None,
        token_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Unified VQ head — covers training loss + inference sample + lookup.

        Three input-driven dispatch paths:

          * **Training** (``hidden_states`` + ``gt_token_ids``):
              ``logits = generation_head(hidden_states)``, then CE w.r.t.
              ``gt_token_ids``.  Returns ``{"gen_loss": scalar}``.
              Caller must pre-slice both tensors to gen-image positions
              and apply the next-token shift; this method does not
              inspect the LLM input sequence to discover gen positions.

          * **Inference sample** (``hidden_states`` alone, no
            ``gt_token_ids``): project the last position via
            ``generation_head``, sample with multinomial, then codebook
            lookup + ``generation_aligner`` for the next embed.
            Returns ``{"vq_token_id", "embed"}`` — the FSM
            ``image_vq`` body uses this path, sampling lives here because
            the VQ vocab is local to this module.

          * **Inference lookup** (``token_id``, when the caller already
            sampled): codebook lookup via ``generation_embeddings``, then
            project to LLM hidden size via ``generation_aligner``.
            Returns ``{"embed"}`` — useful for non-AR sampling strategies
            (e.g. classifier-free guidance) that produce ``token_id``
            outside the FSM body.

        The branches are mutually exclusive at runtime: present-input → run,
        absent-input → skip — so a single YAML node serves both training and
        inference.
        """
        out: Dict[str, Any] = {}

        if hidden_states is not None and gt_token_ids is not None:
            logits = self.generation_head(hidden_states)
            out["gen_loss"] = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                gt_token_ids.reshape(-1).long(),
                ignore_index=-100,
            )
            return out

        if hidden_states is not None:
            # Inference sample path: hidden_states → next vq_token_id → embed
            last_logits = self.generation_head(hidden_states[:, -1:, :])  # (B, 1, V)
            probs = torch.softmax(last_logits.squeeze(1), dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)  # (B, 1)
            embed_raw = self.generation_embeddings(sampled)  # (B, 1, embed_dim)
            out["vq_token_id"] = sampled.squeeze(-1)  # (B,) for FSM transition checks
            out["embed"] = self.generation_aligner(embed_raw)  # (B, 1, llm_hidden)
            return out

        if token_id is not None:
            if token_id.ndim == 1:
                token_id = token_id.unsqueeze(1)  # (B, 1)
            embed_raw = self.generation_embeddings(token_id)  # (B, 1, embed_dim)
            out["embed"] = self.generation_aligner(embed_raw)  # (B, 1, llm_hidden)
            return out

        return out

    def decode_pixels(self, vq_token_ids: torch.Tensor) -> torch.Tensor:
        """Decode a sequence of VQ token IDs to pixel values.

        Parameters
        ----------
        vq_token_ids:
            Long tensor of shape ``(B, N_patches)``.

        Returns
        -------
        Float tensor ``(B, H, W, 3)`` in [0, 1].
        """
        pixel_values = self.vqmodel.decode(vq_token_ids)  # (B, 3, H, W)
        return pixel_values.permute(0, 2, 3, 1)

    # ── Build lifecycle ───────────────────────────────────────────────────────

    @classmethod
    def _build_nn_module(cls, cfg: Dict[str, Any], init_device: str = "cpu") -> "JanusVQDecoder":
        """Construct JanusVQDecoder from a raw config dict.

        ``cfg["freeze_vqvae"]`` (default ``True``) freezes the VQVAE backbone
        after construction so only the generation projection layers are trained,
        matching the original Janus training recipe.
        """
        config = JanusVQDecoderConfig(
            vq_config=cfg.get("vq_config"),
            freeze_vqvae=cfg.get("freeze_vqvae", True),
        )
        with torch.device(init_device):
            return cls(config)

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    @classmethod
    def from_janus(cls, janus_model) -> "JanusVQDecoder":
        """Extract VQ decoder weights from a loaded JanusForConditionalGeneration."""
        vq_cfg = janus_model.config.vq_config
        cfg = JanusVQDecoderConfig(
            vq_config=vq_cfg.to_dict() if hasattr(vq_cfg, "to_dict") else vars(vq_cfg),
            freeze_vqvae=True,
        )
        dec = cls(cfg)
        dec.vqmodel.load_state_dict(janus_model.model.vqmodel.state_dict())
        dec.generation_embeddings.load_state_dict(janus_model.model.generation_embeddings.state_dict())
        dec.generation_aligner.load_state_dict(janus_model.model.generation_aligner.state_dict())
        dec.generation_head.load_state_dict(janus_model.model.generation_head.state_dict())
        return dec
