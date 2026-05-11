"""
JanusVQDecoder — OmniModule wrapping the Janus VQVAE + generation head.

Corresponds to JanusModel.vqmodel, JanusModel.generation_embeddings,
JanusModel.generation_aligner and JanusModel.generation_head in the original
JanusForConditionalGeneration checkpoint.

VQ generation loop (inside the FSM ``image_vq`` state)
------------------------------------------------------
The AR-LLM generates one VQ token ID per step.  For each step:

  1. ``ar_llm`` generates the next VQ token ID: ``vq_token_id`` (int).
  2. Connection ``ar_to_vq``: ``{from: ar_llm, output: vq_token_id, to: vq_decoder, as: token_id}``
  3. ``vq_decoder.generate_step(token_id=vq_token_id)`` →
       • looks up the codebook embedding for ``token_id``
       • projects it to LLM hidden size via ``generation_aligner``
       • returns ``embed`` (shape: ``(1, 1, llm_hidden)``)
  4. Connection ``vq_dec_to_ar``: ``{from: vq_decoder, output: embed, to: ar_llm, as: inputs_embeds}``
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
Training forward:
  ``gen_embeds``      Projected generation embeddings (teacher forcing inputs).
  ``vq_token_ids``    Ground-truth VQ token IDs (shift targets for LLM loss).
  ``vq_loss``         VQVAE embedding / commitment loss.

generate_step (single VQ lookup):
  ``embed``           Shape ``(B, 1, llm_hidden)`` — next-step input for the LLM.
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

    def forward(
        self,
        gen_image_patches: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Training forward.

        If ``gen_image_patches`` (float, shape ``(B, 3, H, W)``) is present,
        encode them with the VQVAE to get token IDs + generation embeddings.
        Otherwise returns an empty dict (text-only batch).
        """
        if gen_image_patches is None:
            return {}

        with torch.no_grad() if self.config.freeze_vqvae else torch.enable_grad():
            vq_out = self.vqmodel.encode(gen_image_patches)
        vq_token_ids = vq_out.image_tokens  # (B, N_patches)
        vq_loss = vq_out.embedding_loss if hasattr(vq_out, "embedding_loss") else None

        # Generation embeddings: lookup + project to LLM hidden size
        gen_embeds_raw = self.generation_embeddings(vq_token_ids)  # (B, N, embed_dim)
        gen_embeds = self.generation_aligner(gen_embeds_raw)       # (B, N, llm_hidden)

        out = {
            "gen_embeds": gen_embeds,
            "vq_token_ids": vq_token_ids,
        }
        if vq_loss is not None:
            out["vq_loss"] = vq_loss
        return out

    def generate_step(
        self,
        token_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Single VQ token → LLM embedding lookup.

        Called by the FSM for each step of the ``image_vq`` state body.

        Parameters
        ----------
        token_id:
            Long tensor of shape ``(B,)`` or ``(B, 1)`` — the VQ token ID
            produced by ``ar_llm.generate_step`` in the same FSM step.

        Returns
        -------
        ``{"embed": (B, 1, llm_hidden)}`` for injection into AR-LLM.
        """
        if token_id is None:
            return {}

        if token_id.ndim == 1:
            token_id = token_id.unsqueeze(1)  # (B, 1)

        embed_raw = self.generation_embeddings(token_id)   # (B, 1, embed_dim)
        embed = self.generation_aligner(embed_raw)          # (B, 1, llm_hidden)
        return {"embed": embed}

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
