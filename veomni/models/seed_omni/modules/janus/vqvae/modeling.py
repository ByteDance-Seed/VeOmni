"""Janus VQVAE codec + generation projection head.

``JanusVqvae(JanusVqvaeModuleMixin, PreTrainedModel)`` — codec weights here;
encode/decode graph hooks in ``modulemixin.py``.

Call-site split
---------------
* :meth:`pre_forward` — stash ``conversation_list``; ``method="encode"`` pulls
  assistant ``image`` pixels, ``method="decode"`` assembles llama hidden rows +
  ``gen_ids`` labels.
* :meth:`encode` — ``pixel_values`` → ``image_embeds`` + ``vq_token_ids``.
* :meth:`decode` — training CE: ``hidden_states`` + ``labels`` → ``loss``.
* :meth:`post_forward` — write ``image_embeds`` / ``janus_vqvae_labels`` back onto
  ``conversation_list`` (encode path).

Graph entry points (YAML ``module: janus_vqvae.<method>``):

  ``encode``   — training encode node (pixels → image embeds + VQ ids).
  ``decode``   — training VQ cross-entropy loss head.
  ``generate`` — inference VQ AR step (lm_head → sample → embed → merge).
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.models.janus.modeling_janus import (
    JanusVQVAE,
    JanusVQVAEAlignerMLP,
    JanusVQVAEHead,
)

from ......distributed.parallel_state import get_parallel_state
from .configuration import JanusVqvaeConfig
from .modulemixin import JanusVqvaeModuleMixin, JanusVqvaeTraceMixin
from .processing import JanusVqvaeProcessor


class JanusVqvae(JanusVqvaeModuleMixin, JanusVqvaeTraceMixin, PreTrainedModel):
    """VQVAE + generation head for Janus VQ image generation.

    The VQVAE encoder/decoder is frozen by default (matching the Janus
    paper).  Only the generation projection layers
    (``generation_embeddings``, ``generation_aligner``,
    ``generation_head``) are trainable.
    """

    config_class = JanusVqvaeConfig
    image_processor_class = JanusVqvaeProcessor
    base_model_prefix = "janus_vqvae"
    main_input_name = "pixel_values"
    _no_split_modules: list = []
    # The inner ``JanusVQVAE`` declares gradient-checkpointing support, so the
    # mixin advertises it too (keeps the wrapper's capability accurate and lets
    # the trainer's GC guard pass).  Note: the codec is frozen by default
    # (``config.freeze`` → :meth:`freeze_model`) and runs under ``no_grad`` in
    # training, so GC here is effectively inert — only the trainable
    # generation_* heads see grads.
    supports_gradient_checkpointing = True

    def __init__(self, config: JanusVqvaeConfig):
        super().__init__(config)
        self.config = config
        self.vqmodel = JanusVQVAE._from_config(config.vq_config)
        self.generation_embeddings = nn.Embedding(config.vq_config.num_embeddings, config.vq_config.embed_dim)
        self.generation_aligner = JanusVQVAEAlignerMLP(config.vq_config)
        self.generation_head = JanusVQVAEHead(config.vq_config)
        self._image_processor: JanusVqvaeProcessor = None
        self.post_init()

    def freeze_model(self) -> None:
        """Partial freeze: only the inner VQVAE codec (``vqmodel``).

        Matches the Janus recipe — the generation projection heads
        (``generation_embeddings`` / ``generation_aligner`` /
        ``generation_head``) stay trainable, so this module still gets an
        optimizer (over those heads).  Overrides the base whole-module
        default; gated on ``config.freeze`` (default ``True``).
        """
        if self.config.freeze:
            self.vqmodel.requires_grad_(False)

    def _encode_pixels(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad() if self.config.freeze else torch.enable_grad():
            vq_out = self.vqmodel.encode(pixel_values)
        vq_token_ids = vq_out.image_tokens
        image_embeds = self.generation_aligner(self.generation_embeddings(vq_token_ids))
        if vq_token_ids.dim() == 1:
            b = pixel_values.size(0)
            vq_token_ids = vq_token_ids.reshape(b, -1)
            image_embeds = image_embeds.reshape(b, vq_token_ids.size(1), image_embeds.size(-1))
        return {"image_embeds": image_embeds, "vq_token_ids": vq_token_ids}

    def _dummy_encode_outputs(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Zero stand-ins shaped exactly like :meth:`_encode_pixels`' output (same
        batch + per-image token count) for the non-FSDP dummy, whose codec forward
        is skipped (no gradient anchor needed without FSDP). Emitting real-shaped
        zeros instead of ``None`` lets the pre/post hooks treat every batch
        identically — no ``None`` / dummy special-casing downstream."""
        vq = self.config.vq_config
        downsample = 2 ** (len(vq.channel_multiplier) - 1)
        b, _, h, w = pixel_values.shape
        num_tokens = (h // downsample) * (w // downsample)
        return {
            "image_embeds": pixel_values.new_zeros(b, num_tokens, vq.projection_dim),
            "vq_token_ids": pixel_values.new_zeros(b, num_tokens, dtype=torch.long),
        }

    def encode(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        is_dummy: bool = False,
    ) -> Dict[str, Any]:
        # Real pixels → always encode. A worker-built dummy is only the training
        # FSDP gradient anchor, so it runs the codec solely under training + FSDP;
        # otherwise (inference, or no FSDP) there is nothing to anchor, so skip the
        # forward but still emit real-shaped zeros so the output is uniform with a
        # real encode (pre/post never branch on dummy).
        if is_dummy and not (self.training and get_parallel_state().fsdp_enabled):
            return self._dummy_encode_outputs(pixel_values)
        return self._encode_pixels(pixel_values)

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_dummy: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if hidden_states is None or labels is None:
            return {}
        if is_dummy:
            # Under training + FSDP, route the dummy span through the generation
            # head so its reduce-scatter fires on every rank (the gradient anchor).
            # Otherwise (inference, or no FSDP) no collective sync is needed, so
            # return a 0.0 loss directly.
            if self.training and get_parallel_state().fsdp_enabled:
                return {"loss": self.generation_head(hidden_states).sum() * 0.0}
            return {"loss": hidden_states.sum() * 0.0}
        return {"loss": self._vq_loss(hidden_states, labels)}

    def _vq_loss(self, hidden_states: torch.Tensor, gt_token_ids: torch.Tensor) -> torch.Tensor:
        # ``hidden_states`` is already teacher-forcing aligned in
        # ``_prepare_decode_inputs`` (row i = hidden after the previous token,
        # predicting VQ id i), so no further shift here — shifting again would
        # mis-align by one and bleed across concatenated image spans. Labels are
        # pure VQ codebook ids (every decoded row is an image token, no -100).
        labels = gt_token_ids.to(hidden_states.device)
        logits = self.generation_head(hidden_states)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
