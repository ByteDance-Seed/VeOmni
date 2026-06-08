"""
JanusVqvae — Janus' VQVAE + generation projection head as one OmniModule.

Mixin form:
``class JanusVqvae(JanusVqvaeTrainModuleMixin, JanusVqvaeInferModuleMixin, PreTrainedModel)``.

Call-site split (V2)
--------------------
* :meth:`pre_forward` — stash ``conversation_list``; ``method="encode"`` pulls
  assistant ``image`` pixels, ``method="decode"`` assembles llama hidden rows +
  ``gen_ids`` labels.
* :meth:`encode` — pure encoder: ``pixel_values`` → ``image_embeds`` +
  ``vq_token_ids``.
* :meth:`decode` — training CE head: ``hidden_states`` + ``labels`` → ``_loss``.
* :meth:`post_forward` — write ``image_embeds`` / ``janus_vqvae_labels`` back onto
  ``conversation_list`` (encode path).

Graph entry points (YAML ``module: janus_vqvae.<method>``):

  ``encode``   — training encode node (pixels → image embeds + VQ ids).
  ``decode``   — training VQ cross-entropy loss head.
  ``generate`` — inference VQ AR step (lm_head → sample → embed → merge).
"""

from typing import Any, List

import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.janus.modeling_janus import (
    JanusVQVAE,
    JanusVQVAEAlignerMLP,
    JanusVQVAEHead,
)

from .configuration import JanusVqvaeConfig
from .modulemixin import JanusVqvaeInferModuleMixin, JanusVqvaeTrainModuleMixin
from .processing import JanusVqvaeProcessor


class JanusVqvae(JanusVqvaeTrainModuleMixin, JanusVqvaeInferModuleMixin, PreTrainedModel):
    """VQVAE + generation head for Janus VQ image generation.

    The VQVAE encoder/decoder is frozen by default (matching the Janus
    paper).  Only the generation projection layers
    (``generation_embeddings``, ``generation_aligner``,
    ``generation_head``) are trainable.
    """

    config_class = JanusVqvaeConfig
    processor_class = JanusVqvaeProcessor
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
        self._processor: JanusVqvaeProcessor = None

        # NB: the ``config.freeze`` knob is honoured by :meth:`freeze_model`
        # (called once by the trainer after build), not here.

        # Per-image VQ-token buffer used by :meth:`generate` to accumulate
        # sampled tokens between FSM iterations.  Reset on each
        # ``image_complete`` signal; finalize keeps the decoded PIL
        # images so the caller can collect them after the run.
        # Inference cache
        self._vq_buffer: List[int] = []

        # Auto-populated by :meth:`OmniModule.from_pretrained` from
        # ``<weights_path>/preprocessor_config.json``.  Used by
        # :meth:`generate` to convert the VQVAE's ``[-1, 1]`` float output
        # back into a PIL image — see :class:`JanusVqvaeProcessor`.
        # Training cache
        self._conversation_carrier: Any = None

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
