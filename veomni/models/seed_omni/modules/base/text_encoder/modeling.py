"""
TextEncoder — generic OmniModule wrapping the word-token embedding (``wte``)
and language-model head (``lm_head``) extracted from any causal LLM.

Why a separate module?
----------------------
For a unified multi-modal AR setup, the embedding lookup and the projection
back to vocab logits are conceptually the *same* abstraction as a discrete-
image VQ codec — pre-stage and post-stage of a backbone:

    nn.Embedding (encode)  ─►  LLM backbone  ─►  lm_head (decode)
       |     ▲                                         |     ▲
       ▼     │                                         ▼     │
     input_ids               hidden_states         logits / next-token

By exposing this pair as an OmniModule with ``encode`` / ``decode``
call-site methods, the same downstream graph code that handles vision-VAE
works for text — and the backbone module no longer owns vocab-dependent
layers.

Mixin form
----------
``TextEncoder`` multi-inherits ``OmniModule`` (the SeedOmni V2 mixin) and
:class:`transformers.PreTrainedModel` so HuggingFace ``from_pretrained`` /
``save_pretrained`` work natively — no custom build hooks required.

Tied vs. untied weights
-----------------------
``config.tie_word_embeddings`` (default ``True``):

* ``True``  — there is no separate ``lm_head`` parameter.  Both encode and
              decode go through ``self.embed_tokens``: encode does a lookup,
              decode does ``F.linear(h, embed_tokens.weight)``.  Most modern
              LLMs (Qwen-1.5 small, GPT-2, etc.) tie weights this way.
* ``False`` — an additional ``self.lm_head: nn.Linear`` is allocated with
              its own parameters.  Used by LLaMA / Janus / Mistral.

When set to ``False`` and ``config.lm_head_bias`` is ``True`` the linear
head gains a bias term.  Default mirrors HuggingFace causal-LM
conventions (no bias).

Connection outputs
------------------
``encode``:
  ``inputs_embeds``    Float tensor ``(B, T, hidden_size)``.

``decode``:
  :class:`~veomni.utils.model_outputs.CausalLMOutputWithLogProbs` with
  ``logits`` and optional ``loss`` (training).  Inference sampling lives in
  model-specific subclasses (e.g. Janus ``decode`` with ``conversation_list``).
  :meth:`post_forward` maps ``loss`` →
  ``_loss`` for the OmniModel graph.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration import TextEncoderConfig
from .modulemixin import TextEncoderModuleMixin


class TextEncoder(TextEncoderModuleMixin, PreTrainedModel):
    """Word-token embedding + LM head as a single OmniModule."""

    config_class = TextEncoderConfig
    base_model_prefix = ""
    _no_split_modules: list = []
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(self, config: TextEncoderConfig):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        self._tokenizer: Optional[Any] = None
        self.post_init()

    # ── Gradient checkpointing (no-op — nothing to recompute) ──────────────────

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """No-op: an embedding/head has no activations worth checkpointing.

        Overridden so a uniform ``enable`` call from the trainer is accepted
        silently instead of raising in ``PreTrainedModel`` (see class note).
        """
        return

    def gradient_checkpointing_disable(self) -> None:
        """No-op counterpart to :meth:`gradient_checkpointing_enable`."""
        return

    def forward(self, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        return self.encode(**kwargs)

    def encode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        embeds = self.embed_tokens(input_ids)
        return {
            "inputs_embeds": embeds.squeeze(0) if embeds.size(0) == 1 else embeds,
        }

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[torch.LongTensor] = None,
        **_,
    ) -> dict:
        logits = self._project(hidden_states)
        loss: torch.Tensor | None = None

        if shift_labels is not None:
            ce_sum = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid = (shift_labels.view(-1) != -100).sum().clamp(min=1)
            loss = ce_sum / n_valid
        elif labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = labels[..., 1:].contiguous()
            ce_sum = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid = (shift_targets != -100).sum().clamp(min=1)
            loss = ce_sum / n_valid

        return {
            "loss": loss,
            "logits": logits,
        }
