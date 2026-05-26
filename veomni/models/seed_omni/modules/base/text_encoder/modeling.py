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
  ``logits``           Float tensor ``(B, T, vocab_size)``.
  ``_loss``            Scalar token-mean CE loss (training, when
                       ``labels`` is given).
  ``input_ids``        Long tensor ``(B, 1)`` — sampled next token for the
                       next FSM step (HF ``generate``-aligned; same field
                       name as encode input).
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ....module import OmniModule
from .configuration import TextEncoderConfig


class TextEncoder(OmniModule, PreTrainedModel):
    """Word-token embedding + LM head as a single OmniModule.

    Multi-inherits :class:`OmniModule` (V2 mixin) and HuggingFace
    :class:`PreTrainedModel` so HF ``from_pretrained`` / ``save_pretrained``
    work natively against ``<weights_path>/{config.json,
    model.safetensors}``.
    """

    config_class = TextEncoderConfig
    base_model_prefix = "text_encoder"
    _no_split_modules: list = []
    main_input_name = "input_ids"

    def __init__(self, config: TextEncoderConfig):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """HF ``PreTrainedModel`` requires this; mirrors LLaMA's init."""
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    # ── Call-site methods ─────────────────────────────────────────────────────

    def forward(self, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        """Default forward — alias for :meth:`encode` (the most common default)."""
        return self.encode(**kwargs)

    def encode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        **_,
    ) -> Dict[str, Any]:
        """Word-token embedding lookup.

        During autoregressive inference with KV cache, only the latest token
        needs embedding — earlier positions are already in ``past_key_values``.
        When ``past_key_values`` is set and ``input_ids`` is a growing
        ``(B, T)`` sequence, embed ``input_ids[:, -1:]`` only.  On the first
        prompt pass (no cache), embed the full ``input_ids`` prompt.

        Returns ``{"inputs_embeds": (B, T, hidden_size)}`` — or ``{}`` when
        ``input_ids`` is missing (text-free batch / micro-batch).
        """
        if input_ids is None:
            return {}
        ids = input_ids
        if past_key_values is not None and isinstance(ids, torch.Tensor) and ids.ndim == 2 and ids.size(-1) > 1:
            ids = ids[:, -1:]
        if isinstance(ids, torch.Tensor) and ids.ndim == 1:
            ids = ids.unsqueeze(1)
        return {"inputs_embeds": self.embed_tokens(ids)}

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **_,
    ) -> Dict[str, Any]:
        """Unified LM-head projection — training (loss) + inference (sample).

        * **Training** (``hidden_states`` + ``labels``):
            project to vocab logits, compute next-token-shifted CE.
            Returns ``{"logits", "_loss"}`` (token-mean over non-ignored
            positions).
        * **Inference** (``hidden_states`` only):
            project the last position to logits, sample (temperature /
            top-p) and return the next token.  Returns
            ``{"logits", "input_ids"}`` where ``input_ids`` is ``(B, 1)``.
        """
        if hidden_states is None:
            return {}

        logits = self._project(hidden_states)
        out: Dict[str, Any] = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            out["_loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            next_token_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = self._top_p_filter(next_token_logits, top_p)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out["input_ids"] = next_token

        return out

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocab logits (tied or untied)."""
        if self.config.tie_word_embeddings:
            return F.linear(hidden_states, self.embed_tokens.weight)
        return self.lm_head(hidden_states)

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus filtering: zero out the long tail beyond cumulative-p."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative - sorted_probs > top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        return logits.scatter(1, sorted_indices, sorted_logits)

    # ── Training-side dummy forward ────────────────────────────────────────────

    def dummy_inputs(self, *, batch_size: int, device: Any, dtype: Any) -> Dict[str, Any]:
        """Return a length-1 zero ``input_ids`` placeholder.

        Used by the trainer when a micro-batch is missing ``input_ids``
        (e.g. a pure image-tokens micro-batch that pre-cached
        ``inputs_embeds`` from disk).  See module-doc "Training vs.
        inference no input semantics" in
        :mod:`veomni.models.seed_omni.module`.
        """
        del dtype  # input_ids is always int64
        return {
            "input_ids": torch.zeros(batch_size, 1, dtype=torch.long, device=device),
        }
