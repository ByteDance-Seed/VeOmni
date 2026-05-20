"""
TextEmbed — generic OmniModule wrapping the word-token embedding (`wte`)
and language-model head (`lm_head`) extracted from any causal LLM.

Why a separate module?
----------------------
For a unified multi-modal AR setup, the embedding lookup and the projection
back to vocab logits are conceptually the *same* abstraction as a
discrete-image VQ codec — pre-stage and post-stage of a backbone:

    nn.Embedding (encode)  ─►  LLM backbone  ─►  lm_head (decode)
       |     ▲                                         |     ▲
       ▼     │                                         ▼     │
     input_ids               hidden_states         logits / next-token

By exposing this pair as an OmniModule with ``encode`` / ``decode`` call-site
methods, the same downstream graph code that handles vision-VAE works for
text — and the LLM module becomes a pure hidden-state backbone, no longer
owning vocabulary-dependent layers.

Call-site methods (graph entry points)
--------------------------------------
This module exposes two call-site methods that map 1-to-1 to YAML
``nodes:`` entries of the form ``module: text_embed.<method>``:

  ``encode``   — ``input_ids`` (B, T) → ``inputs_embeds`` (B, T, D).
                 Plain ``nn.Embedding`` lookup.

  ``decode``   — Unified head, two paths dispatched by inputs:
                 * **Training**  (``hidden_states`` + ``labels``):
                     ``logits = embed_tokens.weight @ h``  (tied) or
                     ``logits = lm_head(h)``               (untied),
                     followed by next-token-shifted cross-entropy.
                     Returns ``{"logits": ..., "lm_loss": scalar}``.
                 * **Inference** (``hidden_states`` only, no labels):
                     same projection, then sample (temperature / top-p)
                     the last position.  Returns
                     ``{"logits", "last_token_id", "input_ids"}`` so the
                     next FSM step can re-encode the sampled token.

``forward`` aliases :meth:`encode` (the default training entry).

Tied vs. untied weights
-----------------------
``config.tie_word_embeddings`` (default ``True``) controls the LM-head:

* ``True``  — there is no separate ``lm_head`` parameter.  Both encode and
              decode go through ``self.embed_tokens``: encode does a lookup,
              decode does ``F.linear(h, embed_tokens.weight)``.  Most modern
              LLMs (Qwen-1.5 small, GPT-2, etc.) tie weights this way.
* ``False`` — an additional ``self.lm_head: nn.Linear`` is allocated with
              its own parameters.  Used by LLaMA / Janus / Mistral.

When set to ``False`` and ``config.lm_head_bias`` is ``True`` the linear
head also gains a bias term.  The default mirrors HuggingFace causal-LM
conventions (no bias).

Connection outputs
------------------
``encode``:
  ``inputs_embeds``    Float tensor ``(B, T, hidden_size)``.

``decode``:
  ``logits``           Float tensor ``(B, T, vocab_size)``.
  ``lm_loss``          Scalar CE loss (training, when ``labels`` is given).
  ``last_token_id``    Long tensor ``(B,)`` — sampled next token (inference).
  ``input_ids``        Long tensor ``(B, 1)`` — same id, ready to be fed
                       back to ``encode`` on the next FSM step.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from ...module import OmniModule


class TextEmbedConfig(PretrainedConfig):
    """Configuration for :class:`TextEmbed`.

    Parameters
    ----------
    vocab_size:
        Number of token IDs in the embedding / projection.
    hidden_size:
        LLM hidden-state dimension.  Must match the backbone model.
    tie_word_embeddings:
        If ``True``, ``decode`` projects via ``embed_tokens.weight`` (no
        separate ``lm_head``).  If ``False``, an independent
        ``nn.Linear`` is allocated.  Default: ``True``.
    lm_head_bias:
        Only meaningful when ``tie_word_embeddings`` is ``False``.  When
        ``True``, the untied ``lm_head`` gains a bias term.  Default: ``False``.
    """

    model_type = "text_embed"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        tie_word_embeddings: bool = True,
        lm_head_bias: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tie_word_embeddings = tie_word_embeddings
        self.lm_head_bias = lm_head_bias
        super().__init__(**kwargs)


class TextEmbed(OmniModule):
    """Word-token embedding + LM head as a single OmniModule.

    Two call-site methods (``encode`` / ``decode``) plus a tied/untied
    weight option together replace the ``embed_tokens`` and ``lm_head``
    layers that traditionally live on the LLM backbone.  See module
    docstring for the full schema.
    """

    config_class = TextEmbedConfig
    _no_split_modules: list = []

    def __init__(self, config: TextEmbedConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # Tied: ``decode`` reuses ``embed_tokens.weight`` directly.
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)

    # ── Call-site methods ─────────────────────────────────────────────────────

    def forward(self, **kwargs) -> Dict[str, Any]:
        """Default forward — alias for :meth:`encode` (the most common default)."""
        return self.encode(**kwargs)

    def encode(self, input_ids: Optional[torch.LongTensor] = None, **_) -> Dict[str, Any]:
        """Word-token embedding lookup.

        Parameters
        ----------
        input_ids:
            Long tensor of shape ``(B, T)`` (or ``(B,)``).

        Returns
        -------
        ``{"inputs_embeds": (B, T, hidden_size)}`` — or ``{}`` when ``input_ids``
        is missing (text-free batch / micro-batch).
        """
        if input_ids is None:
            return {}
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        return {"inputs_embeds": self.embed_tokens(input_ids)}

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **_,
    ) -> Dict[str, Any]:
        """Unified LM-head projection — training (loss) + inference (sample).

        Two paths through one method, dispatched by inputs:

          * **Training** (``hidden_states`` + ``labels``):
              project to vocab logits, compute next-token-shifted CE.
              Returns ``{"logits", "lm_loss"}``.

          * **Inference** (``hidden_states`` only):
              project the last position to logits, sample (temperature /
              top-p) and return the next token.  Returns
              ``{"logits", "last_token_id", "input_ids"}``.

        Both halves run independently — present-input → run, absent-input
        → skip — so a single YAML node serves both training and inference.
        """
        if hidden_states is None:
            return {}

        logits = self._project(hidden_states)
        out: Dict[str, Any] = {"logits": logits}

        if labels is not None:
            # Standard causal-LM next-token shift.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            out["lm_loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            # Inference: sample next token from the last position.
            next_token_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = self._top_p_filter(next_token_logits, top_p)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            out["last_token_id"] = next_token.squeeze(-1)  # (B,) for FSM transition checks
            out["input_ids"] = next_token  # (B, 1) — flows back into encode on the next step

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
        # Shift by one so the first token above ``top_p`` is still kept.
        sorted_indices_to_remove = cumulative - sorted_probs > top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        return logits.scatter(1, sorted_indices, sorted_logits)

    # ── Build lifecycle ───────────────────────────────────────────────────────

    @classmethod
    def _build_nn_module(cls, cfg: Dict[str, Any], init_device: str = "cpu") -> "TextEmbed":
        """Construct TextEmbed from a raw config dict.

        Recognised keys:

        * ``vocab_size``, ``hidden_size``    — required (or read from
          ``text_config`` for HF-style nesting).
        * ``tie_word_embeddings``            — default ``True``.
        * ``lm_head_bias``                   — default ``False``.

        Convenience: a nested ``text_config`` (HF causal-LM style) is also
        accepted and unpacks ``vocab_size`` / ``hidden_size`` /
        ``tie_word_embeddings`` from there.  Top-level keys override.
        """
        text_config = cfg.get("text_config") or {}
        vocab_size = cfg.get("vocab_size", text_config.get("vocab_size"))
        hidden_size = cfg.get("hidden_size", text_config.get("hidden_size"))
        if vocab_size is None or hidden_size is None:
            raise ValueError(
                "TextEmbed requires `vocab_size` and `hidden_size` (top-level "
                "or under nested `text_config`). Got cfg keys: "
                f"{sorted(cfg)}; text_config keys: {sorted(text_config) if text_config else []}."
            )
        config = TextEmbedConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            tie_word_embeddings=cfg.get("tie_word_embeddings", text_config.get("tie_word_embeddings", True)),
            lm_head_bias=cfg.get("lm_head_bias", False),
        )
        with torch.device(init_device):
            return cls(config)
