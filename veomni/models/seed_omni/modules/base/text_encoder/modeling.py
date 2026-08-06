"""Generic word-token embedding (``wte``) + LM head as a graph node.

``TextEncoder(TextEncoderModuleMixin)`` — ``encode`` /
``decode`` call-sites mirror a VQ codec pre/post stage so the backbone stays
vocab-agnostic.  Family-specific chat template / sampling live in
``modules/<family>/text_encoder/``.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from ....omni_pretrained_model import OmniPreTrainedModel
from .configuration import TextEncoderConfig


class TextEncoder(OmniPreTrainedModel):
    """Word-token embedding + LM head."""

    config_class = TextEncoderConfig
    base_model_prefix = ""
    _no_split_modules: list = ["Embedding", "Linear"]
    main_input_name = "input_ids"

    def __init__(self, config: TextEncoderConfig):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.post_init()

    # ── Embedding accessors ────────────────────────────────────────────────────

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def get_output_embeddings(self) -> Optional[nn.Module]:
        # When tied there is no separate ``lm_head`` — ``_project`` reuses
        # ``embed_tokens.weight`` directly. Returning ``embed_tokens`` makes the
        # generic load-time weight-tie a harmless self-assignment instead of
        # crashing on a ``None`` output module.
        if self.config.tie_word_embeddings:
            return self.embed_tokens
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if not self.config.tie_word_embeddings:
            self.lm_head = new_embeddings

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        return self.encode(**kwargs)

    def encode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        embeds = self._embed_tokens(input_ids)
        return {
            "inputs_embeds": embeds.squeeze(0) if embeds.size(0) == 1 else embeds,
        }

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> dict:
        logits = self._project(hidden_states)
        loss: torch.Tensor | None = None

        if shift_labels is not None:
            flat_labels = shift_labels.view(-1)
            ce_sum = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                flat_labels,
                ignore_index=-100,
                reduction="sum",
            )
            n_valid_local = (flat_labels != -100).sum().clamp(min=1)
            loss = ce_sum / n_valid_local
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

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.config.tie_word_embeddings:
            return self.lm_head(hidden_states)
        return F.linear(hidden_states, self.embed_tokens.weight)
