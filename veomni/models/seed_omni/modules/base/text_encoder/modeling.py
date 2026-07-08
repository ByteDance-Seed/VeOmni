"""Generic word-token embedding (``wte``) + LM head as a graph node.

``TextEncoder(TextEncoderModuleMixin, PreTrainedModel)`` — ``encode`` /
``decode`` call-sites mirror a VQ codec pre/post stage so the backbone stays
vocab-agnostic.  Family-specific chat template / sampling live in
``modules/<family>/text_encoder/``.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import reduce_sequence_parallel_loss
from ....mixins.emb_parallel import EmbParallelMixin
from .configuration import TextEncoderConfig
from .modulemixin import TextEncoderMetricMeterMixin, TextEncoderModuleMixin


class TextEncoder(TextEncoderModuleMixin, TextEncoderMetricMeterMixin, EmbParallelMixin, PreTrainedModel):
    """Word-token embedding + LM head."""

    config_class = TextEncoderConfig
    base_model_prefix = ""
    _no_split_modules: list = ["Embedding", "Linear"]
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

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
        input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        embeds = self._embed_tokens(input_ids)
        return {
            "inputs_embeds": embeds.squeeze(0) if embeds.size(0) == 1 else embeds,
        }

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding lookup, vocab-parallel-aware when ``emb`` extra-parallel is on
        (:class:`EmbParallelMixin` — AllToAllEmbedding + emb_fsdp hidden gather)."""
        return self.emb_parallel_lookup(self.embed_tokens, input_ids)

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
            # Token-weighted mean over the FULL data-parallel + SP mesh
            # (``dp_sp`` == ``fsdp_group``): every valid token is weighted equally
            # and the objective is IDENTICAL no matter how the global batch is
            # split into DP vs SP — the invariant that makes per-module SP
            # accuracy-transparent. ``reduce_sequence_parallel_loss`` all-reduces
            # this rank's local CE sum / valid-token count over ``dp_sp`` and its
            # backward scales grads by ``|dp_sp|``, exactly cancelling FSDP2's
            # reduce-scatter (÷dp_shard_sp) + HSDP all-reduce (÷dp_replicate) so
            # the gradient is the true global token-mean. (A plain per-rank
            # ``ce_sum/n_valid`` would instead give a DP mean-of-means that
            # over-weights ranks holding few valid tokens.)
            ps = get_parallel_state()
            n_valid_local = (flat_labels != -100).sum()
            local_mean = ce_sum / n_valid_local.clamp(min=1)
            loss = reduce_sequence_parallel_loss(local_mean, n_valid_local.to(local_mean.dtype), group=ps.fsdp_group)
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

        # Tied head reuses ``embed_tokens.weight`` directly, bypassing
        # ``embed_tokens.__call__`` — and ``embed_tokens`` is its own FSDP2 unit
        # (``_no_split_modules``), so the pre-forward all-gather that would
        # materialize + cast the weight never fires here. :class:`EmbParallelMixin`
        # reconstructs this rank's slice (dual of ``_embed_tokens``) and projects
        # via ``VocabParallelLinear`` under ``emb`` (plain ``F.linear`` otherwise).
        return self.emb_parallel_project(hidden_states, self.embed_tokens.weight)
