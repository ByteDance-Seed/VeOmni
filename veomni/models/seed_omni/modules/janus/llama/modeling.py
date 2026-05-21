"""
JanusLlama — Janus' LLaMA backbone (no wte, no lm_head) as one OmniModule.

Mixin form: ``class JanusLlama(OmniModule, PreTrainedModel)``.

The backbone *contains* a vanilla :class:`transformers.LlamaModel` whose
``embed_tokens`` has been replaced with an :class:`nn.Identity` — the
word-token embedding lives in the sibling
:class:`~veomni.models.seed_omni.modules.base.TextEmbed` module.  This
keeps the LLaMA forward path unchanged; ``inputs_embeds`` (passed in by
the graph from the ``tok_encode`` node) is what actually flows through.

Multi-modal embedding scatter
-----------------------------
:meth:`pre_forward` is the only place that knows how to merge image
embeddings into ``inputs_embeds``: ``masked_scatter`` at
``input_ids == config.image_token_id`` (understanding) or
``config.gen_image_token_id`` (generation) positions.  This is the
"backbone owns the multimodal merge" pattern from design.md §10.

Sequence parallelism
--------------------
When the global :class:`ParallelState` has SP enabled, :meth:`pre_forward`
slices every sequence-domain tensor with
:func:`veomni.distributed.sequence_parallel.data.sp_pad_and_slice` and
:meth:`post_forward` all-gathers ``hidden_states`` back to full length so
downstream nodes (``tok_decode`` / ``vae_decode``) receive non-sliced
tensors.

Connection outputs
------------------
``hidden_states``  — final LLaMA hidden states ``(B, T, D)``.  CE / sampling
                     live in the head modules.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaModel

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel.data import gather_outputs, sp_pad_and_slice

from ....module import OmniModule
from .configuration import JanusLlamaConfig


class JanusLlama(OmniModule, PreTrainedModel):
    """LLaMA backbone with multi-modal embedding scatter (no wte, no lm_head).

    Image-embedding injection (``und_image_embeds`` / ``gen_image_embeds``)
    is performed via ``masked_scatter`` using a boolean mask derived from
    ``input_ids`` placeholder tokens — same strategy as the original Janus.
    ``inputs_embeds`` is required: there is no fallback ``embed_tokens``
    lookup inside this module.
    """

    config_class = JanusLlamaConfig
    base_model_prefix = "janus_llama"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, config: JanusLlamaConfig):
        super().__init__(config)

        text_cfg = LlamaConfig(**config.text_config) if config.text_config else LlamaConfig()
        self._text_cfg = text_cfg
        self.language_model = LlamaModel._from_config(text_cfg)
        # Drop the embed_tokens parameters — owned by sibling TextEmbed.
        self.language_model.set_input_embeddings(nn.Identity())

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if hasattr(module, "_init_weights"):
            return

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        und_image_embeds: Optional[torch.FloatTensor] = None,
        gen_image_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Multi-modal merge + SP slice (when enabled).

        Steps (in order):

          1. Scatter ``und_image_embeds`` / ``gen_image_embeds`` into
             ``inputs_embeds`` at placeholder token positions.  Done here
             rather than in :meth:`forward` so the SP slice (step 2) sees
             a fully-merged sequence — otherwise an image whose tokens
             span an SP boundary would lose its embedding.
          2. SP pad-and-slice every sequence-domain tensor.
          3. Forward pass through the LLaMA backbone and post-forward
             gather (the latter in :meth:`post_forward`).
        """
        if inputs_embeds is None:
            raise ValueError("JanusLlama.pre_forward expects `inputs_embeds` from `tok_encode`.")

        # Multi-modal merge.  The ``+ x.sum() * 0.0`` anchor below is the
        # FSDP grad-sync guard for training-side dummy forward (see module-
        # doc "Training vs. inference no input semantics" in
        # :mod:`veomni.models.seed_omni.module`): when the placeholder mask
        # is all-False (text-only micro-batch with dummy zero image
        # embeds), ``masked_scatter`` would otherwise drop the gradient
        # path back to the upstream encoder and FSDP grad-reduce would
        # mismatch across DP ranks.  The anchor adds a zero-valued
        # contribution that *is* part of the autograd graph, so the
        # upstream module always receives a (zero) gradient.
        if und_image_embeds is not None and input_ids is not None:
            inputs_embeds = self._scatter_image_embeds(
                inputs_embeds, input_ids, und_image_embeds, self.config.image_token_id
            )
            if self.training:
                inputs_embeds = inputs_embeds + und_image_embeds.sum() * 0.0
        if gen_image_embeds is not None and input_ids is not None:
            inputs_embeds = self._scatter_image_embeds(
                inputs_embeds, input_ids, gen_image_embeds, self.config.gen_image_token_id
            )
            if self.training:
                inputs_embeds = inputs_embeds + gen_image_embeds.sum() * 0.0

        ps = get_parallel_state()
        if ps.sp_enabled:
            if input_ids is not None:
                input_ids = sp_pad_and_slice(input_ids, dim=1, pad_value=0)
            inputs_embeds = sp_pad_and_slice(inputs_embeds, dim=1, pad_value=0)
            if labels is not None:
                labels = sp_pad_and_slice(labels, dim=1, pad_value=-100)
            if attention_mask is not None:
                attention_mask = sp_pad_and_slice(attention_mask, dim=1, pad_value=0)
            if position_ids is not None:
                position_ids = sp_pad_and_slice(position_ids, dim=1, pad_value=0)

        return dict(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    def post_forward(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """All-gather ``hidden_states`` across SP ranks (when SP enabled)."""
        ps = get_parallel_state()
        if ps.sp_enabled and "hidden_states" in outputs:
            outputs["hidden_states"] = gather_outputs(outputs["hidden_states"], gather_dim=1, scale_grad=True)
        return outputs

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Backbone forward.  ``inputs_embeds`` is required (no internal wte)."""
        if inputs_embeds is None:
            raise ValueError(
                "JanusLlama.forward expects `inputs_embeds` (produced by "
                "`text_embed.encode`). Word-token embedding has moved to "
                "the TextEmbed module."
            )

        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **{k: v for k, v in kwargs.items() if k in ("cache_position",)},
        )
        return {
            "hidden_states": lm_out.last_hidden_state,
            "past_key_values": lm_out.past_key_values,
        }

    def generate_step(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Single auto-regressive backbone step (FSM ``image_vq`` / ``text_ar``).

        Returns hidden states only; sampling lives downstream in
        ``text_embed.decode`` / :meth:`JanusVqvae.decode`.
        """
        if inputs_embeds is None:
            raise ValueError(
                "JanusLlama.generate_step expects `inputs_embeds` (from text_embed.encode or vqvae.decode)."
            )
        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return {
            "hidden_states": lm_out.last_hidden_state,
            "past_key_values": lm_out.past_key_values,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _scatter_image_embeds(
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        image_embeds: torch.Tensor,
        placeholder_token_id: int,
    ) -> torch.Tensor:
        """Replace placeholder positions in ``inputs_embeds`` with ``image_embeds``."""
        mask = (input_ids == placeholder_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        flat_embeds = image_embeds.reshape(-1, inputs_embeds.size(-1))
        return inputs_embeds.masked_scatter(mask, flat_embeds)
