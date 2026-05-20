"""
JanusLLM — pure backbone OmniModule (LLaMA causal model, no wte / no lm_head).

Corresponds to ``JanusModel.language_model`` in the original
``JanusForConditionalGeneration`` checkpoint, *minus* the word-token
embeddings and language-model head.  Those are now owned by the generic
:class:`~veomni.models.seed_omni.modules.text.TextEmbed` module so they
can be reused across LLM families and shared / tied independently.

Training
--------
Receives (all optional, mixed as needed):

  ``inputs_embeds``     (B, T, D) — text-side embeddings produced by
                        ``text_embed.encode`` (tok_encode node).  This module
                        treats them as the canonical input — there is **no**
                        internal ``embed_tokens`` lookup.
  ``input_ids``         (B, T)    — only used to locate placeholder positions
                        for image-embedding scatter (no embedding lookup).
  ``attention_mask``    (B, T)
  ``position_ids``      (B, T)
  ``und_image_embeds``  (B, N_und, D) — understanding image embeddings
                        scattered into ``inputs_embeds`` at
                        ``config.image_token_id`` positions.
  ``gen_image_embeds``  (B, N_gen, D) — generation image embeddings
                        scattered at ``config.gen_image_token_id`` positions.

Connection outputs
------------------
``hidden_states`` — final hidden states ``(B, T, D)``.  Cross-entropy and
                    sampling now live in ``text_embed.decode`` /
                    ``vq_decoder.decode``; this module never produces a loss
                    on its own.

generate_step outputs
---------------------
Same as ``forward`` — ``hidden_states`` only.  The FSM body invokes
``text_embed.decode`` next to sample from the projected logits; for VQ
states ``vq_decoder.decode`` consumes ``hidden_states`` and samples /
embeds from its own codebook.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel.data import gather_outputs, sp_pad_and_slice

from ...module import OmniModule


class JanusLLMConfig(PretrainedConfig):
    model_type = "janus_llm"

    def __init__(
        self,
        text_config: Optional[Dict] = None,
        image_token_id: int = 100577,
        gen_image_token_id: int = 100578,
        **kwargs,
    ):
        self.text_config = text_config or {}
        self.image_token_id = image_token_id
        self.gen_image_token_id = gen_image_token_id
        super().__init__(**kwargs)


class JanusLLM(OmniModule):
    """LLaMA backbone with multi-modal embedding scatter (no wte, no lm_head).

    Image-embedding injection (``und_image_embeds`` / ``gen_image_embeds``)
    is performed via ``masked_scatter`` using a boolean mask derived from
    ``input_ids`` placeholder tokens — the same strategy used in the
    original Janus code.  ``inputs_embeds`` is required: there is no
    fallback ``embed_tokens`` lookup inside this module.
    """

    config_class = JanusLLMConfig
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, config: JanusLLMConfig):
        super().__init__()
        self.config = config

        from transformers import AutoConfig

        text_cfg = AutoConfig.for_model(**config.text_config) if config.text_config else None
        if text_cfg is None:
            from transformers import LlamaConfig

            text_cfg = LlamaConfig()

        self.language_model = AutoModel.from_config(config=text_cfg)
        # Word embeddings are owned by the sibling `text_embed` module.
        # Replacing with Identity drops the parameters here so weight loading
        # only touches the backbone (the LLaMA forward path always uses the
        # `inputs_embeds=` branch downstream of us).
        self.language_model.set_input_embeddings(nn.Identity())

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Packing + SP slice for the LLM micro-batch.

        Steps (when SP is enabled):
          1. ``sp_pad_and_slice`` pads the sequence length to be divisible by
             ``sp_size``, then returns this rank's contiguous slice.
          2. The same slice is applied to all sequence-domain tensors —
             ``input_ids``, ``inputs_embeds``, ``labels``, ``attention_mask``,
             ``position_ids`` — so SP ranks are aligned.
          3. ``und_image_embeds`` and other patch-domain tensors are passed
             through unchanged; ``_scatter_image_embeds`` in ``forward``
             will inject only those whose placeholder tokens land in this
             rank's sequence slice.

        Note on multi-image SP routing
        --------------------------------
        When a sequence contains multiple images and SP splits the sequence,
        placeholder tokens for different images may land on different ranks.
        ``masked_scatter`` handles this correctly as long as each rank's
        ``und_image_embeds`` slice contains exactly the patches for the
        placeholders present in that rank's ``input_ids`` chunk.  In practice
        this is guaranteed when sequence packing assigns at most one image per
        packed sub-sequence (the standard convention for Janus training).
        """
        ps = get_parallel_state()
        if ps.sp_enabled:
            if input_ids is not None:
                input_ids = sp_pad_and_slice(input_ids, dim=1, pad_value=0)
            if inputs_embeds is not None:
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
        """SP gather: all-gather ``hidden_states`` across SP ranks.

        After gather, downstream nodes (``text_embed.decode``,
        ``vq_decoder.decode``, etc.) receive the full-length hidden state
        sequence rather than the SP-local slice.  The gather applies a
        ``1/sp_size`` gradient scaling on the backward pass to compensate
        for the all-reduce in SP loss averaging.
        """
        ps = get_parallel_state()
        if ps.sp_enabled and "hidden_states" in outputs:
            outputs["hidden_states"] = gather_outputs(
                outputs["hidden_states"],
                gather_dim=1,
                scale_grad=True,
            )
        return outputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        und_image_embeds: Optional[torch.FloatTensor] = None,
        gen_image_embeds: Optional[torch.FloatTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Backbone forward.

        Steps:
          1. ``inputs_embeds`` is required (text wte is owned by the
             ``text_embed`` module — no internal fallback lookup here).
          2. Scatter ``und_image_embeds`` at understanding-image positions.
          3. Scatter ``gen_image_embeds`` at generation-image positions.
          4. Run the LLaMA backbone.
          5. Return ``hidden_states`` (no logits, no loss — those live in
             ``text_embed.decode``).
        """
        if inputs_embeds is None:
            raise ValueError(
                "JanusLLM.forward expects `inputs_embeds` (produced by `text_embed.encode`). "
                "The internal word-token embedding has been moved to the TextEmbed module."
            )

        if und_image_embeds is not None and input_ids is not None:
            inputs_embeds = self._scatter_image_embeds(
                inputs_embeds, input_ids, und_image_embeds, self.config.image_token_id
            )

        if gen_image_embeds is not None and input_ids is not None:
            inputs_embeds = self._scatter_image_embeds(
                inputs_embeds, input_ids, gen_image_embeds, self.config.gen_image_token_id
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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Single auto-regressive backbone step.

        Accepts pre-computed ``inputs_embeds`` (from ``text_embed.encode``
        for text-AR steps, or ``vq_decoder.decode`` for image_vq feedback
        steps).  ``input_ids`` is optional and only used by the placeholder
        scatter — typically ``None`` in AR loops once the prefill is done.
        Returns hidden states; sampling lives downstream in
        ``text_embed.decode`` / ``vq_decoder.decode``.
        """
        if inputs_embeds is None:
            raise ValueError(
                "JanusLLM.generate_step expects `inputs_embeds` (from text_embed.encode or "
                "vq_decoder.decode). Sampling and embedding lookup are no longer in this module."
            )
        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return {
            "hidden_states": lm_out.last_hidden_state,  # (B, 1, D) at decode-time
            "past_key_values": lm_out.past_key_values,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _scatter_image_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        image_embeds: torch.Tensor,
        placeholder_token_id: int,
    ) -> torch.Tensor:
        """Replace placeholder positions in ``inputs_embeds`` with ``image_embeds``."""
        mask = (input_ids == placeholder_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        flat_embeds = image_embeds.reshape(-1, inputs_embeds.size(-1))
        return inputs_embeds.masked_scatter(mask, flat_embeds)

    # ── Build lifecycle ───────────────────────────────────────────────────────

    @classmethod
    def _build_nn_module(cls, cfg: Dict[str, Any], init_device: str = "cpu") -> "JanusLLM":
        """Construct JanusLLM from a raw config dict.

        Stores ``cfg["weights_path"]`` on the instance so :meth:`get_assets`
        can load the tokenizer without re-reading the config.
        """
        config = JanusLLMConfig(
            text_config=cfg.get("text_config"),
            image_token_id=cfg.get("image_token_id", 100577),
            gen_image_token_id=cfg.get("gen_image_token_id", 100578),
        )
        with torch.device(init_device):
            module = cls(config)
        module._weights_path = cfg.get("weights_path")
        return module

    def get_assets(self):
        """Return the LLaMA tokenizer bundled with this module's weights.

        Called by :meth:`OmniModel.collect_assets` after all modules are built.
        The tokenizer is used by the data pipeline and for saving model assets.
        Returns an empty list if no ``weights_path`` was set.
        """
        weights_path = getattr(self, "_weights_path", None)
        if not weights_path:
            return []
        from transformers import AutoTokenizer

        return [AutoTokenizer.from_pretrained(weights_path)]

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    @classmethod
    def from_janus(cls, janus_model) -> "JanusLLM":
        """Extract LLaMA backbone weights from a loaded JanusForConditionalGeneration.

        ``embed_tokens`` and ``lm_head`` are intentionally **not** loaded —
        they are owned by the sibling ``text_embed`` module (see
        :class:`~veomni.models.seed_omni.modules.text.TextEmbed`).
        """
        text_cfg = janus_model.config.text_config
        cfg = JanusLLMConfig(
            text_config=text_cfg.to_dict() if hasattr(text_cfg, "to_dict") else vars(text_cfg),
        )
        llm = cls(cfg)
        # Filter the wte out of the source state-dict so the load is strict-clean.
        src = janus_model.model.language_model.state_dict()
        src = {k: v for k, v in src.items() if not k.startswith("embed_tokens.")}
        llm.language_model.load_state_dict(src, strict=False)
        return llm
