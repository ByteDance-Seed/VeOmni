"""
JanusLLM — OmniModule wrapping the Janus LLaMA language model + LM head.

Corresponds to JanusModel.language_model + JanusForConditionalGeneration.lm_head
in the original checkpoint.

This module is the central driver of the OmniModel:

Training
--------
Receives (all optional, mixed as needed):

  ``input_ids``         (B, T) — token IDs for the full sequence.
  ``attention_mask``    (B, T) — causal mask.
  ``labels``            (B, T) — -100 outside supervised positions.
  ``und_image_embeds``  (B, N_und, D) — understanding image embeddings
                        injected at image-placeholder positions.
  ``gen_image_embeds``  (B, N_gen, D) — generation image embeddings
                        (teacher-forcing inputs for VQ generation sequences).
  ``inputs_embeds``     (B, T, D) — pre-built embedding sequence (overrides
                        input_ids lookup + image injection).

If ``und_image_embeds`` is provided, the module scatters them into the token
embedding sequence at image-placeholder positions (token id =
``config.image_token_id``, defaulting to Janus's placeholder token).

If ``gen_image_embeds`` is provided they are prepended / injected at
generation-placeholder positions similarly.

Connection outputs (training)
------------------------------
``lm_loss``       Scalar cross-entropy loss over text positions.
``hidden_states`` Last hidden states ``(B, T, D)`` for downstream connections
                  (e.g. conditioning DiT).

generate_step outputs
---------------------
``last_token_id``    Int64 scalar — the sampled next token (for FSM transitions).
``inputs_embeds``    (B, 1, D) — the embedding of ``last_token_id`` to pass back
                     when the FSM state re-injects it (``vq_dec_to_ar``).
``past_key_values``  KV cache for incremental decoding.
``logits``           (B, 1, V) — raw logits (for sampling / beam search).
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
    """LLaMA language model with multi-modal embedding injection.

    The model accepts both understanding and generation image embeddings and
    injects them at the appropriate placeholder positions.  Injection is done
    via ``masked_scatter`` using a boolean mask derived from the ``input_ids``
    placeholder tokens — the same strategy used in the original Janus code.
    """

    config_class = JanusLLMConfig
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, config: JanusLLMConfig):
        super().__init__()
        self.config = config

        from transformers import AutoConfig

        text_cfg = AutoConfig.for_model(**config.text_config) if config.text_config else None
        if text_cfg is not None:
            self.language_model = AutoModel.from_config(config=text_cfg)
            hidden_size = text_cfg.hidden_size
            vocab_size = text_cfg.vocab_size
        else:
            # Fallback small model for testing
            from transformers import LlamaConfig

            text_cfg = LlamaConfig()
            self.language_model = AutoModel.from_config(config=text_cfg)
            hidden_size = text_cfg.hidden_size
            vocab_size = text_cfg.vocab_size

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Packing + SP slice for the LLM micro-batch.

        Steps (when SP is enabled):
          1. ``sp_pad_and_slice`` pads the sequence length to be divisible by
             ``sp_size``, then returns this rank's contiguous slice.
          2. The same slice is applied to ``labels``, ``attention_mask``, and
             ``position_ids`` so all sequence-parallel tensors are aligned.
          3. ``und_image_embeds`` and other non-sequence tensors are passed
             through unchanged; the ``_scatter_image_embeds`` in ``forward``
             will naturally inject only the embeds whose placeholder tokens
             landed in this rank's sequence slice.

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
            if labels is not None:
                labels = sp_pad_and_slice(labels, dim=1, pad_value=-100)
            if attention_mask is not None:
                attention_mask = sp_pad_and_slice(attention_mask, dim=1, pad_value=0)
            if position_ids is not None:
                position_ids = sp_pad_and_slice(position_ids, dim=1, pad_value=0)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    def post_forward(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """SP gather: all-gather ``hidden_states`` across SP ranks.

        After gather, downstream connections (e.g. a DiT conditioner) receive
        the full-length hidden state sequence rather than the SP-local slice.
        The gather applies a ``1/sp_size`` gradient scaling on the backward
        pass to compensate for the all-reduce in SP loss averaging.
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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        und_image_embeds: Optional[torch.FloatTensor] = None,
        gen_image_embeds: Optional[torch.FloatTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Training forward.

        Builds the embedding sequence by:
          1. Looking up ``input_ids`` (unless ``inputs_embeds`` provided).
          2. Scattering ``und_image_embeds`` at understanding-image positions.
          3. Scattering ``gen_image_embeds`` at generation-image positions.
          4. Running the LLaMA backbone.
          5. Computing cross-entropy loss against ``labels``.
        """
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Inject understanding image embeddings
        if und_image_embeds is not None and input_ids is not None:
            inputs_embeds = self._scatter_image_embeds(
                inputs_embeds, input_ids, und_image_embeds, self.config.image_token_id
            )

        # Inject generation image embeddings
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
        hidden_states = lm_out.last_hidden_state  # (B, T, D)
        logits = self.lm_head(hidden_states)  # (B, T, V)

        lm_loss = None
        if labels is not None:
            lm_loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        out: Dict[str, Any] = {
            "hidden_states": hidden_states,
            "logits": logits,
            "past_key_values": lm_out.past_key_values,
        }
        if lm_loss is not None:
            out["lm_loss"] = lm_loss
        return out

    def generate_step(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Single auto-regressive decoding step.

        Accepts either ``input_ids`` (last generated token) or
        ``inputs_embeds`` (pre-computed embedding from ``vq_dec_to_ar``).
        Returns the sampled next token plus KV cache for the next step.
        """
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden = lm_out.last_hidden_state  # (B, 1, D)
        logits = self.lm_head(hidden)  # (B, 1, V)
        next_token_logits = logits[:, -1, :]  # (B, V)

        # Sample
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        if top_p < 1.0:
            next_token_logits = self._top_p_filter(next_token_logits, top_p)
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        return {
            "last_token_id": next_token.squeeze(-1),  # (B,) for FSM transition check
            "input_ids": next_token,  # feed back as next step input
            "logits": logits,
            "past_key_values": lm_out.past_key_values,
            "vq_token_id": next_token.squeeze(-1),  # alias for vq_decoder connection
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

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        return logits.scatter(1, sorted_indices, sorted_logits)

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
        """Extract LLM weights from a loaded JanusForConditionalGeneration."""
        text_cfg = janus_model.config.text_config
        cfg = JanusLLMConfig(
            text_config=text_cfg.to_dict() if hasattr(text_cfg, "to_dict") else vars(text_cfg),
        )
        llm = cls(cfg)
        llm.language_model.load_state_dict(janus_model.model.language_model.state_dict())
        llm.lm_head.load_state_dict(janus_model.lm_head.state_dict())
        return llm
