"""Generic word-token embedding (``wte``) + LM head as a graph node.

``TextEncoder(InferenceMixin, OmniPreTrainedModel)`` — mirrors HF's own
``PreTrainedModel`` + ``GenerationMixin`` split: ``OmniPreTrainedModel``
owns weights / ``forward`` (here, ``encode`` / ``decode``, which mirror a
VQ codec pre/post stage so the backbone stays vocab-agnostic), while
``InferenceMixin`` owns the FSM ``generate`` sampling helpers shared by every
text encoder (``_sample_token`` / ``_encode_prompt`` / ``_flush_text_generated``
/ …). Each family's ``modeling.py`` subclasses ``TextEncoder`` and implements
the concrete ``generate()`` (ChatML autoregression, T2I signal emission, …)
plus its own chat template. VeOmni training-graph hooks (``encode_pre`` /
``decode_post`` / SP-awareness) live in ``modules/<family>/text_encoder/accelerated.py``.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from veomni.utils.tensor_utils import naflatten, unflatten

from ....omni_pretrained_model import OmniPreTrainedModel
from ....utils.conversation import ConversationItem, seal_outputs
from .chat_template import TextEncoderChatTemplate
from .configuration import TextEncoderConfig


_SAMPLING_KWARGS = ("temperature", "top_p", "do_sample")


def scatter_text_encoder_embeds(
    conversation_list: list[list[ConversationItem]],
    segment_embeds: list[torch.Tensor],
) -> None:
    """Write packed text embeds back onto conversation text segments."""
    segment_embeds_iterator = iter(segment_embeds)
    for sample in conversation_list:
        for part in sample:
            if part.type != "text":
                continue
            part.value = next(segment_embeds_iterator)
    if next(segment_embeds_iterator, None) is not None:
        raise RuntimeError("TextEncoder text segment count mismatch during embed scatter.")


class InferenceMixin:
    """FSM ``generate`` + shared sampling helpers — analogous to HF's ``GenerationMixin``.

    Inference differs substantially across text encoders (Qwen ChatML
    autoregression keyed on eos / ``<|im_end|>``; Janus T2I with ``<boi>`` /
    ``<eoi>`` + classifier-free guidance), so each concrete module owns its
    ``generate``. This mixin only provides the shared sampling / embedding
    helpers below, plus the FSM inference state initialized in
    :meth:`TextEncoder.__init__`.

    Listed *before* :class:`~....omni_pretrained_model.OmniPreTrainedModel` in
    :class:`TextEncoder`'s bases: ``OmniPreTrainedModel`` ships no-op
    ``reset_local_inference_state`` / ``finalize`` defaults (kept so
    inference-only modules that don't mix this in still satisfy the FSM
    runtime's unconditional ``module.finalize(ctx=...)`` call), and MRO
    resolves left-to-right — put second, those no-ops would shadow the real
    implementations below.
    """

    def reset_local_inference_state(self) -> None:
        self._text_token_cache.clear()

    def reset_global_inference_state(self) -> None:
        self.reset_local_inference_state()
        self._bos_injected = False
        self._prompt_encoded = False

    def _encode_prompt(self, conversation_list: List[ConversationItem]) -> Dict[str, Any]:
        """First FSM step: embed the already-prepared prompt + scatter back.

        The inference CPU preprocessor (run before the FSM, mirroring training's
        collator) has already applied the chat template, appended the generation
        prompt and tokenized every text row, so this only packs the token ids,
        embeds them through :meth:`~TextEncoder.encode`, and scatters the segment
        embeds — the text-encoder twin of the per-step "pack → encode → scatter".
        Subsequent ``generate`` calls autoregress (``_prompt_encoded`` guards
        re-entry).
        """
        self._prompt_encoded = True
        for part in conversation_list:
            part.meta.pop("labels", None)
        input_ids = self._chat_template.pack_input_ids(conversation_list)
        input_ids, batch_shape = naflatten(input_ids)
        input_ids = input_ids.to(self.device)
        inputs_embeds = self.encode(input_ids)["inputs_embeds"]
        scatter_text_encoder_embeds([conversation_list], unflatten(inputs_embeds, batch_shape))
        return {"conversation_list": conversation_list}

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Dict[str, Any] = dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """One FSM inference step — implemented per module (see class docstring)."""
        del conversation_list, generation_kwargs, kwargs
        raise NotImplementedError(f"{type(self).__name__} must implement generate().")

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative - sorted_probs > top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        return logits.scatter(1, sorted_indices, sorted_logits)

    def _sample_token(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        **kwargs,
    ) -> int:
        del kwargs
        hidden_states = hidden_states.to(self.device)
        last = hidden_states[:, -1, :]
        logits = self._project(last) if last.dim() == 2 else self._project(last.squeeze(0))
        if not do_sample:
            return int(logits.argmax(dim=-1).item())
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        if top_p < 1.0:
            logits = self._top_p_filter(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return token

    def _token_id_tensor(self, token_id: int) -> torch.Tensor:
        device = self.device
        return torch.tensor([[token_id]], dtype=torch.long, device=device)

    @staticmethod
    def _extract_sampling_kwargs(
        generation_kwargs: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {"temperature": temperature, "top_p": top_p, "do_sample": True}
        if generation_kwargs:
            for k in _SAMPLING_KWARGS:
                if k in generation_kwargs:
                    merged[k] = generation_kwargs[k]
        for k in _SAMPLING_KWARGS:
            if k in kwargs:
                merged[k] = kwargs[k]
        return merged

    def finalize(self, *, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not self._text_token_cache:
            return {}
        flushed = self._flush_text_generated(ctx["conversation_list"])
        if not flushed:
            return {}
        return {"generated": flushed}

    def _flush_text_generated(self, conversation_list: List[ConversationItem]) -> Dict[str, Any]:
        token_ids = list(self._text_token_cache)
        self._text_token_cache.clear()
        if not token_ids:
            return {}
        meta = {"token_ids": token_ids}
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        seal_outputs(conversation_list, new_type="text")
        return {"type": "text", "value": text, "meta": meta}


class TextEncoder(InferenceMixin, OmniPreTrainedModel):
    """Word-token embedding + LM head, plus shared FSM ``generate`` sampling helpers."""

    config_class = TextEncoderConfig
    base_model_prefix = ""
    _no_split_modules: list = ["Embedding"]
    main_input_name = "input_ids"

    def __init__(self, config: TextEncoderConfig):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._chat_template: Optional[TextEncoderChatTemplate] = None
        # FSM inference state (see ``InferenceMixin.generate`` / ``_encode_prompt``):
        # first step embeds the whole (pre-templated, pre-tokenized) prompt; later
        # steps autoregress.
        self._prompt_encoded: bool = False
        self._text_token_cache: list[int] = []
        self._bos_injected: bool = False
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
        del kwargs
        logits = self._project(hidden_states)
        loss: torch.Tensor | None = None

        # Local, parallel-state-free CE: kernel selection and the DP+SP token
        # weighting are the runtime's business, so ``accelerated.py`` overrides
        # this with the fused ops dispatch (see ``TrainingMixin.decode``).
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


__all__ = ["InferenceMixin", "TextEncoder", "scatter_text_encoder_embeds"]
