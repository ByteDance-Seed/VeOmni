"""VeOmni-accelerated TextEncoder — training / inference graph hooks."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_outputs,
    reduce_sequence_parallel_loss,
    slice_input_tensor,
    sp_pad,
)
from veomni.utils.tensor_utils import naflatten, unflatten

from ....mixins.base_mixin import BaseMixin
from ....mixins.emb_parallel_mixin import EmbParallelMixin
from ....mixins.inference_module_mixin import InferenceModuleMixin
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy, seal_outputs
from .chat_template import TextEncoderChatTemplate
from .configuration import TextEncoderConfig
from .modeling import TextEncoder


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


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks shared by every text encoder."""

    config: TextEncoderConfig
    device: torch.device
    _chat_template: TextEncoderChatTemplate
    _encode_batch_shape: torch.LongTensor | None

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        input_ids = self._prepare_encode_inputs(self._conversation_carrier)

        # Metering: this rank's OWN packed token count, stashed BEFORE the SP slice
        # below. The meter sums over the DP group only, so each rank reports just its
        # own tokens (not the SP peers that hold the same replicated sample) —
        # identical to the non-SP run for both SP-disabled and uniform SP.
        self.metric_meter_set_seqlens("encode", [int(input_ids.numel())])

        if get_parallel_state().sp_size > 1:
            # SP input-slice: hand this rank its ``1/sp_size`` slice of the packed
            # token sequence (wte is a per-token lookup, so slicing is exact —
            # mirrors VeOmni's whole-model SP where the embedding is
            # sequence-sharded). Every SP rank already holds the same ``input_ids``
            # (the dataloader replicates each shard); pad it to a multiple of
            # ``sp_size`` and take this rank's contiguous chunk. ``encode_post``
            # all-gathers the embeds back.
            self._sp_own_len = input_ids.size(0)
            input_ids = sp_pad(input_ids, dim=0, pad_value=0)
            input_ids = slice_input_tensor(input_ids, dim=0, padding=False, group=get_parallel_state().sp_group)
        return {"input_ids": input_ids}

    @post_forward("encode")
    def encode_post(self, inputs_embeds: torch.Tensor) -> Dict[str, Any]:
        if get_parallel_state().sp_size > 1:
            # SP output-gather: all-gather token shards back to the full sequence
            # (autograd-aware; backward sums grads across the SP group), then drop
            # the SP pad tail so the packed length matches its carrier below.
            inputs_embeds = gather_outputs(inputs_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            inputs_embeds = inputs_embeds.narrow(0, 0, self._sp_own_len)
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        batch_shape = self._encode_batch_shape
        self._encode_batch_shape = None
        scatter_text_encoder_embeds(conversation, unflatten(inputs_embeds, batch_shape))
        return {"conversation_list": conversation}

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        hidden_states, shift_labels = self._prepare_decode_inputs(self._conversation_carrier)
        # No SP sharding here: the lm-head loss is a token-weighted mean over the whole
        # ``dp_sp`` (``fsdp_group``) mesh (``modeling.decode`` →
        # ``reduce_sequence_parallel_loss``), so each rank scoring only its OWN span
        # yields the identical global loss + gradient regardless of the DP/SP split.
        # Skipping the gather-concat keeps peak logits at this rank's own span.
        return {"hidden_states": hidden_states, "shift_labels": shift_labels}

    @post_forward("decode")
    def decode_post(self, loss: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        # V2 single-loss protocol: drop logits, rename ``loss`` → ``_loss``.
        if loss is not None:
            return {"_loss": loss, "conversation_list": conversation}
        # TODO: scatter logits for rl training
        return {"conversation_list": conversation}

    def _prepare_encode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> torch.Tensor:
        input_ids: list[torch.Tensor] = []
        self._encode_batch_shape = None
        for sample in conversation_list or []:
            input_ids.extend(self._chat_template.pack_input_ids(sample))
        # ``naflatten`` keeps the shape on CPU (avoids the post-forward D2H sync);
        # the flat ids may be CPU (worker path) or device (fallback) — move once.
        input_ids, self._encode_batch_shape = naflatten(input_ids)
        input_ids = input_ids.to(self.device, non_blocking=True)
        return input_ids

    def _prepare_decode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []

        for sample in conversation_list:
            for part in sample:
                if is_dummy(part):
                    continue
                hidden_states = part.value
                if hidden_states.dim() == 3:
                    hidden_states = hidden_states.squeeze(0)
                if part.type == "text":
                    labels = part.meta["labels"]
                    assert labels.shape[0] == hidden_states.shape[0]
                    hidden_states_chunks.append(hidden_states)
                    label_chunks.append(labels)
                elif part.type in ("image", "video"):
                    # Vision segment carries projected patch embeds; keep one row
                    # (no label) so the sequence stays aligned, like the backbone.
                    hidden_states_chunks.append(hidden_states[-1:])
                    label_chunks.append(torch.full((1,), -100, dtype=torch.long))

        hidden_states = torch.cat(hidden_states_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)  # CPU

        labels = labels[..., 1:].contiguous()
        shift_labels = F.pad(labels, (0, 1), "constant", -100)
        # Single H2D move (labels are tiny host-side bookkeeping); the loss forward
        # orders after the backbone kernels on the GPU stream without a CPU stall.
        shift_labels = shift_labels.to(device=hidden_states.device, non_blocking=True)
        return hidden_states, shift_labels

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        self._encode_batch_shape: torch.LongTensor | None = None
        # Active sample's packed token count. Under SP the encode output-gather hook
        # (``encode_sp_post``) narrows the all-gathered (seq-padded) embeds to it.
        self._sp_own_len: Optional[int] = None


class InferenceMixin(InferenceModuleMixin):
    """Generation-FSM hooks shared by every text encoder."""

    config: TextEncoderConfig
    device: torch.device
    _chat_template: TextEncoderChatTemplate
    _tokenizer: Any
    _prompt_encoded: bool
    _text_token_cache: list[int]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._text_token_cache: list[int] = []
        self._bos_injected: bool = False
        # First FSM step embeds the whole (pre-templated, pre-tokenized) prompt;
        # later steps autoregress. Set once the prompt has been encoded.
        self._prompt_encoded: bool = False

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
        embeds them through :meth:`encode`, and scatters the segment embeds — the
        text-encoder twin of the per-step "pack → encode → scatter". Subsequent
        ``generate`` calls autoregress (``_prompt_encoded`` guards re-entry).
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

    @abstractmethod
    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Dict[str, Any] = dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """One FSM inference step — implemented per module.

        Inference differs substantially across text encoders (Qwen ChatML
        autoregression keyed on eos / ``<|im_end|>``; Janus T2I with ``<boi>`` /
        ``<eoi>`` + classifier-free guidance), so each concrete module owns its
        ``generate``. The base provides only the shared sampling / embedding
        helpers (:meth:`_sample_token`, :meth:`_token_id_tensor`,
        :func:`scatter_text_encoder_embeds`, :meth:`_flush_text_generated`).
        """
        raise NotImplementedError

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


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for the text encoder (wte + lm_head)."""

    config: TextEncoderConfig

    def estimate_flops(self, seqlens: List[int]) -> float:
        # This module owns wte (an embedding lookup ≈ 0 FLOPs) + the lm_head
        # projection (hidden → vocab); the transformer layers belong to the
        # backbone module. fwd+bwd ⇒ 6x; lm_head params = vocab * hidden.
        lm_head_n = self.config.vocab_size * self.config.hidden_size
        return 6 * lm_head_n * sum(seqlens) / 1e12

    # Token lengths come from ``metric_meter_set_seqlens`` in ``encode_pre`` (the
    # full pre-slice packed length); ``decode`` stashes nothing and contributes
    # no tokens (its lm_head FLOPs are covered by the ``encode`` count). So the
    # default ``metric_meter_token_lengths`` (drains the stash) is used as-is.


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceMixin, MeterMixin, EmbParallelMixin):
    """Shared training / inference plumbing for every text encoder.

    Concrete modules (janus / qwen3 / qwen3vl / bagel) subclass this and define
    ``XxxTextEncoderPreprocessor`` in their ``processing.py`` by subclassing
    :class:`~veomni.models.seed_omni.modules.base.text_encoder.processing.TextEncoderPreprocessor`
    and implementing :meth:`~veomni.models.seed_omni.modules.base.text_encoder.processing.TextEncoderPreprocessor.build_chat_template`
    with the module-local chat template.  Register via ``preprocessor_class`` on
    the family mixin, plus the ``encode_pre`` / ``encode_post`` / ``decode_pre`` /
    ``decode_post`` pass-through hooks in each module's ``accelerated.py``.
    """

    _chat_template: TextEncoderChatTemplate

    def get_parallel_plan(self):
        from .parallel_plan import get_parallel_plan as _get_parallel_plan

        return _get_parallel_plan()

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = TextEncoderChatTemplate(tokenizer)


class TextEncoderAccelerated(VeOmniMixin, TextEncoder):
    """Training/runtime text encoder — vocab-parallel embed + SP-aware CE."""

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb_parallel_lookup(self.embed_tokens, input_ids)

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.config.tie_word_embeddings:
            return self.lm_head(hidden_states)
        return self.emb_parallel_project(hidden_states, self.embed_tokens.weight)

    def decode(
        self,
        hidden_states: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        shift_labels: torch.LongTensor | None = None,
        **kwargs: Any,
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


__all__ = ["TextEncoderAccelerated", "scatter_text_encoder_embeds"]
