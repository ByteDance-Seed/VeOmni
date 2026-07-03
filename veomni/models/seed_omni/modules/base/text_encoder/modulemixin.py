from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_outputs,
    slice_input_tensor,
    sp_gather_seqs,
    sp_pad,
    sp_take_own_seq,
)
from veomni.utils.tensor_utils import naflatten, unflatten

from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.modulemixin import ModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy, seal_outputs
from .chat_template import TextEncoderChatTemplate
from .configuration import TextEncoderConfig


_SAMPLING_KWARGS = ("temperature", "top_p", "do_sample")


class TextEncoderModuleMixin(ModuleMixin):
    """Shared training / inference plumbing for every text encoder.

    Concrete modules (janus / qwen3 / qwen3vl) subclass this and, for
    discoverability, explicitly re-declare their own ``XxxTextEncoderCPUPreprocessor``
    + ``build_cpu_preprocessor`` and the ``encode_pre`` / ``encode_post`` /
    ``decode_pre`` / ``decode_post`` pass-through hooks (mirroring the per-module
    image preprocessors), so a reader finds each module's worker-prep and call-site
    code in its own file rather than only here.
    """

    _chat_template: TextEncoderChatTemplate

    def init_omni_state(self) -> None:
        # Training state
        self._conversation_carrier: Any = None
        self._encode_batch_shape: torch.LongTensor | None = None
        # SP state for the wte ``encode`` pass (mirrors the backbone), created only
        # when this module runs SP: the combined SP-group packed length before the
        # seq pad/slice, the per-rank lengths, and this rank's index within the SP
        # group — used by ``encode_post`` to gather the embeds back and narrow to
        # this rank's own segment.
        if get_parallel_state().sp_enabled:
            self._encode_sp_seqlen: Optional[int] = None
            self._encode_sp_rep_lengths: Optional[List[int]] = None
            self._encode_sp_group_index: int = 0

        # Inference state
        self._text_token_cache: list[int] = []
        self._bos_injected: bool = False
        # First FSM step embeds the whole (pre-templated, pre-tokenized) prompt;
        # later steps autoregress. Set once the prompt has been encoded.
        self._prompt_encoded: bool = False

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

    # training hooks
    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        input_ids = self._prepare_encode_inputs(self._conversation_carrier)

        # Metering: this rank's OWN packed token count, pre-gather / pre-slice.
        # The meter sums over the DP group only, so each rank reports just its own
        # tokens (not the module_sp ranks SP aggregates) — identical to the non-SP
        # run for both SP-disabled and per-module SP.
        self.metric_meter_set_seqlens("encode", [int(input_ids.numel())])

        ps = get_parallel_state()
        if ps.sp_enabled:
            # SP the wte lookup, matching VeOmni's whole-model SP (where the
            # embedding is sequence-sharded too). ``input_ids`` is the flat packed
            # token sequence; wte is a per-token lookup, so slicing it is exact.
            # Mirror the backbone: aggregate the module SP group's distinct
            # sequences into the combined module-group sequence, pad to a multiple
            # of module sp_size, and embed only this rank's slice. ``encode_post``
            # all-gathers the embeds and narrows to this rank's own segment before
            # the per-item scatter (the carrier stays in the per-rank layout
            # between graph nodes). wte params are sharded on the dp_shard_sp mesh,
            # so FSDP's /sp grad averaging is already compensated by the loss-level
            # ``reduce_sequence_parallel_loss`` (x sp) — no extra scaling here.
            input_ids, rep_lengths, group_index = sp_gather_seqs(input_ids, dim=0)
            self._encode_sp_rep_lengths = rep_lengths
            self._encode_sp_group_index = group_index
            self._encode_sp_seqlen = input_ids.size(0)
            input_ids = sp_pad(input_ids, dim=0, pad_value=0)
            input_ids = slice_input_tensor(input_ids, dim=0, padding=False, group=ps.sp_group)
        return {"input_ids": input_ids}

    @post_forward("encode")
    def encode_post(self, inputs_embeds: torch.Tensor) -> Dict[str, Any]:
        ps = get_parallel_state()
        if ps.sp_enabled and self._encode_sp_seqlen is not None:
            # Gather this rank's embed slice back into the combined module-group
            # sequence over the MODULE SP group, drop the SP pad tail
            # (autograd-aware), then narrow to this rank's own segment so the
            # carrier returns to the per-rank layout expected downstream.
            inputs_embeds = gather_outputs(
                inputs_embeds,
                gather_dim=0,
                padding_dim=0,
                unpad_dim_size=self._encode_sp_seqlen,
                group=ps.sp_group,
            )
            inputs_embeds = sp_take_own_seq(
                inputs_embeds,
                dim=0,
                seg_lengths=self._encode_sp_rep_lengths,
                sp_rank=self._encode_sp_group_index,
            )
            self._encode_sp_seqlen = None
            self._encode_sp_rep_lengths = None
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        batch_shape = self._encode_batch_shape
        self._encode_batch_shape = None
        self._scatter_text_embeds(conversation, unflatten(inputs_embeds, batch_shape))
        return {"conversation_list": conversation}

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        hidden_states, shift_labels = self._prepare_decode_inputs(self._conversation_carrier)
        ps = get_parallel_state()
        if ps.sp_enabled:
            # Shard the loss span across the MODULE SP group. The backbone already
            # restored its hidden states to the per-rank layout, so first aggregate
            # the module SP group's distinct spans into the combined sequence
            # (autograd-aware), then pad to a multiple of module sp_size (labels
            # pad with -100 so the tail is ignored) and score only this rank's
            # slice. ``modeling.decode`` aggregates the per-rank CE via
            # ``reduce_sequence_parallel_loss`` over the module SP group (which
            # also undoes FSDP's /sp averaging on the dp_shard_sp mesh).
            # ``shift_labels`` is already shifted on the FULL sequence, so slicing
            # stays correct. The loss is a scalar — no per-rank restore needed.
            hidden_states, _, _ = sp_gather_seqs(hidden_states, dim=0)
            shift_labels, _, _ = sp_gather_seqs(shift_labels, dim=0)
            hidden_states = sp_pad(hidden_states, dim=0, pad_value=0)
            shift_labels = sp_pad(shift_labels, dim=0, pad_value=-100)
            hidden_states = slice_input_tensor(hidden_states, dim=0, padding=False, group=ps.sp_group)
            shift_labels = slice_input_tensor(shift_labels, dim=0, padding=False, group=ps.sp_group)
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

    def _scatter_text_embeds(
        self,
        conversation_list: list[list[ConversationItem]],
        segment_embeds: list[torch.Tensor],
    ) -> None:
        segment_embeds_iterator = iter(segment_embeds)
        for sample in conversation_list:
            for part in sample:
                if part.type != "text":
                    continue
                part.value = next(segment_embeds_iterator)
        if next(segment_embeds_iterator, None) is not None:
            raise RuntimeError("TextEncoder text segment count mismatch during embed scatter.")

    # inference hooks
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
        self._scatter_text_embeds([conversation_list], unflatten(inputs_embeds, batch_shape))
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
        :meth:`_scatter_text_embeds`, :meth:`_flush_text_generated`).
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


class TextEncoderMetricMeterMixin(MetricMeterMixin):
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


__all__ = ["TextEncoderModuleMixin", "TextEncoderMetricMeterMixin"]
