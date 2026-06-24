from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from veomni.utils.tensor_utils import naflatten, unflatten

from ....mixins.modulemixin import ModuleMixin, post_forward, pre_forward
from ....mixins.tracemixin import TraceMixin
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
        return {"input_ids": input_ids}

    @post_forward("encode")
    def encode_post(self, inputs_embeds: torch.Tensor) -> Dict[str, Any]:
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


class TextEncoderTraceMixin(TraceMixin):
    """Per-module training-trace for the text encoder (wte + lm_head)."""

    config: TextEncoderConfig

    def estimate_flops(self, seqlens: List[int]) -> float:
        # This module owns wte (an embedding lookup ≈ 0 FLOPs) + the lm_head
        # projection (hidden → vocab); the transformer layers belong to the
        # backbone module. fwd+bwd ⇒ 6x; lm_head params = vocab * hidden.
        lm_head_n = self.config.vocab_size * self.config.hidden_size
        return 6 * lm_head_n * sum(seqlens) / 1e12

    def trace_token_lengths(self, method: str, data: Dict[str, Any]) -> List[int]:
        # Count once, on encode: `input_ids` is the full packed sequence
        # (pre-LLM, never SP-sliced → SP-safe). The decode pass runs lm_head over
        # the same sequence, so its FLOPs are already covered by this count;
        # decode itself contributes nothing (returns []).
        if method != "encode":
            return []
        input_ids = data.get("input_ids")
        if input_ids is None:
            return []
        return [int(input_ids.numel())]


__all__ = ["TextEncoderModuleMixin", "TextEncoderTraceMixin"]
