from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from veomni.utils.tensor_utils import unflatten

from ....conversation import ConversationItem, seal_outputs
from ....module import ModuleMixin, post_forward
from ....tracemixin import TraceMixin
from .configuration import TextEncoderConfig


_SAMPLING_KWARGS = ("temperature", "top_p", "do_sample")


class TextEncoderModuleMixin(ModuleMixin):
    def init_omni_state(self) -> None:
        # Training state
        self._conversation_carrier: Any = None
        self._encode_batch_shape: torch.LongTensor | None = None

        # Inference state
        self._text_token_cache: list[int] = []
        self._bos_injected: bool = False

    def get_parallel_plan(self):
        from .parallel_plan import get_parallel_plan as _get_parallel_plan

        return _get_parallel_plan()

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    # training hooks
    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        batch_shape = self._encode_batch_shape
        self._encode_batch_shape = None
        inputs_embeds = outputs.get("inputs_embeds")
        if conversation is not None and inputs_embeds is not None and batch_shape is not None:
            self._scatter_text_embeds(conversation, unflatten(inputs_embeds, batch_shape))
        return {"conversation_list": conversation}

    @post_forward("decode")
    def decode_post(self, **outputs: Any) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        # V2 single-loss protocol: drop logits, rename ``loss`` → ``_loss``.
        outputs.pop("logits", None)
        loss = outputs.pop("loss", None)
        if loss is not None:
            outputs["_loss"] = loss
        outputs["conversation_list"] = conversation
        return outputs

    def _scatter_text_embeds(
        self,
        conversation_list: list[list[ConversationItem]],
        segment_embeds: list[torch.Tensor],
    ) -> None:
        dtype = self.dtype
        segment_embeds_iterator = iter(segment_embeds)
        for sample in conversation_list:
            for part in sample:
                if part.type != "text":
                    continue
                part.value = next(segment_embeds_iterator).to(device=self.device, dtype=dtype)
        if next(segment_embeds_iterator, None) is not None:
            raise RuntimeError("TextEncoder text segment count mismatch during embed scatter.")

    # inference hooks
    def reset_local_inference_state(self) -> None:
        self._text_token_cache.clear()

    def reset_global_inference_state(self) -> None:
        self.reset_local_inference_state()
        self._bos_injected = False

    def generate(self, conversation_list: list[list[ConversationItem]], **generation_kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("TextEncoderModuleMixin.generate is not implemented")

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
