from typing import Any, Dict, List, Optional

import torch

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....mixins.training_module_mixin import post_forward, pre_forward
from ....utils.conversation import ConversationItem, maybe_merge_outputs
from ...base.text_encoder.modulemixin import (
    InferenceMixin as BaseInferenceMixin,
)
from ...base.text_encoder.modulemixin import (
    TrainingMixin as BaseTrainingMixin,
)
from ...base.text_encoder.modulemixin import (
    VeOmniMixin as BaseVeOmniMixin,
)
from .chat_template import Qwen3VLChatTemplate
from .configuration import Qwen3VLTextEncoderConfig
from .processing import Qwen3VLTextEncoderPreprocessor


SIGNAL_TEXT_DONE = "text_done"


class TrainingMixin(BaseTrainingMixin):
    config: Qwen3VLTextEncoderConfig
    device: torch.device
    _chat_template: Qwen3VLChatTemplate

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return super().encode_pre(conversation_list, **kwargs)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().encode_post(**outputs)

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return super().decode_pre(conversation_list, **kwargs)

    @post_forward("decode")
    def decode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().decode_post(**outputs)


class InferenceMixin(BaseInferenceMixin):
    config: Qwen3VLTextEncoderConfig
    device: torch.device
    _chat_template: Qwen3VLChatTemplate
    _prompt_encoded: bool
    _text_token_cache: list[int]

    def encode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """IDE stub — implemented on :class:`Qwen3VLTextEncoder` in ``modeling.py``."""
        ...

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Dict[str, Any] = dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        tail = conversation_list[-1]
        if not self._prompt_encoded:
            # First step: the request was already templated + tokenized by the
            # inference CPU preprocessor (run before the FSM), so just embed + scatter.
            return self._encode_prompt(conversation_list)

        if tail.type == "output":
            outputs: Dict[str, Any] = {"conversation_list": conversation_list}
            hidden_states = tail.value
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            sampling = self._extract_sampling_kwargs(generation_kwargs, 1.0, 1.0, {})
            output_token_id = self._sample_token(hidden_states, **sampling)
            self._text_token_cache.append(output_token_id)
            input_ids = self._token_id_tensor(output_token_id)
            inputs_embeds = self.encode(input_ids)["inputs_embeds"]

            tail.value = inputs_embeds
            maybe_merge_outputs(conversation_list)

            if output_token_id in (self._chat_template.eos_token_id, self._chat_template.im_end_token_id):
                outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
                outputs["generated"] = self._flush_text_generated(conversation_list)
            return outputs

        raise ValueError(f"Invalid conversation tail type: {tail.type}")


class VeOmniMixin(TrainingMixin, InferenceMixin, BaseVeOmniMixin):
    """Qwen3-VL ``TextEncoder`` — ChatML templating + tokenize + wte / lm_head.

    Image / video items (already carrying merged vision embeds from
    ``qwen3vl_vision``) pass through ``encode`` untouched: they keep their
    ``(N, D)`` value, get wrapped by ``<|vision_start|>`` / ``<|vision_end|>``
    text rows, and the backbone splices them in by segment order. Only ``text``
    rows are tokenized and embedded here.

    The encode/decode plumbing (prepare / scatter) lives in
    :class:`BaseVeOmniMixin`; the hooks below are explicit pass-throughs
    (for findability). Only the chat template and the ChatML ``generate`` FSM
    (autoregression keyed on eos / ``<|im_end|>``) are Qwen3-VL-specific.
    """

    _chat_template: Qwen3VLChatTemplate
    preprocessor_class = Qwen3VLTextEncoderPreprocessor

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = Qwen3VLChatTemplate(tokenizer)


__all__ = ["VeOmniMixin"]
