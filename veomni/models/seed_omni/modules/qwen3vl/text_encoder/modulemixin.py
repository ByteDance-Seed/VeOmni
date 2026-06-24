from typing import Any, Dict, List, Optional

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....mixins.modulemixin import CPUPreprocessor, post_forward, pre_forward
from ....utils.conversation import ConversationItem, maybe_merge_outputs
from ...base.text_encoder.modulemixin import TextEncoderModuleMixin
from .chat_template import Qwen3VLChatTemplate


SIGNAL_TEXT_DONE = "text_done"


class Qwen3VLTextEncoderCPUPreprocessor(CPUPreprocessor):
    """Worker-side ``apply_chat_template`` → tokenize → merge for the Qwen3-VL text encoder.

    Holds only the (picklable) :class:`Qwen3VLChatTemplate` — never the model — so it
    runs in DataLoader workers and overlaps with GPU compute. Builds CPU tensors;
    the main process's thin ``encode_pre`` does the single ``.to(device)``.
    """

    def __init__(self, chat_template: Qwen3VLChatTemplate) -> None:
        self._chat_template = chat_template

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        for sample in conversation_list or []:
            parts = self._chat_template.tokenize_conversation(sample, add_generation_prompt=inference)
            sample.clear()
            sample.extend(parts)


class Qwen3VLTextEncoderModuleMixin(TextEncoderModuleMixin):
    """Qwen3-VL ``TextEncoder`` — ChatML templating + tokenize + wte / lm_head.

    Image / video items (already carrying merged vision embeds from
    ``qwen3vl_vision``) pass through ``encode`` untouched: they keep their
    ``(N, D)`` value, get wrapped by ``<|vision_start|>`` / ``<|vision_end|>``
    text rows, and the backbone splices them in by segment order. Only ``text``
    rows are tokenized and embedded here.

    The encode/decode plumbing (prepare / scatter) and the ChatML ``generate``
    live in :class:`TextEncoderModuleMixin`; the hooks + CPU preprocessor below are
    explicit pass-throughs (for findability). Only the chat template is model-specific.
    """

    _chat_template: Qwen3VLChatTemplate

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = Qwen3VLChatTemplate(tokenizer)

    def build_cpu_preprocessor(self) -> Optional[CPUPreprocessor]:
        """Worker-side chat-template + tokenize (see :class:`Qwen3VLTextEncoderCPUPreprocessor`)."""
        if getattr(self, "_chat_template", None) is None:
            return None
        return Qwen3VLTextEncoderCPUPreprocessor(self._chat_template)

    # training hooks (explicit pass-through to TextEncoderModuleMixin for findability)
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

    # inference hooks — ChatML autoregression keyed on eos / <|im_end|>
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


__all__ = ["Qwen3VLTextEncoderModuleMixin"]
