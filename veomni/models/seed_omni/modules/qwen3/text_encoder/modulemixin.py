from typing import Any, Dict, List, Optional

import torch

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....mixins.modulemixin import CPUPreprocessor, post_forward, pre_forward
from ....utils.conversation import ConversationItem, maybe_merge_outputs
from ...base.text_encoder.modulemixin import TextEncoderModuleMixin
from ...qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from .chat_template import Qwen3ChatTemplate
from .configuration import Qwen3TextEncoderConfig


SIGNAL_TEXT_DONE = "text_done"


class Qwen3TextEncoderCPUPreprocessor(CPUPreprocessor):
    """Worker-side ``apply_chat_template`` → tokenize → merge for the Qwen3 text encoder.

    Holds only the (picklable) chat template (text-only or the reused Qwen3-VL image
    template) — never the model — so it runs in DataLoader workers and overlaps with
    GPU compute. Builds CPU tensors; the main process's thin ``encode_pre`` does the
    single ``.to(device)``.
    """

    def __init__(self, chat_template: Qwen3ChatTemplate | Qwen3VLChatTemplate) -> None:
        self._chat_template = chat_template

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        for sample in conversation_list or []:
            parts = self._chat_template.tokenize_conversation(sample, add_generation_prompt=inference)
            sample.clear()
            sample.extend(parts)


class Qwen3TextEncoderModuleMixin(TextEncoderModuleMixin):
    """Qwen3 ChatML text encoder, optionally image-aware.

    With ``config.enable_image`` the module uses the Qwen3-VL image ChatML template
    (image items wrapped in ``<|vision_start|> … <|vision_end|>``; the sibling
    vision module supplies the projected patch embeds) and handles image/video
    parts in decode — enough to bootstrap a text-only Qwen3 into image
    understanding. In that mode :meth:`freeze_model` trains only the vision
    special-token embedding rows, whose ids it resolves from its own tokenizer
    (the user can't know them, but the module can). With it off the behaviour is
    the original text-only Qwen3 path.

    The encode/decode plumbing (prepare / scatter) and the ChatML ``generate`` live
    in :class:`TextEncoderModuleMixin`; the hooks + CPU preprocessor below are explicit
    pass-throughs (for findability). Only the chat-template selection and the
    image-mode freeze are genuinely Qwen3-specific.
    """

    config: Qwen3TextEncoderConfig
    _chat_template: Qwen3ChatTemplate | Qwen3VLChatTemplate

    # Vision special tokens whose embedding rows bootstrap image understanding;
    # ids are resolved from the tokenizer at freeze time (see :meth:`freeze_model`).
    _VISION_SPECIAL_TOKENS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>")

    def init_omni_state(self) -> None:
        super().init_omni_state()
        self._trainable_row_mask: Optional[torch.Tensor] = None

    @property
    def _enable_image(self) -> bool:
        return self.config.enable_image

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        # Only the template differs: image mode reuses the Qwen3-VL ChatML template
        # (adds the vision wrap tokens); otherwise the text-only Qwen3 ChatML.
        if self._enable_image:
            self._chat_template = Qwen3VLChatTemplate(tokenizer)
        else:
            self._chat_template = Qwen3ChatTemplate(tokenizer)

    def build_cpu_preprocessor(self) -> Optional[CPUPreprocessor]:
        """Worker-side chat-template + tokenize (see :class:`Qwen3TextEncoderCPUPreprocessor`)."""
        if getattr(self, "_chat_template", None) is None:
            return None
        return Qwen3TextEncoderCPUPreprocessor(self._chat_template)

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

    # ── Freeze: in image mode, train only the vision special-token rows ──────
    def freeze_model(self) -> None:
        if not self._enable_image:
            return  # fully trainable (default text-only behaviour)
        # The user can't know the vision special-token ids, but the module can:
        # resolve them from its own tokenizer so only those rows stay trainable.
        ids = [int(self._tokenizer.convert_tokens_to_ids(tok)) for tok in self._VISION_SPECIAL_TOKENS]
        weight = self.embed_tokens.weight
        weight.requires_grad_(True)
        keep = torch.zeros(weight.shape[0], dtype=torch.bool)
        keep[ids] = True
        self._trainable_row_mask = keep

        def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
            mask = self._trainable_row_mask.to(device=grad.device)
            return grad * mask.unsqueeze(1).to(grad.dtype)

        weight.register_hook(_mask_grad)


__all__ = ["Qwen3TextEncoderModuleMixin"]
