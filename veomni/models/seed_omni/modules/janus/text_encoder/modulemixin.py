from typing import Any, Dict, List, Optional

import torch

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....mixins.modulemixin import CPUPreprocessor, post_forward, pre_forward
from ....utils.conversation import ConversationItem, maybe_merge_outputs
from ...base.text_encoder.modulemixin import TextEncoderModuleMixin
from .chat_template import JanusChatTemplate


# Janus signal keys
SIGNAL_START_IMAGE_GEN = "start_image_gen"
SIGNAL_TEXT_DONE = "text_done"


class JanusTextEncoderCPUPreprocessor(CPUPreprocessor):
    """Worker-side ``apply_chat_template`` → tokenize → merge for the Janus text encoder.

    Holds only the (picklable) :class:`JanusChatTemplate` — never the model — so it
    runs in DataLoader workers and overlaps with GPU compute. Builds CPU tensors;
    the main process's thin ``encode_pre`` does the single ``.to(device)``.
    """

    def __init__(self, chat_template: JanusChatTemplate) -> None:
        self._chat_template = chat_template

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        for sample in conversation_list or []:
            parts = self._chat_template.tokenize_conversation(sample, add_generation_prompt=inference)
            sample.clear()
            sample.extend(parts)


class JanusTextEncoderModuleMixin(TextEncoderModuleMixin):
    """Janus ``TextEncoder`` — image-aware ChatML with ``<boi>`` / ``<eoi>`` emitters.

    The encode/decode call-site plumbing (prepare / scatter) lives in
    :class:`TextEncoderModuleMixin`; the hooks + CPU preprocessor below are explicit
    pass-throughs (for findability). Only the chat template and the T2I-aware
    ``generate`` FSM (BOS injection, ``<boi>`` / ``<eoi>`` signals, classifier-free
    guidance arming) are genuinely Janus-specific.
    """

    _chat_template: JanusChatTemplate

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = JanusChatTemplate(tokenizer)

    def build_cpu_preprocessor(self) -> Optional[CPUPreprocessor]:
        """chat-template + tokenize (see :class:`JanusTextEncoderCPUPreprocessor`)."""
        return JanusTextEncoderCPUPreprocessor(self._chat_template)

    # training hooks
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

    # inference hooks
    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Dict[str, Any] = dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        tail = conversation_list[-1]
        if not self._prompt_encoded:
            # First step: the request was already templated + tokenized by the
            # inference CPU preprocessor (run before the FSM), so just embed + scatter.
            return self._encode_prompt(conversation_list)

        if tail.type == "output":
            outputs: Dict[str, Any] = {"conversation_list": conversation_list}
            hidden_states: torch.Tensor = tail.value
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            sampling = self._extract_sampling_kwargs(generation_kwargs, 1.0, 1.0, kwargs)
            output_token_id = self._sample_token(hidden_states, **sampling)
            self._text_token_cache.append(output_token_id)
            input_ids = self._token_id_tensor(output_token_id)
            inputs_embeds = self.encode(input_ids)["inputs_embeds"]

            tail.value = inputs_embeds
            maybe_merge_outputs(conversation_list)

            if output_token_id == self._chat_template.boi_token_id:
                self._maybe_arm_cfg_for_image_gen(conversation_list, generation_kwargs, outputs)
                outputs[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
            elif output_token_id == self._chat_template.eos_token_id:
                outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE

            if FSM_SIGNAL_KEY in outputs and outputs[FSM_SIGNAL_KEY] in (SIGNAL_TEXT_DONE, SIGNAL_START_IMAGE_GEN):
                outputs["generated"] = self._flush_text_generated(conversation_list)
            return outputs

        raise ValueError(f"Invalid type: {tail.type}")

    def emit_image_start(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert conversation_list[-1].type == "output" and conversation_list[-1].value.shape[0] == 1
        conversation_list.pop()
        output_token_id = self._chat_template.boi_token_id
        input_ids = self._token_id_tensor(output_token_id)
        inputs_embeds = self.encode(input_ids)["inputs_embeds"]
        cfg_uncond_inputs_embeds = self._maybe_arm_cfg_for_image_gen(
            conversation_list,
            generation_kwargs,
        )
        conversation_list.append(
            ConversationItem(
                type="output",
                value=inputs_embeds,
                role="assistant",
                meta={
                    "cfg_uncond_inputs_embeds": cfg_uncond_inputs_embeds,
                },
            )
        )
        return {"conversation_list": conversation_list}

    def emit_image_end(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        assert conversation_list[-1].type == "output" and conversation_list[-1].value.shape[-2] == 1
        conversation_list.pop()
        output_token_id = self._chat_template.eoi_token_id
        input_ids = self._token_id_tensor(output_token_id)
        inputs_embeds = self.encode(input_ids)["inputs_embeds"]
        conversation_list.append(
            ConversationItem(
                type="output",
                value=inputs_embeds,
                role="assistant",
                meta={
                    "collapse_cfg": True,
                },
            )
        )
        return {"conversation_list": conversation_list}

    def _maybe_arm_cfg_for_image_gen(
        self,
        conversation_list: Optional[List[ConversationItem]],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> None:
        cfg_w = generation_kwargs.get("guidance_scale")
        if cfg_w is None or float(cfg_w) <= 1.0:
            return
        uncond = self._build_cfg_uncond_embeds(conversation_list)
        return uncond

    def _build_cfg_uncond_embeds(
        self,
        conversation_list: List[ConversationItem],
    ) -> Optional[torch.Tensor]:
        bos_id = self._chat_template.bos_token_id
        pad_id = self._chat_template.pad_token_id

        input_ids: List[int] = [bos_id]
        assert conversation_list[0].type == "text"
        input_ids.extend([pad_id] * (len(conversation_list[0].value) - 1))

        for part in conversation_list[1:]:
            if part.type == "output":
                break
            input_ids.extend([pad_id] * len(part.value))

        uncond_inputs_embeds = self._embed_tokens(torch.tensor(input_ids, dtype=torch.long, device=self.device))
        return uncond_inputs_embeds


__all__ = ["JanusTextEncoderModuleMixin"]
