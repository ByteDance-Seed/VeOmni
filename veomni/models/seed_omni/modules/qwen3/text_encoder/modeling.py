from typing import Any, Dict, List, Optional

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....utils.conversation import ConversationItem, maybe_merge_outputs
from ...base.text_encoder.modeling import TextEncoder
from .configuration import Qwen3TextEncoderConfig
from .processing import Qwen3TextEncoderPreprocessor


SIGNAL_TEXT_DONE = "text_done"


class Qwen3TextEncoder(TextEncoder):
    """Qwen3 ChatML text encoder, optionally image-aware.

    With ``config.enable_image`` the module uses the Qwen3-VL image ChatML
    template (image items wrapped in ``<|vision_start|> … <|vision_end|>``; the
    sibling vision module supplies the projected patch embeds) — the CPU
    preprocessor (see :class:`Qwen3TextEncoderPreprocessor`) picks the right
    template at bind time. Only the ChatML autoregression (keyed on eos /
    ``<|im_end|>``) is implemented here; accelerated-only behavior (SP,
    vision-freeze) lives in ``accelerated.py``.
    """

    config_class = Qwen3TextEncoderConfig
    preprocessor_class = Qwen3TextEncoderPreprocessor

    def __init__(self, config: Qwen3TextEncoderConfig):
        super().__init__(config)

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


__all__ = ["Qwen3TextEncoder", "SIGNAL_TEXT_DONE"]
