from typing import Any, Dict, List, Optional

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....utils.conversation import ConversationItem, maybe_merge_outputs
from ...base.text_encoder.modeling import TextEncoder
from .configuration import Qwen3VLTextEncoderConfig
from .processing import Qwen3VLTextEncoderPreprocessor


SIGNAL_TEXT_DONE = "text_done"


class Qwen3VLTextEncoder(TextEncoder):
    """Qwen3-VL ``TextEncoder`` — ChatML templating + tokenize + wte / lm_head.

    Image / video items (already carrying merged vision embeds from
    ``qwen3vl_vision``) pass through ``encode`` untouched: they keep their
    ``(N, D)`` value, get wrapped by ``<|vision_start|>`` / ``<|vision_end|>``
    text rows, and the backbone splices them in by segment order. Only ``text``
    rows are tokenized and embedded here. The ChatML ``generate`` FSM
    (autoregression keyed on eos / ``<|im_end|>``) is Qwen3-VL-specific;
    accelerated-only behavior (SP) lives in ``accelerated.py``.
    """

    config_class = Qwen3VLTextEncoderConfig
    preprocessor_class = Qwen3VLTextEncoderPreprocessor

    def __init__(self, config: Qwen3VLTextEncoderConfig):
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


__all__ = ["Qwen3VLTextEncoder", "SIGNAL_TEXT_DONE"]
