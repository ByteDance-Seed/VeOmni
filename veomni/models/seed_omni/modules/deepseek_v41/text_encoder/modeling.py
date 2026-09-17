# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4.1 token embedding + LM head (``deepseek_v41_text_encoder``).

The module is the generic :class:`TextEncoder` -- ``embed_tokens`` + an untied
``lm_head`` -- plus the V4.1 chat template and its autoregressive ``generate``.

It is also, by the G1 ruling, where the DSpark draft stages get their embedding
and their logits: ``mtp.*`` in the released checkpoint aliases the backbone's
``embed`` / ``head`` rather than carrying its own, so the draft path calls
:meth:`encode` / :meth:`decode` here instead of duplicating 1.32B of parameters
inside ``deepseek_v41_llm``. (``DSparkMarkovHead``'s own small rank-256 embed and
head are a separate thing and do live in the backbone module.)
"""

from typing import Any, Dict, List, Optional

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....utils.conversation import ConversationItem, maybe_merge_outputs
from ...base.text_encoder.modeling import TextEncoder
from .configuration import DeepseekV41TextEncoderConfig
from .processing import DeepseekV41TextEncoderPreprocessor


SIGNAL_TEXT_DONE = "text_done"


class DeepseekV41TextEncoder(TextEncoder):
    """V4.1 ``TextEncoder`` -- DeepSeek chat templating + tokenize + wte / lm_head.

    Image items pass through ``encode`` untouched: ``deepseek_v41_vision`` has
    already replaced their value with the span's ``(L, D)`` embeddings (delimiters
    included), and the backbone splices them by segment order. Only ``text`` rows
    are tokenized and embedded here.
    """

    config_class = DeepseekV41TextEncoderConfig
    preprocessor_class = DeepseekV41TextEncoderPreprocessor
    base_model_prefix = "deepseek_v41_text_encoder"

    def __init__(self, config: DeepseekV41TextEncoderConfig):
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
            # First step: the inference CPU preprocessor already templated and
            # tokenized the request (the V4.1 template ends on the assistant
            # header), so this only embeds and scatters.
            return self._encode_prompt(conversation_list)

        if tail.type != "output":
            raise ValueError(f"Invalid conversation tail type: {tail.type}")

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
        # The id, not just its embedding: deepseek_v41_ngram hashes token ids and
        # this is the only place the sampled one reaches the carrier. The key
        # matches what the chat template stashes on prompt text items, so the
        # n-gram node reads one field whether it is prefilling or stepping.
        #
        # After the merge, not before: merging rebuilds the tail, and a meta set
        # on the pre-merge item would be dropped on the step that merges -- the
        # n-gram node would then hash a stale id with nothing raising.
        conversation_list[-1].meta["input_ids"] = input_ids

        # V4.1 has one stop token: the turn ends on the end-of-sentence marker that
        # `_render_assistant` appends. There is no second ChatML-style `im_end`.
        if output_token_id == self._chat_template.eos_token_id:
            outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
            outputs["generated"] = self._flush_text_generated(conversation_list)
        return outputs


__all__ = ["DeepseekV41TextEncoder", "SIGNAL_TEXT_DONE"]
