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
``embed`` / ``head`` rather than carrying its own. ``generate`` writes the
noise-padded draft block onto the carrier; ``draft`` samples it with the LM
head plus the Markov residual from the last MTP stage.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

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
        self._dspark_queue: List[int] = []

    def reset_global_inference_state(self) -> None:
        super().reset_global_inference_state()
        self._dspark_queue = []

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
        spec_rest: List[int] = []
        if self._dspark_queue:
            proposed = self._dspark_queue.pop(0)
            if proposed == output_token_id and output_token_id != self.config.eos_token_id:
                spec_rest = list(self._dspark_queue)
            self._dspark_queue.clear()
        self._text_token_cache.append(output_token_id)
        token_ids = [output_token_id, *spec_rest]
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        inputs_embeds = self.encode(input_ids)["inputs_embeds"]

        tail.value = inputs_embeds
        # Concatenate only consecutive *sampled* output spans. Prefill writes a
        # hidden-state output with no ``input_ids``; merging that with the first
        # token's embedding makes the backbone see a multi-token block whose
        # n-gram lookup still covers one position. Engram then broadcasts the
        # empty slice to sequence length 0 and decode refuses it.
        if (
            len(conversation_list) >= 2
            and conversation_list[-2].type == "output"
            and "input_ids" in conversation_list[-2].meta
        ):
            maybe_merge_outputs(conversation_list)
        # The ids, not just their embeddings: deepseek_v41_ngram hashes token ids
        # and this is the only place the sampled ones reach the carrier. A hit
        # writes the rest of the DSpark block too, so one FSM step can verify
        # several drafts instead of launching the graph once per token.
        #
        # After the merge, not before: merging rebuilds the tail, and a meta set
        # on the pre-merge item would be dropped on the step that merges -- the
        # n-gram node would then hash a stale id with nothing raising.
        tail = conversation_list[-1]
        tail.meta["input_ids"] = input_ids
        if spec_rest:
            tail.meta["dspark_spec_ids"] = torch.tensor(spec_rest, dtype=torch.long, device=self.device)
        else:
            tail.meta.pop("dspark_spec_ids", None)
        self._stash_dspark_block(tail, output_token_id)
        outputs["dspark_lm_head_weight"] = (
            self.embed_tokens.weight if self.config.tie_word_embeddings else self.lm_head.weight
        )
        outputs["dspark_sampling"] = sampling

        # V4.1 has one stop token: the turn ends on the end-of-sentence marker that
        # `_render_assistant` appends. There is no second ChatML-style `im_end`.
        if output_token_id == self.config.eos_token_id:
            outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
            outputs["generated"] = self._flush_text_generated(conversation_list)
        return outputs

    def draft(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Dict[str, Any] = dict,
        dspark_hiddens: Optional[torch.Tensor] = None,
        dspark_start_id: Optional[torch.Tensor] = None,
        dspark_markov_embed_weight: Optional[torch.Tensor] = None,
        dspark_markov_head_weight: Optional[torch.Tensor] = None,
        dspark_accepted_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sample the DSpark block with the shared LM head plus the Markov residual.

        Official ``DSparkBlock.forward_head``: for each draft slot, add
        ``markov_head(token_i)`` to ``lm_head(hidden_i)`` and sample token i+1.
        The resulting ids are proposed for later AR steps; ``generate`` only
        keeps a proposal when it matches the token the backbone actually
        sampled, so the output distribution stays the AR one.

        ``dspark_accepted_ids`` are extra drafts the backbone already verified
        this step -- they go on the token cache here because ``generate`` only
        committed the first sampled id before the verify ran.
        """
        del kwargs
        outputs: Dict[str, Any] = {"conversation_list": conversation_list}
        if dspark_accepted_ids is not None:
            for token_id in dspark_accepted_ids.reshape(-1).tolist():
                token_id = int(token_id)
                self._text_token_cache.append(token_id)
                if token_id == self.config.eos_token_id:
                    outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
                    outputs["generated"] = self._flush_text_generated(conversation_list)
                    return outputs
        if (
            dspark_hiddens is None
            or dspark_start_id is None
            or dspark_markov_embed_weight is None
            or dspark_markov_head_weight is None
            or self.config.dspark_block_size <= 0
        ):
            return outputs
        hidden = dspark_hiddens.to(self.device)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        sampling = self._extract_sampling_kwargs(generation_kwargs, 1.0, 1.0, {})
        token_id = int(dspark_start_id.reshape(-1)[0].item())
        drafts: List[int] = []
        for index in range(hidden.shape[1]):
            logits = self._project(hidden[:, index])
            token = torch.tensor([token_id], dtype=torch.long, device=self.device)
            markov_embed = F.embedding(token, dspark_markov_embed_weight)
            logits = logits + F.linear(markov_embed, dspark_markov_head_weight)
            token_id = self._sample_from_logits(logits, **sampling)
            drafts.append(token_id)
            if token_id == self.config.eos_token_id:
                break
        self._dspark_queue = drafts
        return outputs

    def _stash_dspark_block(self, tail: ConversationItem, token_id: int) -> None:
        block_size = self.config.dspark_block_size
        if block_size <= 0:
            return
        block = torch.full(
            (1, block_size),
            self.config.dspark_noise_token_id,
            dtype=torch.long,
            device=self.device,
        )
        block[0, 0] = token_id
        tail.meta["dspark_input_ids"] = block
        tail.meta["dspark_embeds"] = self.encode(block)["inputs_embeds"]
        noise = torch.tensor([[self.config.dspark_noise_token_id]], dtype=torch.long, device=self.device)
        tail.meta["dspark_noise_embed"] = self.encode(noise)["inputs_embeds"]

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        **kwargs,
    ) -> int:
        del kwargs
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if not do_sample:
            return self._agree_on_token(int(logits.argmax(dim=-1).item()))
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        if top_p < 1.0:
            logits = self._top_p_filter(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return self._agree_on_token(int(token.item()))


__all__ = ["DeepseekV41TextEncoder", "SIGNAL_TEXT_DONE"]
