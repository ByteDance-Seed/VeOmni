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

"""Training-graph hooks for the DeepSeek-V4.1 text encoder.

Encode / decode plumbing -- the SP slice and gather around ``embed_tokens``,
the fused linear+CE head, the vocab-parallel lookup -- is the shared base's.
V4.1 adds the chat template (``thinking_mode`` / ``reasoning_effort``) and,
when the backbone scattered ``meta['dspark_hidden']``, a Qwen3.5-style MTP CE:
``decode_pre`` extra-shifts the already-causal labels into ``mtp_labels``
(predict t+2), and ``decode`` scores the packed DSpark hidden against those.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from veomni.utils.constants import IGNORE_INDEX

from .....mixins.training_module_mixin import post_forward, pre_forward
from .....utils.conversation import ConversationItem, is_dummy
from ....base.text_encoder.accelerated import TrainingMixin as BaseTrainingMixin
from ....base.text_encoder.accelerated import VeOmniMixin as BaseVeOmniMixin
from ..chat_template import DeepseekV41ChatTemplate
from ..configuration import DeepseekV41TextEncoderConfig
from ..modeling import DeepseekV41TextEncoder


class TrainingMixin(BaseTrainingMixin):
    config: DeepseekV41TextEncoderConfig
    device: torch.device
    _chat_template: DeepseekV41ChatTemplate

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # The shared batch still holds ngram_embeds from the sibling node.
        del kwargs
        return super().encode_pre(conversation_list)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().encode_post(**outputs)

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        packed = super().decode_pre(conversation_list)
        dspark_hidden = self._pack_dspark_hidden(conversation_list)
        if dspark_hidden is not None:
            packed["dspark_hidden_states"] = dspark_hidden
            packed["mtp_labels"] = self._shift_mtp_labels(packed["shift_labels"], conversation_list)
        return packed

    def _shift_mtp_labels(
        self,
        shift_labels: torch.Tensor,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> torch.Tensor:
        """Qwen3.5 MTP: extra-shift already-causal labels so the draft head predicts t+2.

        The base ``decode_pre`` already shifted once (t+1) and masked each later
        sample's first label. Extra-shifting that IGNORE covers the second-to-last
        token of the previous sample, not the last, so the last packed index of
        every sample is masked again -- otherwise it would score the next sample.
        """
        mtp_labels = F.pad(shift_labels[..., 1:].contiguous(), (0, 1), "constant", IGNORE_INDEX)
        offset = 0
        for sample in conversation_list or []:
            sample_len = 0
            for part in sample:
                if is_dummy(part):
                    continue
                if part.type == "text":
                    sample_len += part.meta["labels"].shape[0]
                elif part.type in ("image", "video"):
                    sample_len += 1
            if sample_len:
                mtp_labels[offset + sample_len - 1] = IGNORE_INDEX
                offset += sample_len
        return mtp_labels

    def _pack_dspark_hidden(self, conversation_list: Optional[list[list[ConversationItem]]]) -> Optional[torch.Tensor]:
        """Same walk as ``_prepare_decode_inputs``, reading ``meta['dspark_hidden']``."""
        chunks = []
        for sample in conversation_list or []:
            for part in sample:
                if is_dummy(part):
                    continue
                hidden = part.meta.get("dspark_hidden")
                if hidden is None:
                    return None
                if hidden.dim() == 3:
                    hidden = hidden.squeeze(0)
                if part.type == "text":
                    chunks.append(hidden)
                elif part.type in ("image", "video"):
                    chunks.append(hidden[-1:])
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)

    def decode(
        self,
        hidden_states: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        shift_labels: torch.LongTensor | None = None,
        dspark_hidden_states: torch.Tensor | None = None,
        mtp_labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> dict:
        if dspark_hidden_states is not None and hidden_states is not None:
            if dspark_hidden_states.shape[:-1] != hidden_states.shape[:-1]:
                raise RuntimeError(
                    "DSpark hidden packed to "
                    f"{tuple(dspark_hidden_states.shape[:-1])} but the LM head packed "
                    f"{tuple(hidden_states.shape[:-1])}. Scatter and decode must walk "
                    "the same non-dummy conversation parts."
                )
        outputs = super().decode(hidden_states=hidden_states, labels=labels, shift_labels=shift_labels, **kwargs)
        if dspark_hidden_states is None or outputs.get("loss") is None:
            return outputs
        if mtp_labels is None:
            if shift_labels is None:
                return outputs
            mtp_labels = F.pad(shift_labels[..., 1:].contiguous(), (0, 1), "constant", IGNORE_INDEX)
        draft = super().decode(
            hidden_states=dspark_hidden_states,
            labels=None,
            shift_labels=mtp_labels,
            **kwargs,
        )
        if draft.get("loss") is not None:
            outputs["loss"] = outputs["loss"] + draft["loss"]
        return outputs

    @post_forward("decode")
    def decode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().decode_post(**outputs)


class VeOmniMixin(TrainingMixin, BaseVeOmniMixin):
    _chat_template: DeepseekV41ChatTemplate

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = DeepseekV41ChatTemplate(
            tokenizer,
            thinking_mode=self.config.thinking_mode,
            reasoning_effort=self.config.reasoning_effort,
        )


class DeepseekV41TextEncoderAccelerated(VeOmniMixin, DeepseekV41TextEncoder):
    pass


__all__ = ["DeepseekV41TextEncoderAccelerated"]
