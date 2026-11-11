# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""VeOmni-accelerated ``LladaUniTextEncoder`` — dLLM masking and sample-mean loss reduction."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.loss_utils.dllm import dllm_masked_cross_entropy
from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.loss_utils import dllm_mean_global_loss
from veomni.utils.tensor_utils import naflatten

from .....mixins.training_module_mixin import post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ....text_encoder.accelerated import (
    TrainingMixin as BaseTrainingMixin,
)
from ....text_encoder.accelerated import (
    VeOmniMixin as BaseVeOmniMixin,
)
from ..chat_template import (
    _OMNI_TOKENIZED,
    LladaUniChatMarkers,
    prepare_llada2_uni_sample_training,
)
from ..configuration import LladaUniTextEncoderConfig
from ..modeling import LladaUniTextEncoder


class TrainingMixin(BaseTrainingMixin):
    """Training-graph hooks for ``LladaUniTextEncoder`` (dLLM masking + sample-mean CE)."""

    config: LladaUniTextEncoderConfig
    device: torch.device
    dtype: torch.dtype
    _tokenizer: Any
    _chat_markers: LladaUniChatMarkers

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._chat_markers = LladaUniChatMarkers()

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        return super().encode_pre(conversation_list)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().encode_post(**outputs)

    def _prepare_encode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> torch.Tensor:
        input_ids: list[torch.Tensor] = []
        self._encode_batch_shape = None
        for sample in conversation_list or []:
            if not (sample and sample[0].meta.get(_OMNI_TOKENIZED)):
                prepare_llada2_uni_sample_training(
                    sample,
                    tokenizer=self._tokenizer,
                    config=self.config,
                    markers=self._chat_markers,
                    device=self.device,
                )
            input_ids.extend(part.value for part in sample if part.type == "text" and not part.is_dummy)
        flat_ids, self._encode_batch_shape = naflatten(input_ids)
        return flat_ids.to(self.device, non_blocking=True)

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        hidden_states, labels, loss_weight = self._prepare_dllm_decode_inputs(conversation_list)
        num_samples = len(conversation_list) if conversation_list else 0
        return {
            "hidden_states": hidden_states,
            "labels": labels,
            "loss_weight": loss_weight,
            "num_samples": num_samples,
        }

    @post_forward("decode")
    def decode_post(self, loss: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
        return super().decode_post(loss=loss, logits=logits)

    def _prepare_dllm_decode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        weight_chunks: list[torch.Tensor] = []

        for sample in conversation_list or []:
            for part in sample:
                if part.is_dummy or part.type != "text":
                    continue
                hidden_states = part.value
                if hidden_states.dim() == 3:
                    hidden_states = hidden_states.squeeze(0)
                labels = part.meta["labels"]
                loss_weight = part.meta["loss_weight"]
                assert labels.shape[0] == hidden_states.shape[0]
                hidden_chunks.append(hidden_states)
                label_chunks.append(labels)
                weight_chunks.append(loss_weight)

        hidden_states = torch.cat(hidden_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0).to(device=hidden_states.device, non_blocking=True)
        loss_weight = torch.cat(weight_chunks, dim=0).to(device=hidden_states.device, non_blocking=True)
        return hidden_states, labels, loss_weight

    def decode(
        self,
        hidden_states: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        loss_weight: torch.Tensor | None = None,
        num_samples: int = 1,
        **kwargs: Any,
    ) -> dict:
        """Distributed dLLM masked-CE loss normalized by the global sample count."""
        del kwargs
        loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None

        if labels is not None and loss_weight is not None:
            selected = labels != IGNORE_INDEX
            if bool(selected.any()):
                sel_logits = self._project(hidden_states[selected])
                local_weighted_sum = dllm_masked_cross_entropy(
                    sel_logits,
                    labels[selected],
                    loss_weight[selected],
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
            else:
                local_weighted_sum = self._project(hidden_states[:1]).sum() * 0.0
            # In SeedOmni SP, `llada2_uni_llm.forward_post` already gathered the full
            # sequence onto every rank in the SP group, so `local_weighted_sum` is
            # already the whole SP group's sum (`sum_spans_sp_group=True`).
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                loss = dllm_mean_global_loss(
                    local_weighted_sum,
                    num_samples,
                    sum_spans_sp_group=get_parallel_state().sp_enabled,
                )
            else:
                loss = local_weighted_sum / max(int(num_samples), 1)
        else:
            logits = self._project(hidden_states)

        return {"loss": loss, "logits": logits}


class VeOmniMixin(TrainingMixin, BaseVeOmniMixin):
    config: LladaUniTextEncoderConfig

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._eos_token_id = (
            int(tokenizer.eos_token_id) if getattr(tokenizer, "eos_token_id", None) is not None else None
        )


class LladaUniTextEncoderAccelerated(VeOmniMixin, LladaUniTextEncoder):
    pass


__all__ = ["LladaUniTextEncoderAccelerated", "TrainingMixin", "VeOmniMixin"]
