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

"""VeOmni-accelerated TextEncoder — training graph hooks."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_outputs,
    slice_input_tensor,
    sp_pad,
)
from veomni.models.loss_utils import ForCausalLMLoss
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.tensor_utils import naflatten, unflatten

from .....mixins.base_mixin import BaseMixin
from .....mixins.metric_meter_mixin import MetricMeterMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ..chat_template import TextEncoderChatTemplate
from ..configuration import TextEncoderConfig
from ..modeling import TextEncoder, scatter_text_encoder_embeds


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks shared by every text encoder."""

    config: TextEncoderConfig
    device: torch.device
    _chat_template: TextEncoderChatTemplate
    _encode_batch_shape: torch.LongTensor | None

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        input_ids = self._prepare_encode_inputs(self._conversation_carrier)

        self.metric_meter_set_seqlens("encode", [int(input_ids.numel())])

        if get_parallel_state().sp_size > 1:
            self._sp_own_len = input_ids.size(0)
            input_ids = sp_pad(input_ids, dim=0, pad_value=0)
            input_ids = slice_input_tensor(input_ids, dim=0, padding=False, group=get_parallel_state().sp_group)
        return {"input_ids": input_ids}

    @post_forward("encode")
    def encode_post(self, inputs_embeds: torch.Tensor) -> Dict[str, Any]:
        if get_parallel_state().sp_size > 1:
            inputs_embeds = gather_outputs(inputs_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            inputs_embeds = inputs_embeds.narrow(0, 0, self._sp_own_len)
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        batch_shape = self._encode_batch_shape
        self._encode_batch_shape = None
        scatter_text_encoder_embeds(conversation, unflatten(inputs_embeds, batch_shape))
        return {"conversation_list": conversation}

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        hidden_states, shift_labels = self._prepare_decode_inputs(self._conversation_carrier)
        return {"hidden_states": hidden_states, "shift_labels": shift_labels}

    @post_forward("decode")
    def decode_post(self, loss: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
        del logits
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        if loss is not None:
            return {"_loss": loss, "conversation_list": conversation}
        return {"conversation_list": conversation}

    def _prepare_encode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> torch.Tensor:
        input_ids: list[torch.Tensor] = []
        self._encode_batch_shape = None
        for sample in conversation_list or []:
            input_ids.extend(self._chat_template.pack_input_ids(sample))
        input_ids, self._encode_batch_shape = naflatten(input_ids)
        input_ids = input_ids.to(self.device, non_blocking=True)
        return input_ids

    def _prepare_decode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        sample_starts: list[int] = []
        decode_len = 0

        for sample in conversation_list:
            sample_start = decode_len
            for part in sample:
                if part.is_dummy:
                    continue
                hidden_states = part.value
                if hidden_states.dim() == 3:
                    hidden_states = hidden_states.squeeze(0)
                if part.type == "text":
                    labels = part.meta["labels"]
                    assert labels.shape[0] == hidden_states.shape[0]
                    hidden_states_chunks.append(hidden_states)
                    label_chunks.append(labels)
                    decode_len += labels.shape[0]
                elif part.type in ("image", "video"):
                    hidden_states_chunks.append(hidden_states[-1:])
                    label_chunks.append(torch.full((1,), -100, dtype=torch.long))
                    decode_len += 1
            if decode_len > sample_start:
                sample_starts.append(sample_start)

        hidden_states = torch.cat(hidden_states_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)

        for start in sample_starts[1:]:
            labels[start] = IGNORE_INDEX

        labels = labels[..., 1:].contiguous()
        shift_labels = F.pad(labels, (0, 1), "constant", -100)
        shift_labels = shift_labels.to(device=hidden_states.device, non_blocking=True)
        return hidden_states, shift_labels

    @staticmethod
    def _drop_unsupervised_decode_rows(
        hidden_states: torch.Tensor,
        labels: torch.LongTensor | None,
        shift_labels: torch.LongTensor | None,
        target_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.LongTensor | None, torch.LongTensor | None, torch.Tensor]:
        supervised = target_labels != IGNORE_INDEX
        if bool(supervised.all()):
            return hidden_states, labels, shift_labels, target_labels
        hidden_states = hidden_states[supervised]
        target_labels = target_labels[supervised]
        if labels is not None:
            labels = labels[supervised]
        if shift_labels is not None:
            shift_labels = shift_labels[supervised]
        return hidden_states, labels, shift_labels, target_labels

    def decode(
        self,
        hidden_states: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        shift_labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> dict:
        del kwargs
        loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None
        target_labels = shift_labels if shift_labels is not None else labels
        fsdp_group = get_parallel_state().fsdp_group
        num_supervised_tokens = None if target_labels is None else (target_labels != IGNORE_INDEX).sum()
        eager_ce_op = VeomniOp("cross_entropy_loss", "standard", "eager")

        if target_labels is None:
            logits = self._project(hidden_states)
        elif num_supervised_tokens > 0:
            hidden_states, labels, shift_labels, target_labels = self._drop_unsupervised_decode_rows(
                hidden_states, labels, shift_labels, target_labels
            )
            if self.lm_head is not None and self.lm_head.bias is None:
                loss, logits, _ = ForCausalLMLoss(
                    logits=None,
                    labels=target_labels,
                    shift_labels=shift_labels,
                    vocab_size=self.config.vocab_size,
                    hidden_states=hidden_states,
                    weights=self.lm_head.weight,
                    loss_reduction_group=fsdp_group,
                    op=self.veomni_ce_loss,
                )
            else:
                logits = self._project(hidden_states)
                loss, _, _ = ForCausalLMLoss(
                    logits=logits,
                    labels=target_labels,
                    shift_labels=shift_labels,
                    vocab_size=self.config.vocab_size,
                    num_items_in_batch=num_supervised_tokens.clamp(min=1),
                    loss_reduction_group=fsdp_group,
                    op=eager_ce_op,
                )
        else:
            logits = self._project(hidden_states)
            loss, _, _ = ForCausalLMLoss(
                logits=logits,
                labels=target_labels,
                shift_labels=shift_labels,
                vocab_size=self.config.vocab_size,
                num_items_in_batch=num_supervised_tokens.clamp(min=1),
                loss_reduction_group=fsdp_group,
                op=eager_ce_op,
            )

        return {
            "loss": loss,
            "logits": logits,
        }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        self._encode_batch_shape: torch.LongTensor | None = None
        self._sp_own_len: Optional[int] = None
        self.veomni_ce_loss = VeomniOp(
            "cross_entropy_loss",
            "standard",
            resolve_op_impl("cross_entropy_loss_implementation"),
        )


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for the text encoder (wte + lm_head)."""

    config: TextEncoderConfig

    def estimate_flops(self, seqlens: List[int]) -> float:
        lm_head_n = self.config.vocab_size * self.config.hidden_size
        return 6 * lm_head_n * sum(seqlens) / 1e12


class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin):
    """Shared training / inference plumbing for every text encoder."""

    _chat_template: TextEncoderChatTemplate

    def get_parallel_plan(self):
        from ..parallel_plan import get_parallel_plan as _get_parallel_plan

        return _get_parallel_plan()

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = TextEncoderChatTemplate(tokenizer)


class TextEncoderAccelerated(VeOmniMixin, TextEncoder):
    """Training/runtime text encoder — SP-aware hooks + fused-CE ``decode``."""


__all__ = ["TextEncoderAccelerated", "TrainingMixin", "VeOmniMixin", "scatter_text_encoder_embeds"]
