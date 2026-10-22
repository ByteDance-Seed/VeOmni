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

"""VeOmni-accelerated ``LladaUniLlm`` — training-graph hooks, Ulysses SP, EP plan, and FLOPs meter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.distributed._tensor import Shard

from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids, valid_seqlens_from_cu_seqlens
from veomni.utils.tensor_utils import unflatten

from .....mixins.base_mixin import BaseMixin
from .....mixins.metric_meter_mixin import MetricMeterMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ....base.llm_packing import (
    fold_fsdp_dummy_anchors,
    pack_llm_conversations_for_forward,
    scatter_llm_hidden_states,
)
from ..configuration import LladaUniLlmConfig
from ..modeling import LladaUniLlm


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks for :class:`LladaUniLlm` (Ulysses SP + packed segments)."""

    config: LladaUniLlmConfig
    device: torch.device
    training: bool

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Optional[list[list[ConversationItem]]] = None
        self._pack_inputs_embeds_shape: Optional[torch.Tensor] = None
        self._sp_own_len: Optional[int] = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        inputs_embeds, attention_mask, position_ids, inputs_embeds_shape = pack_llm_conversations_for_forward(
            conversation_list, self.device
        )

        if self.training and get_parallel_state().fsdp_enabled:
            inputs_embeds = fold_fsdp_dummy_anchors(inputs_embeds, conversation_list)

        self._conversation_carrier = conversation_list
        self._pack_inputs_embeds_shape = inputs_embeds_shape

        (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
            position_ids
        )
        self.metric_meter_set_seqlens(
            "forward", [int(s) for s in valid_seqlens_from_cu_seqlens(cu_seq_lens_q).tolist()]
        )

        if get_parallel_state().sp_size > 1:
            group = get_parallel_state().sp_group
            self._sp_own_len = inputs_embeds.size(1)
            embeds = sp_pad(inputs_embeds, dim=1, pad_value=0)
            mask = sp_pad(attention_mask, dim=1, pad_value=1)
            pids = sp_pad(position_ids, dim=1, pad_value=0)
            (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(pids)
            embeds = slice_input_tensor(embeds, dim=1, padding=False, group=group)
            pids = slice_input_tensor(pids, dim=1, padding=False, group=group)
            return dict(
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=pids,
                cu_seq_lens_q=cu_q,
                cu_seq_lens_k=cu_k,
                max_length_q=max_q,
                max_length_k=max_k,
            )

        return dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            **kwargs,
        )

    @post_forward("forward")
    def forward_post(self, **outputs: Any) -> Dict[str, Any]:
        hidden_states = outputs.get("hidden_states")

        if get_parallel_state().sp_size > 1:
            hidden_states = gather_outputs(hidden_states, gather_dim=1, group=get_parallel_state().sp_group)
            hidden_states = hidden_states.narrow(1, 0, self._sp_own_len)

        conversation = self._conversation_carrier
        pack_shape = self._pack_inputs_embeds_shape
        self._conversation_carrier = None
        self._pack_inputs_embeds_shape = None

        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
        scatter_llm_hidden_states(conversation, unflatten(hidden_states, pack_shape))
        return {"conversation_list": conversation}


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for the LLaDA2.0-Uni backbone (Bailing/DeepSeek-MoE FLOPs)."""

    config: LladaUniLlmConfig

    def estimate_flops(self, seqlens: List[int]) -> float:
        cfg = self.config.text_config
        hidden = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", hidden // num_heads)

        first_k_dense = getattr(cfg, "first_k_dense_replace", 0) or 0
        num_dense_layers = min(first_k_dense, num_layers)
        num_moe_layers = max(num_layers - num_dense_layers, 0)

        dense_mlp_n = hidden * cfg.intermediate_size * 3
        active_experts = cfg.num_experts_per_tok + (getattr(cfg, "num_shared_experts", 0) or 0)
        moe_mlp_n = hidden * cfg.moe_intermediate_size * 3 * active_experts + hidden * cfg.num_experts
        attn_linear_n = hidden * (num_heads * head_dim + 2 * num_kv_heads * head_dim) + (num_heads * head_dim) * hidden

        per_layer_dense = dense_mlp_n + attn_linear_n
        per_layer_moe = moe_mlp_n + attn_linear_n
        dense_n = per_layer_dense * num_dense_layers + per_layer_moe * num_moe_layers

        tokens = sum(seqlens)
        seqlen_sq = sum(s * s for s in seqlens)
        dense_flops = 6 * dense_n * tokens
        attn_flops = 12 * seqlen_sq * head_dim * num_heads * num_layers
        return (dense_flops + attn_flops) / 1e12


class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin):
    """LLaDA2.0-Uni backbone hooks + Expert-Parallel plan."""

    config: LladaUniLlmConfig

    def get_parallel_plan(self) -> ParallelPlan:
        ep_plan = {
            "language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
            "language_model.layers.*.mlp.experts.down_proj": Shard(0),
        }
        return ParallelPlan(extra_parallel_plan={"ep": ep_plan})


class LladaUniLlmAccelerated(VeOmniMixin, LladaUniLlm):
    pass


__all__ = ["LladaUniLlmAccelerated", "MeterMixin", "TrainingMixin", "VeOmniMixin"]
