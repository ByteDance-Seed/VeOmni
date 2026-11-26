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

"""VeOmni-side hooks for ``deepseek_v41_mtp``: zero-copy handoff from ``deepseek_v41_llm``,
scatter of ``dspark_hidden`` / ``dspark_markov_bias`` onto ``conversation_list``,
per-module FLOPs meter, and the EP parallel plan over ``model.mtp.*.mlp.experts.*``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed.tensor import Shard

from .......distributed.parallel_plan import ParallelPlan
from .......distributed.parallel_state import get_parallel_state
from .......utils.tensor_utils import unflatten
from .....mixins.base_mixin import BaseMixin
from .....mixins.metric_meter_mixin import MetricMeterMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ....base.sp_shard import split_sp_sharded_tensor
from ...ngram.modeling import pack_ngram_inputs
from ...quant_scheme import quantized_storage_dtype
from ..configuration import DeepseekV41MtpConfig
from ..modeling import DeepseekV41Mtp


class TrainingMixin(TrainingModuleMixin):
    """Zero-copy consumer of ``dspark_inputs`` emitted by ``deepseek_v41_llm``."""

    config: DeepseekV41MtpConfig
    device: torch.device
    training: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._pack_inputs_embeds_shape: torch.Tensor | None = None
        self._sequence_shard: Any = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        dspark_inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        if not self.dspark_trains:
            self._pack_inputs_embeds_shape = None
            self._sequence_shard = None
            return {}
        if not dspark_inputs:
            raise ValueError(
                "deepseek_v41_mtp is trainable in the graph but deepseek_v41_llm did not emit "
                "`dspark_inputs`. Set `emit_draft_hiddens: true` on `deepseek_v41_llm` "
                "(with non-empty `dspark_target_layer_ids` and `dspark_block_size > 0`)."
            )

        self._pack_inputs_embeds_shape = dspark_inputs["pack_shape"]
        self._sequence_shard = dspark_inputs["sequence_shard"]
        sample_seqlens = dspark_inputs.get("sample_seqlens")
        if sample_seqlens:
            self.metric_meter_set_seqlens("forward", sample_seqlens)

        # Image rows never carry a DSpark label, so the id they pack to is arbitrary.
        dspark_input_ids, _, _ = pack_ngram_inputs(conversation_list, self.device, pad_token_id=0)
        self._use_factored_dspark_markov = True
        return {
            "inputs_embeds": dspark_inputs["inputs_embeds"],
            "position_ids": dspark_inputs["position_ids"],
            "draft_hiddens": dspark_inputs["draft_hiddens"],
            "image_mask": dspark_inputs["image_mask"],
            "sequence_shard": dspark_inputs["sequence_shard"],
            "dspark_input_ids": dspark_input_ids,
        }

    @post_forward("forward")
    def forward_post(self, **outputs: Any) -> dict[str, Any]:
        conversation = self._conversation_carrier
        pack_shape = self._pack_inputs_embeds_shape
        shard = self._sequence_shard
        self._conversation_carrier = None
        self._pack_inputs_embeds_shape = None
        self._sequence_shard = None

        dspark_head_weight = outputs.get("dspark_markov_head_weight")
        dspark_rows = {
            meta_key: outputs[key]
            for key, meta_key in (
                ("dspark_hidden_states", "dspark_hidden"),
                ("dspark_markov_bias", "dspark_markov_bias"),
            )
            if outputs.get(key) is not None
        }
        if dspark_rows and pack_shape is not None and conversation is not None:
            ps = get_parallel_state()
            if shard is not None and not ps.skip_sp_gather_between_modules:
                dspark_rows = {
                    key: shard.gather(rows).narrow(1, 0, shard.valid_len) for key, rows in dspark_rows.items()
                }
            use_sharded_scatter = shard is not None and ps.skip_sp_gather_between_modules
            lengths = pack_shape.prod(-1).tolist()
            first_non_dummy = None
            for meta_key, rows in dspark_rows.items():
                if rows.dim() == 3 and rows.size(0) == 1:
                    rows = rows.squeeze(0)
                row_chunks = (
                    split_sp_sharded_tensor(rows, lengths, ps.sp_rank, ps.sp_size)
                    if use_sharded_scatter
                    else unflatten(rows, pack_shape)
                )
                row_iter = iter(row_chunks)
                for sample in conversation:
                    for part in sample:
                        if part.is_dummy:
                            continue
                        if first_non_dummy is None:
                            first_non_dummy = part
                        part.meta[meta_key] = next(row_iter)
                if next(row_iter, None) is not None:
                    raise RuntimeError(f"{meta_key} segment count exceeds non-dummy conversation items.")
            if dspark_head_weight is not None and first_non_dummy is not None:
                first_non_dummy.meta["dspark_markov_head_weight"] = dspark_head_weight

        # Clear ``dspark_inputs`` on the shared batch so references to
        # ``draft_hiddens`` / ``inputs_embeds`` are released before decode.
        return {"conversation_list": conversation, "dspark_inputs": None}


class MeterMixin(MetricMeterMixin):
    """Per-module training FLOPs meter for the DSpark stages."""

    config: DeepseekV41MtpConfig

    def estimate_flops(self, seqlens: list[int]) -> float:
        config = self.config
        n_stages = config.num_nextn_predict_layers if config.dspark_block_size > 0 else 0
        if n_stages <= 0:
            return 0.0
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.head_dim
        n_routed = config.dspark_n_routed_experts or config.n_routed_experts
        n_activated = config.dspark_num_experts_per_tok or config.num_experts_per_tok

        attn_linear = (
            hidden * config.q_lora_rank
            + config.q_lora_rank * heads * head_dim
            + hidden * head_dim
            + heads * head_dim * config.o_lora_rank
            + config.o_groups * config.o_lora_rank * hidden
        )
        moe = (
            n_activated * 3 * hidden * config.moe_intermediate_size
            + config.n_shared_experts * 3 * hidden * config.moe_intermediate_size
            + hidden * n_routed
        )
        mhc = 2 * (2 + config.hc_mult) * config.hc_mult * config.hc_mult * hidden
        main_proj = len(config.dspark_target_layer_ids) * hidden * hidden
        per_token = (attn_linear + moe + mhc) * n_stages + main_proj

        tokens = sum(seqlens)
        dense_flops = 6 * per_token * tokens
        # DSpark attention is window-only (no compressed KV / indexer).
        attn_flops = 12 * tokens * config.sliding_window * head_dim * heads * n_stages
        return (dense_flops + attn_flops) / 1e12


class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin):
    """Family assembly plus the expert-parallel plan over ``model.mtp.*``."""

    def get_parallel_plan(self) -> ParallelPlan:
        ep_plan = {
            "model.mtp.*.mlp.experts.gate_up_proj": Shard(0),
            "model.mtp.*.mlp.experts.down_proj": Shard(0),
        }
        if quantized_storage_dtype(self.config) is not None:
            ep_plan["model.mtp.*.mlp.experts.gate_up_proj_scale"] = Shard(0)
            ep_plan["model.mtp.*.mlp.experts.down_proj_scale"] = Shard(0)
        return ParallelPlan(extra_parallel_plan={"ep": ep_plan})


class DeepseekV41MtpAccelerated(VeOmniMixin, DeepseekV41Mtp):
    """``deepseek_v41_mtp`` with VeOmni training hooks and EP plan mixed in."""


__all__ = ["DeepseekV41MtpAccelerated", "MeterMixin", "TrainingMixin", "VeOmniMixin"]
