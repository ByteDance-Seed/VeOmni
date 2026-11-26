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

"""Standalone DeepSeek-V4.1 DSpark / MTP OmniModule (``deepseek_v41_mtp``).

Holds the three DSpark draft stages (``model.mtp.{0,1,2}``) as an independent
node in the training graph so that:

* runs that do not train DSpark omit the node completely instead of loading and
  freezing ~14 B of weights inside ``deepseek_v41_llm``;
* ``deepseek_v41_mtp`` carries its own EP size (128 routed experts vs the
  backbone's 384), optimizer / LR, freeze policy, and FSDP2 stack.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_layers import GradientCheckpointingLayer

from ......ops import VeomniOp
from ......ops.kernels.dsa.sparse_mqa_target import sparse_attn_tilelang
from ......utils.device import get_gpu_compute_capability
from ...module_modeling_base import PretrainedOmniModule
from ..llm.modeling import (
    DeepseekV4Experts,
    DeepseekV4RMSNorm,
    DeepseekV41Attention,
    DeepseekV41HyperConnection,
    DeepseekV41RotaryEmbedding,
    DeepseekV41SparseMoeBlock,
    DeepseekV41TopKRouter,
    DSparkConfidenceHead,
    DSparkMarkovHead,
    SequenceShard,
    apply_interleaved_rope,
    dense_sparse_attn,
    dispatch_linear,
    expand_to_streams,
    identity_pre_mix,
    linear_quant_dtype,
    main_window_indices,
    make_linear,
    mhc_collapse,
    mhc_expand,
    require_fp32_pre_mix,
    resolve_module_op_impl,
    round_window_kv,
)
from .configuration import DeepseekV41MtpConfig


class DeepseekV41DSparkBlock(GradientCheckpointingLayer):
    """One DSpark stage: window attention over the backbone's main KV, then MoE."""

    def __init__(self, config: DeepseekV41MtpConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        stage_id = layer_idx - config.num_hidden_layers
        self.stage_id = stage_id
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id

        n_routed, n_activated = config.moe_config_for_layer(layer_idx)
        self.self_attn = DeepseekV41Attention(config, layer_idx)
        if self.self_attn.compress_ratio != 0:
            raise ValueError(
                f"DSpark stage {stage_id} (layer {layer_idx}) has compress_ratio "
                f"{self.self_attn.compress_ratio}; official DSparkAttention is window-only"
            )
        self.mlp = DeepseekV41SparseMoeBlock(
            config, layer_idx=layer_idx, n_routed_experts=n_routed, num_experts_per_tok=n_activated
        )
        self.mlp.gate._uniform_route_image_mask = True
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.attn_hc = DeepseekV41HyperConnection(config)
        self.ffn_hc = DeepseekV41HyperConnection(config)
        self.veomni_mhc_post = self.attn_hc.veomni_mhc_post

        if stage_id == 0:
            n_targets = len(config.dspark_target_layer_ids)
            self.main_proj = make_linear(
                config.hidden_size * n_targets,
                config.hidden_size,
                bias=False,
                quant_dtype=linear_quant_dtype(config),
            )
            self.veomni_main_proj = VeomniOp(
                "linear",
                "standard",
                resolve_module_op_impl(config, "linear_implementation"),
                site=f"layers.{layer_idx}.main_proj",
            )
            self.main_norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        if stage_id == config.num_nextn_predict_layers - 1:
            self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
            self.markov_head = DSparkMarkovHead(config.vocab_size, config.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(config.hidden_size + config.dspark_markov_rank)

    def project_main(self, draft_hiddens: list[torch.Tensor]) -> torch.Tensor:
        """``main_norm(main_proj(cat(target-layer inputs)))`` -- stage 0 only."""
        if self.stage_id != 0:
            raise RuntimeError("only the first DSpark stage owns main_proj")
        return self.main_norm(dispatch_linear(self.veomni_main_proj, self.main_proj, torch.cat(draft_hiddens, dim=-1)))

    def forward(
        self,
        hidden_streams: torch.Tensor,
        pre_mix: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        main: list[torch.Tensor] | torch.Tensor,
        shard: SequenceShard | None = None,
        input_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        factored_markov: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """One teacher-forced packed stage: the only way training enters a stage."""
        require_fp32_pre_mix(pre_mix)
        main_x = self.project_main(main) if self.stage_id == 0 else main
        hidden_streams, pre_mix = self.forward_draft_packed(
            hidden_streams, pre_mix, position_ids, rotary_emb, main_x, shard, image_mask=image_mask
        )
        draft = markov_bias = None
        self._last_markov_head_weight = None
        if hasattr(self, "norm"):
            draft = self.collapse_draft(hidden_streams, pre_mix)
            if input_ids is not None:
                if factored_markov:
                    markov_bias, self._last_markov_head_weight = self.markov_head.encode(input_ids)
                else:
                    markov_bias, _ = self.markov_head(input_ids)
        return hidden_streams, pre_mix, main_x, draft, markov_bias

    def _window_kv(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
    ) -> torch.Tensor:
        attn = self.self_attn
        cos, sin = rotary_emb(position_ids, compressed=False)
        kv = attn.kv_norm(dispatch_linear(attn.veomni_wkv, attn.wkv, hidden_states))
        kv = apply_interleaved_rope(kv, cos, sin)
        return round_window_kv(
            kv,
            training=attn.training,
            op=attn.veomni_kv_round_fp8,
        )

    def forward_draft_packed(
        self,
        hidden_streams: torch.Tensor,
        pre_mix: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        main_x: torch.Tensor,
        shard: SequenceShard | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_streams
        attn_pre, attn_post, attn_comb = self.attn_hc(hidden_streams)
        collapsed = mhc_collapse(hidden_streams, pre_mix)
        attn_out = self._packed_draft_attention(
            self.input_layernorm(collapsed), position_ids, rotary_emb, main_x, shard
        )
        hidden_streams = mhc_expand(attn_out, residual, attn_post, attn_comb, op=self.veomni_mhc_post)

        residual = hidden_streams
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(hidden_streams)
        collapsed = mhc_collapse(hidden_streams, attn_pre)
        mlp_out, _ = self.mlp(self.post_attention_layernorm(collapsed), image_mask=image_mask)
        hidden_streams = mhc_expand(mlp_out, residual, ffn_post, ffn_comb, op=self.veomni_mhc_post)
        return hidden_streams, ffn_pre

    def _packed_draft_attention(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        main_x: torch.Tensor,
        shard: SequenceShard | None = None,
    ) -> torch.Tensor:
        if main_x.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                f"packed DSpark expected main_x shape {tuple(hidden_states.shape[:2])}, got {tuple(main_x.shape[:2])}"
            )
        attn = self.self_attn
        batch, seq_len, _ = hidden_states.shape
        cos, sin = rotary_emb(position_ids, compressed=False)

        q_residual = attn.q_norm(dispatch_linear(attn.veomni_wq_a, attn.wq_a, hidden_states))
        query = dispatch_linear(attn.veomni_wq_b, attn.wq_b, q_residual).unflatten(-1, (attn.n_heads, attn.head_dim))
        query = apply_interleaved_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2))

        kv = self._window_kv(hidden_states, position_ids, rotary_emb)
        main_kv = self._window_kv(main_x, position_ids, rotary_emb)
        if shard is None:
            main_idxs = self._main_window_indices(position_ids)
        else:
            main_kv = shard.gather_kv(main_kv)
            main_idxs = shard.local_rows(self._main_window_indices(shard.position_ids))
        pool = torch.cat([main_kv, kv], dim=1)
        own_idxs = main_kv.shape[1] + torch.arange(seq_len, device=hidden_states.device, dtype=torch.int32)
        topk_idxs = torch.cat([main_idxs, own_idxs.view(1, -1, 1).expand(batch, -1, -1)], dim=-1)

        if query.is_cuda and query.dtype == torch.bfloat16 and get_gpu_compute_capability() >= 90:
            output = sparse_attn_tilelang(
                query.contiguous(),
                pool.contiguous(),
                attn.attn_sink.float().contiguous(),
                topk_idxs.contiguous(),
                attn.softmax_scale,
            )
        else:
            output = dense_sparse_attn(query, pool, attn.attn_sink, topk_idxs, attn.softmax_scale)
        output = apply_interleaved_rope(output, cos.unsqueeze(-2), -sin.unsqueeze(-2))
        grouped = dispatch_linear(attn.veomni_wo_a, attn.wo_a, output.reshape(batch, seq_len, attn.n_groups, -1))
        return dispatch_linear(attn.veomni_wo_b, attn.wo_b, grouped.flatten(2))

    def _main_window_indices(self, position_ids: torch.Tensor) -> torch.Tensor:
        return main_window_indices(position_ids, self.self_attn.sliding_window)

    def collapse_draft(self, hidden_streams: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "norm"):
            raise RuntimeError("only the last DSpark stage owns the draft norm")
        return self.norm(mhc_collapse(hidden_streams, pre_mix))


def build_dspark_blocks(config: DeepseekV41MtpConfig) -> nn.ModuleList:
    """``num_nextn_predict_layers`` stages, or empty when the config has none."""
    n_stages = config.num_nextn_predict_layers
    if n_stages <= 0 or config.dspark_block_size <= 0:
        return nn.ModuleList()
    return nn.ModuleList(DeepseekV41DSparkBlock(config, config.num_hidden_layers + stage) for stage in range(n_stages))


class DeepseekV41MtpBackbone(nn.Module):
    """The DSpark stage stack (``model.mtp.*``) plus its rotary frequency buffer."""

    def __init__(self, config: DeepseekV41MtpConfig) -> None:
        super().__init__()
        self.config = config
        self.mtp = build_dspark_blocks(config)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)

    def project_main(self, draft_hiddens: list[torch.Tensor]) -> torch.Tensor:
        if not self.mtp:
            raise RuntimeError("project_main was called but this module has no DSpark stages")
        return self.mtp[0].project_main(draft_hiddens)

    def forward_train(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        draft_hiddens: list[torch.Tensor],
        shard: SequenceShard | None = None,
        input_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        factored_markov: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Teacher-forced DSpark on the packed row."""
        if not self.mtp:
            raise RuntimeError("forward_train was called but this module has no DSpark stages")
        if len(draft_hiddens) != len(self.config.dspark_target_layer_ids):
            raise ValueError(
                f"DSpark expected {len(self.config.dspark_target_layer_ids)} target-layer hiddens, "
                f"got {len(draft_hiddens)}"
            )
        hidden_streams = expand_to_streams(inputs_embeds, self.config.hc_mult)
        pre_mix = identity_pre_mix(hidden_streams, self.config.hc_mult)
        if input_ids is not None and shard is not None:
            input_ids = shard.local_rows(F.pad(input_ids, (0, shard.full_len - input_ids.shape[1])))
        main: Any = draft_hiddens
        draft = markov_bias = None
        last = len(self.mtp) - 1
        for stage_id, stage in enumerate(self.mtp):
            hidden_streams, pre_mix, main, draft, markov_bias = stage(
                hidden_streams,
                pre_mix,
                position_ids,
                self.rotary_emb,
                main,
                shard,
                input_ids if stage_id == last else None,
                image_mask,
                factored_markov if stage_id == last else False,
            )
        return draft, markov_bias


class DeepseekV41Mtp(PretrainedOmniModule):
    """``deepseek_v41_mtp`` -- the DSpark draft stages as a standalone OmniModule."""

    config_class = DeepseekV41MtpConfig
    supports_sp_sharded_handoff = True
    base_model_prefix = "deepseek_v41_mtp"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["DeepseekV41DSparkBlock"]
    supports_gradient_checkpointing = True

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        return attn_implementation

    @staticmethod
    def _create_checkpoint_tensor_converter(model: DeepseekV41Mtp):
        from ..checkpoint_tensor_converter import create_deepseek_v41_mtp_checkpoint_tensor_converter

        return create_deepseek_v41_mtp_checkpoint_tensor_converter(model)

    def __init__(self, config: DeepseekV41MtpConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = DeepseekV41MtpBackbone(config)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if getattr(module, "is_quantized", False):
            return
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, DeepseekV41HyperConnection):
            nn.init.normal_(module.fn, mean=0.0, std=std)
            nn.init.zeros_(module.base)
            nn.init.ones_(module.scale)
        elif isinstance(module, DeepseekV41TopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Experts) and not getattr(module, "experts_are_quantized", False):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(module.down_proj, mean=0.0, std=std)

    def freeze_model(self) -> None:
        if self.config.freeze:
            self.model.mtp.requires_grad_(False)

    @property
    def dspark_trains(self) -> bool:
        return self.training and bool(self.model.mtp) and any(p.requires_grad for p in self.model.mtp.parameters())

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        draft_hiddens: list[torch.Tensor] | None = None,
        image_mask: torch.Tensor | None = None,
        sequence_shard: SequenceShard | None = None,
        dspark_input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if not self.dspark_trains or not draft_hiddens:
            return {}
        if inputs_embeds is None:
            raise ValueError("deepseek_v41_mtp requires inputs_embeds from deepseek_v41_llm.")
        if position_ids is None:
            raise ValueError("deepseek_v41_mtp requires position_ids from deepseek_v41_llm.")
        factored_markov = bool(getattr(self, "_use_factored_dspark_markov", False))
        draft, markov_bias = self.model.forward_train(
            inputs_embeds,
            position_ids,
            draft_hiddens,
            sequence_shard,
            dspark_input_ids,
            image_mask=image_mask,
            factored_markov=factored_markov,
        )
        outputs: dict[str, Any] = {"dspark_hidden_states": draft}
        if markov_bias is not None:
            outputs["dspark_markov_bias"] = markov_bias
            markov_hw = getattr(self.model.mtp[-1], "_last_markov_head_weight", None)
            if markov_hw is not None:
                outputs["dspark_markov_head_weight"] = markov_hw
        return outputs


__all__ = [
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "DeepseekV41DSparkBlock",
    "DeepseekV41Mtp",
    "DeepseekV41MtpBackbone",
    "build_dspark_blocks",
]
