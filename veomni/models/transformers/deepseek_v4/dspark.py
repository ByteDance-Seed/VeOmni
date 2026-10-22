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

"""DSpark MTP draft stages stored under ``mtp.*`` in DeepSeek-V4 checkpoints.

Each stage runs window-only attention over the backbone's projected target-layer
hidden states plus its own draft KV, followed by a routed + shared MoE block.
Stage 0 owns ``main_proj`` and ``main_norm``; the last stage owns ``hc_head``,
``norm``, ``markov_head``, and ``confidence_head``.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers.modeling_layers import GradientCheckpointingLayer

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl, resolve_qat_impl
from veomni.ops.kernels.dsa.sparse_mqa_target import sparse_attn_tilelang
from veomni.ops.qat import fp8_fake_quant_act_prefix, qat_linear
from veomni.utils.device import IS_NPU_AVAILABLE


if TYPE_CHECKING:
    from .configuration_deepseek_v4 import DeepseekV4Config


def dense_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sparse MQA by gather-and-softmax, matching ``sparse_attn_tilelang``.

    Args:
        q: ``[B, S, H, D]``.
        kv: ``[B, S_kv, D]`` -- single MQA KV head shared across all ``H`` query heads.
        attn_sink: ``[H]`` fp32 additive logit per head that takes softmax mass
            without contributing a value.
        topk_idxs: ``[B, S, topk]`` int32 indices into ``kv``'s row axis; negative
            or out-of-range entries are masked.
        sm_scale: softmax scale, defaulting to ``D ** -0.5``.
        return_lse: also return the ``[B, S, H]`` fp32 base-2 log-sum-exp, sink included.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    batch, kv_len = kv.shape[:2]
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    safe = topk_idxs.clamp(0, max(kv_len - 1, 0)).long()

    rows = safe + torch.arange(batch, device=kv.device).view(-1, 1, 1) * kv_len
    gathered = kv.reshape(batch * kv_len, -1).index_select(0, rows.flatten()).view(*rows.shape, kv.shape[-1])

    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * sm_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    sink = attn_sink.float().view(1, 1, -1)
    shift = torch.maximum(scores.max(dim=-1).values, sink)
    exp_scores = (scores - shift.unsqueeze(-1)).exp()
    numerator = torch.einsum("bmhk,bmkd->bmhd", exp_scores, gathered.float())
    denominator = exp_scores.sum(-1) + (sink - shift).exp()

    out = (numerator / denominator.unsqueeze(-1)).to(q.dtype)
    if not return_lse:
        return out
    return out, ((shift + denominator.log()) / math.log(2.0)).float()


def main_window_indices(position_ids: torch.Tensor, sliding_window: int) -> torch.Tensor:
    """``[B, S, min(sliding_window, S)]`` int32: main rows ``q-width .. q-1`` of the query's own sample.

    ``offset > position_ids[q]`` reaches before the sample start, so packed
    ``position_ids`` restarting at zero doubles as the sample-boundary mask.
    """
    seq_len = position_ids.shape[1]
    device = position_ids.device
    width = min(sliding_window, seq_len)
    queries = torch.arange(seq_len, device=device, dtype=torch.int32)
    offsets = torch.arange(width, 0, -1, device=device, dtype=torch.int32)
    indices = (queries[None, :, None] - offsets[None, None, :]).expand(position_ids.shape[0], -1, -1)
    return indices.masked_fill(offsets[None, None, :] > position_ids.to(torch.int32)[:, :, None], -1)


class DSparkMarkovHead(nn.Module):
    """Low-rank residual over the shared LM head, stored on the last MTP stage."""

    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank)
        self.head = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed(token_ids)
        return self.head(embed), embed


class DSparkConfidenceHead(nn.Module):
    """Scalar confidence from the draft hidden concatenated with the Markov embed."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([hidden, markov_embed], dim=-1).float()).squeeze(-1)


def _resolve_default_dsv4_classes() -> tuple[type[nn.Module], ...]:
    if IS_NPU_AVAILABLE:
        from .generated import patched_modeling_deepseek_v4_npu as gen
    else:
        from .generated import patched_modeling_deepseek_v4_gpu as gen
    return (
        gen.DeepseekV4Attention,
        gen.DeepseekV4SparseMoeBlock,
        gen.DeepseekV4RMSNorm,
        gen.DeepseekV4HyperConnection,
        gen.DeepseekV4HyperHead,
    )


def _build_stage_config(config: DeepseekV4Config, layer_idx: int) -> DeepseekV4Config:
    stage_config = copy.copy(config)
    moe_config_fn = getattr(config, "moe_config_for_layer", None)
    if callable(moe_config_fn):
        n_routed, n_activated = moe_config_fn(layer_idx)
    else:
        n_routed = getattr(config, "dspark_n_routed_experts", None) or config.n_routed_experts
        n_activated = getattr(config, "dspark_num_experts_per_tok", None) or config.num_experts_per_tok
    stage_config.n_routed_experts = n_routed
    stage_config.num_experts_per_tok = n_activated
    n_total = max(len(config.layer_types), layer_idx + 1)
    stage_config.layer_types = list(config.layer_types) + ["sliding_attention"] * (n_total - len(config.layer_types))
    stage_config.mlp_layer_types = list(config.mlp_layer_types) + ["moe"] * (n_total - len(config.mlp_layer_types))
    return stage_config


class DeepseekV4DSparkBlock(GradientCheckpointingLayer):
    """One DSpark stage: window attention over the backbone's main KV, then MoE."""

    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        *,
        attention_cls: type[nn.Module] | None = None,
        moe_cls: type[nn.Module] | None = None,
        rms_norm_cls: type[nn.Module] | None = None,
        hyper_connection_cls: type[nn.Module] | None = None,
        hyper_head_cls: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if (
            attention_cls is None
            or moe_cls is None
            or rms_norm_cls is None
            or hyper_connection_cls is None
            or hyper_head_cls is None
        ):
            default_attn, default_moe, default_rms, default_hc, default_hc_head = _resolve_default_dsv4_classes()
            attention_cls = attention_cls or default_attn
            moe_cls = moe_cls or default_moe
            rms_norm_cls = rms_norm_cls or default_rms
            hyper_connection_cls = hyper_connection_cls or default_hc
            hyper_head_cls = hyper_head_cls or default_hc_head

        self.layer_idx = layer_idx
        self.config = config
        stage_id = layer_idx - config.num_hidden_layers
        self.stage_id = stage_id
        self.block_size = getattr(config, "dspark_block_size", 0)
        self.noise_token_id = getattr(config, "dspark_noise_token_id", 128799)

        stage_config = _build_stage_config(config, layer_idx)
        self.self_attn = attention_cls(stage_config, layer_idx)
        self.mlp = moe_cls(stage_config, layer_idx)
        self.input_layernorm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = hyper_connection_cls(config)
        self.ffn_hc = hyper_connection_cls(config)
        self.veomni_mhc_post = VeomniOp("mhc", "post", resolve_op_impl("mhc_implementation"))
        self.qat_implementation = resolve_qat_impl()

        if stage_id == 0:
            target_ids = getattr(config, "dspark_target_layer_ids", None)
            if not target_ids:
                target_ids = list(range(max(0, config.num_hidden_layers - 3), config.num_hidden_layers))
            self.main_proj = nn.Linear(
                config.hidden_size * len(target_ids),
                config.hidden_size,
                bias=False,
            )
            self.main_norm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        if stage_id == config.num_nextn_predict_layers - 1:
            self.hc_head = hyper_head_cls(config)
            self.norm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
            markov_rank = getattr(config, "dspark_markov_rank", 256)
            self.markov_head = DSparkMarkovHead(config.vocab_size, markov_rank)
            self.confidence_head = DSparkConfidenceHead(config.hidden_size + markov_rank)

    def project_main(self, draft_hiddens: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """``main_norm(main_proj(cat(target-layer hiddens)))`` -- stage 0 only."""
        if self.stage_id != 0:
            raise RuntimeError("only the first DSpark stage owns main_proj")
        concatenated = torch.cat(draft_hiddens, dim=-1) if isinstance(draft_hiddens, list) else draft_hiddens
        projected = qat_linear(self.main_proj, concatenated, enabled=self.qat_implementation == "fp8_blockwise")
        return self.main_norm(projected)

    def _window_kv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Rotated window KV ``[B, S, head_dim]`` for ``hidden_states``."""
        attn = self.self_attn
        qat_enabled = attn.qat_implementation == "fp8_blockwise"
        hidden_shape = (*hidden_states.shape[:-1], 1, attn.head_dim)
        kv = (
            attn.kv_norm(qat_linear(attn.kv_proj, hidden_states, enabled=qat_enabled))
            .view(*hidden_shape)
            .transpose(1, 2)
        )
        kv = attn.veomni_rope(kv, cos, sin)
        if qat_enabled and kv.numel() > 0:
            kv = fp8_fake_quant_act_prefix(kv, kv.shape[-1] - attn.config.qk_rope_head_dim, block_size=64)
        return kv.squeeze(1)

    def _packed_draft_attention(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        main_x: torch.Tensor,
        full_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Slot-0 ``DSparkAttention`` at every packed position.

        Teacher forcing the draft slot at every row ``j`` reads main rows
        ``j-sliding_window .. j-1`` of the same sample and its own draft key at
        row ``j``. Under sequence/context parallelism, ``main_kv`` is gathered
        across the SP group so a window may reach into the previous rank.
        """
        if main_x.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                f"packed DSpark expected main_x shape {tuple(hidden_states.shape[:2])}, got {tuple(main_x.shape[:2])}"
            )
        attn = self.self_attn
        qat_enabled = attn.qat_implementation == "fp8_blockwise"
        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, attn.head_dim)

        q_residual = attn.q_a_norm(qat_linear(attn.q_a_proj, hidden_states, enabled=qat_enabled))
        q = attn.q_b_norm(qat_linear(attn.q_b_proj, q_residual, enabled=qat_enabled).view(*hidden_shape)).transpose(
            1, 2
        )
        q = attn.veomni_rope(q, cos, sin)
        query = q.transpose(1, 2)

        kv = self._window_kv(hidden_states, cos, sin)
        main_kv = self._window_kv(main_x, cos, sin)

        parallel_state = get_parallel_state()
        if parallel_state.cp_enabled:
            sp_group, sp_size, sp_rank = parallel_state.cp_group, parallel_state.cp_size, parallel_state.cp_rank
        elif parallel_state.ulysses_enabled:
            sp_group, sp_size, sp_rank = (
                parallel_state.ulysses_group,
                parallel_state.ulysses_size,
                parallel_state.ulysses_rank,
            )
        else:
            sp_group, sp_size, sp_rank = None, 1, 0

        if sp_size > 1:
            if full_position_ids is None:
                full_position_ids = gather_outputs(position_ids, gather_dim=-1, group=sp_group)
            main_kv = gather_outputs(main_kv, gather_dim=1, group=sp_group)
            full_main_idxs = main_window_indices(full_position_ids, attn.sliding_window)
            main_idxs = full_main_idxs.narrow(1, sp_rank * seq_len, seq_len)
        else:
            main_idxs = main_window_indices(position_ids, attn.sliding_window)

        pool = torch.cat([main_kv, kv], dim=1)
        own_idxs = main_kv.shape[1] + torch.arange(seq_len, device=hidden_states.device, dtype=torch.int32)
        topk_idxs = torch.cat([main_idxs, own_idxs.view(1, -1, 1).expand(batch, -1, -1)], dim=-1)

        use_tilelang = (
            query.is_cuda
            and query.dtype == torch.bfloat16
            and getattr(attn.veomni_dsa_attention, "impl", None) == "tilelang"
        )
        if use_tilelang:
            output = sparse_attn_tilelang(
                query.contiguous(),
                pool.contiguous(),
                attn.sinks.float().contiguous(),
                topk_idxs.contiguous(),
                attn.scaling,
            )
        else:
            output = dense_sparse_attn(query, pool, attn.sinks, topk_idxs, attn.scaling)

        output = attn.veomni_rope(output.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = output.reshape(batch, seq_len, attn.config.o_groups, -1)
        grouped = qat_linear(attn.o_a_proj, grouped, enabled=qat_enabled).flatten(2)
        return qat_linear(attn.o_b_proj, grouped, enabled=qat_enabled)

    def collapse_draft(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        """Last-stage ``norm(hc_head(hidden_streams))`` -- the tensor the shared LM head reads."""
        if not hasattr(self, "norm"):
            raise RuntimeError("only the last DSpark stage owns the draft norm and hc_head")
        return self.norm(self.hc_head(hidden_streams))

    def forward(
        self,
        hidden_streams: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        main: list[torch.Tensor] | torch.Tensor,
        full_position_ids: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """One teacher-forced packed DSpark stage."""
        main_x = self.project_main(main) if self.stage_id == 0 else main
        cos, sin = position_embeddings["main"] if isinstance(position_embeddings, dict) else position_embeddings

        post, comb, collapsed = self.attn_hc(hidden_streams)
        attn_out = self._packed_draft_attention(
            self.input_layernorm(collapsed),
            cos,
            sin,
            position_ids,
            main_x,
            full_position_ids,
        )
        hidden_streams = self.veomni_mhc_post(attn_out, hidden_streams, post, comb)

        post, comb, collapsed = self.ffn_hc(hidden_streams)
        mlp_out = self.mlp(self.post_attention_layernorm(collapsed))
        hidden_streams = self.veomni_mhc_post(mlp_out, hidden_streams, post, comb)

        draft = markov_bias = None
        if hasattr(self, "norm"):
            draft = self.collapse_draft(hidden_streams)
            if input_ids is not None:
                markov_bias, _ = self.markov_head(input_ids)
        return hidden_streams, main_x, draft, markov_bias


def build_dspark_blocks(
    config: DeepseekV4Config,
    *,
    attention_cls: type[nn.Module] | None = None,
    moe_cls: type[nn.Module] | None = None,
    rms_norm_cls: type[nn.Module] | None = None,
    hyper_connection_cls: type[nn.Module] | None = None,
    hyper_head_cls: type[nn.Module] | None = None,
) -> nn.ModuleList:
    """Build ``num_nextn_predict_layers`` DSpark stages, or empty ``ModuleList`` when disabled."""
    n_stages = getattr(config, "num_nextn_predict_layers", 0)
    block_size = getattr(config, "dspark_block_size", 0)
    if n_stages <= 0 or block_size <= 0:
        return nn.ModuleList()
    return nn.ModuleList(
        DeepseekV4DSparkBlock(
            config,
            config.num_hidden_layers + stage,
            attention_cls=attention_cls,
            moe_cls=moe_cls,
            rms_norm_cls=rms_norm_cls,
            hyper_connection_cls=hyper_connection_cls,
            hyper_head_cls=hyper_head_cls,
        )
        for stage in range(n_stages)
    )


__all__ = [
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "DeepseekV4DSparkBlock",
    "build_dspark_blocks",
    "dense_sparse_attn",
    "main_window_indices",
]
