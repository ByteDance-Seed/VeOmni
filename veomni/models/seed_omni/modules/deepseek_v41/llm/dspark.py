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

"""DSpark draft stages stored under ``mtp.*`` in the released checkpoint.

Window-only attention that seeds its KV from the backbone's target-layer
inputs, a smaller MoE, and the Markov / confidence heads on the last stage.

``embed`` / ``head`` stay in ``deepseek_v41_text_encoder``. Stage 0's
``main_proj`` concatenates the target-layer hiddens. ``forward_spec`` on the
backbone runs these stages; the text encoder samples the draft block with
the shared LM head plus the Markov residual.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .attention import (
    DeepseekV41Attention,
    DeepseekV41AttentionCache,
    DeepseekV41RotaryEmbedding,
    apply_interleaved_rope,
)
from .configuration import DeepseekV41LlmConfig
from .decode import dense_sparse_attn
from .mhc import DeepseekV41HyperConnection, mhc_collapse, mhc_expand, require_fp32_pre_mix
from .moe import DeepseekV41SparseMoeBlock
from .quantization import fake_quant_linear, round_window_kv
from .quantized import DeepseekV41Linear, linear_quant_dtype
from .reuse import DeepseekV4RMSNorm
from .sequence_shard import SequenceShard


class DSparkMarkovHead(nn.Module):
    """Rank-256 residual over the shared LM head, stored on the last MTP stage."""

    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank)
        self.head = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed(token_ids)
        return self.head(embed), embed


class DSparkConfidenceHead(nn.Module):
    """Scalar confidence from the draft hidden concatenated with the Markov embed."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([hidden, markov_embed], dim=-1).float()).squeeze(-1)


class DeepseekV41DSparkBlock(nn.Module):
    """One DSpark stage: window attention over the backbone's main KV, then MoE."""

    def __init__(self, config: DeepseekV41LlmConfig, layer_idx: int) -> None:
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
        self.mlp = DeepseekV41SparseMoeBlock(config, n_routed_experts=n_routed, num_experts_per_tok=n_activated)
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeepseekV41HyperConnection(config)
        self.ffn_hc = DeepseekV41HyperConnection(config)

        if stage_id == 0:
            n_targets = len(config.dspark_target_layer_ids)
            self.main_proj = DeepseekV41Linear(
                config.hidden_size * n_targets,
                config.hidden_size,
                bias=False,
                quant_dtype=linear_quant_dtype(config),
            )
            self.main_norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if stage_id == config.num_nextn_predict_layers - 1:
            self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(config.vocab_size, config.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(config.hidden_size + config.dspark_markov_rank)

    def project_main(self, draft_hiddens: List[torch.Tensor]) -> torch.Tensor:
        """``main_norm(main_proj(cat(target-layer inputs)))`` -- stage 0 only."""
        if self.stage_id != 0:
            raise RuntimeError("only the first DSpark stage owns main_proj")
        return self.main_norm(self.main_proj(torch.cat(draft_hiddens, dim=-1)))

    def forward(
        self,
        hidden_streams: torch.Tensor,
        pre_mix: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        main: Union[List[torch.Tensor], torch.Tensor],
        shard: Optional[SequenceShard] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """One teacher-forced packed stage: the only way training enters a stage.

        Each stage is an FSDP2 unit, and FSDP gathers its parameters in the
        pre-forward hook. Calling ``project_main`` / ``forward_draft_packed`` /
        ``collapse_draft`` / ``markov_head`` from outside would hand sharded
        DTensors to the matmuls, so all of them run in here.

        ``main`` is the target-layer hidden list at stage 0 and the projected
        ``main_x`` after it. Returns the streams, the carried ``pre`` mix,
        ``main_x``, and on the last stage the LM-head input and -- given the
        slot token ids -- the Markov residual ``forward_head`` adds to its
        logits (else ``None`` for both).
        """
        require_fp32_pre_mix(pre_mix)
        main_x = self.project_main(main) if self.stage_id == 0 else main
        hidden_streams, pre_mix = self.forward_draft_packed(
            hidden_streams, pre_mix, position_ids, rotary_emb, main_x, shard
        )
        draft = markov_bias = None
        if hasattr(self, "norm"):
            draft = self.collapse_draft(hidden_streams, pre_mix)
            if input_ids is not None:
                markov_bias, _ = self.markov_head(input_ids)
        return hidden_streams, pre_mix, main_x, draft, markov_bias

    def seed_window(
        self,
        main_x: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        cache: DeepseekV41AttentionCache,
        *,
        detach: bool = True,
    ) -> None:
        """Write the backbone token's window KV, matching ``DSparkAttention`` at any start_pos.

        Prefill (``S > 1``) replaces the cache with the last ``sliding_window`` rows.
        Decode (``S == 1``) appends into the ring the same way the backbone window does.

        ``detach`` is the decode path: the window is a cache, not a residual.
        Packed training does not use this cache; it rebuilds causal window
        indices over the full row in ``_packed_draft_attention``.
        """
        attn = self.self_attn
        seq_len = main_x.shape[1]
        kv = self._window_kv(main_x, position_ids, rotary_emb)
        if seq_len == 1 and cache.window is not None:
            cache.append_window(kv)
            return
        width = min(attn.sliding_window, seq_len)
        window = kv[:, seq_len - width : seq_len]
        cache.window = window.detach() if detach else window

    def _window_kv(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
    ) -> torch.Tensor:
        """Rotated window KV for ``hidden_states``, matching ``seed_window``."""
        attn = self.self_attn
        cos, sin = rotary_emb(position_ids, compressed=False)
        kv = attn.kv_norm(fake_quant_linear(attn.wkv, hidden_states))
        kv = apply_interleaved_rope(kv, cos, sin)
        return round_window_kv(kv, training=attn.training)

    def forward_draft(
        self,
        hidden_streams: torch.Tensor,
        pre_mix: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        cache: DeepseekV41AttentionCache,
        main_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One DSpark block on a draft block, after the window has been seeded from ``main_x``."""
        residual = hidden_streams
        attn_pre, attn_post, attn_comb = self.attn_hc(hidden_streams)
        collapsed = mhc_collapse(hidden_streams, pre_mix)
        attn_out = self._draft_attention(self.input_layernorm(collapsed), position_ids, rotary_emb, cache, main_x)
        hidden_streams = mhc_expand(attn_out, residual, attn_post, attn_comb)

        residual = hidden_streams
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(hidden_streams)
        collapsed = mhc_collapse(hidden_streams, attn_pre)
        mlp_out, _ = self.mlp(self.post_attention_layernorm(collapsed), image_mask=None)
        hidden_streams = mhc_expand(mlp_out, residual, ffn_post, ffn_comb)
        return hidden_streams, ffn_pre

    def _draft_attention(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        cache: DeepseekV41AttentionCache,
        main_x: torch.Tensor,
    ) -> torch.Tensor:
        """Window attention: queries from the draft tokens, KV = cached main + draft KV.

        ``main_x`` is one verified token (decode) or the prompt (already seeded).
        Official ``DSparkAttention`` also writes that token into the ring here.
        """
        attn = self.self_attn
        if main_x.shape[1] == 1:
            self.seed_window(main_x, position_ids[:, :1], rotary_emb, cache)

        batch, block_size, _ = hidden_states.shape
        # Draft positions sit just past the verified token. Prefill seed used the
        # prompt positions; decode passes the current token's id, so the block is
        # ``p+1 .. p+block_size``.
        draft_positions = position_ids[:, :1] + 1 + torch.arange(block_size, device=hidden_states.device)
        cos, sin = rotary_emb(draft_positions, compressed=False)

        q_residual = attn.q_norm(fake_quant_linear(attn.wq_a, hidden_states))
        query = fake_quant_linear(attn.wq_b, q_residual).unflatten(-1, (attn.n_heads, attn.head_dim))
        query = apply_interleaved_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2))

        kv = attn.kv_norm(fake_quant_linear(attn.wkv, hidden_states))
        kv = apply_interleaved_rope(kv, cos, sin)
        kv = round_window_kv(kv, training=attn.training)

        window = cache.window
        if window is None:
            raise RuntimeError("DSpark window cache is empty; seed it from the backbone token first")
        pool = torch.cat([window, kv], dim=1)
        window_len = window.shape[1]
        topk_idxs = (
            torch.cat(
                [
                    torch.arange(window_len, device=hidden_states.device, dtype=torch.int32),
                    window_len + torch.arange(block_size, device=hidden_states.device, dtype=torch.int32),
                ]
            )
            .view(1, 1, -1)
            .expand(batch, block_size, -1)
        )

        output = dense_sparse_attn(query, pool, attn.attn_sink, topk_idxs, attn.softmax_scale)
        output = apply_interleaved_rope(output, cos.unsqueeze(-2), -sin.unsqueeze(-2))
        grouped = attn.wo_a(output.reshape(batch, block_size, attn.n_groups, -1))
        return attn.wo_b(grouped.flatten(2))

    def forward_draft_packed(
        self,
        hidden_streams: torch.Tensor,
        pre_mix: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        main_x: torch.Tensor,
        shard: Optional[SequenceShard] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced DSpark on a packed row.

        Decode's ``forward_draft`` rebuilds draft positions as ``p+1 .. p+block``
        and attends densely inside that block. Training already has the packed
        ``position_ids``; using the decode formula would put every token at the
        wrong RoPE phase, and seeding only the last window would let position 0
        read the end of the row. Every row runs as block slot 0 once, over the
        full sequence (see ``_packed_draft_attention``); slots past 0 -- the
        noise tokens -- are not trained.
        """
        residual = hidden_streams
        attn_pre, attn_post, attn_comb = self.attn_hc(hidden_streams)
        collapsed = mhc_collapse(hidden_streams, pre_mix)
        attn_out = self._packed_draft_attention(
            self.input_layernorm(collapsed), position_ids, rotary_emb, main_x, shard
        )
        hidden_streams = mhc_expand(attn_out, residual, attn_post, attn_comb)

        residual = hidden_streams
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(hidden_streams)
        collapsed = mhc_collapse(hidden_streams, attn_pre)
        mlp_out, _ = self.mlp(self.post_attention_layernorm(collapsed), image_mask=None)
        hidden_streams = mhc_expand(mlp_out, residual, ffn_post, ffn_comb)
        return hidden_streams, ffn_pre

    def _packed_draft_attention(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        main_x: torch.Tensor,
        shard: Optional[SequenceShard] = None,
    ) -> torch.Tensor:
        """Slot-0 ``DSparkAttention`` at every packed position.

        Decode at verified token ``p`` puts slot token ``x[p+1]`` at position
        ``p+1`` and attends the main-KV ring (``main_x`` of ``.. p``, each at its
        own RoPE position) plus the block's own keys. Teacher forcing the slot
        at every row ``j`` therefore reads main rows ``j-width .. j-1`` of the
        same sample and its own draft key -- never ``main_x[j]`` (that token's
        backbone state is what the slot is predicting past) and never earlier
        draft rows (decode's ring only holds main KV). Rows at a sample start
        see no main KV; the text encoder masks their labels.

        Under ``shard`` the queries are this rank's rows and the main KV is
        gathered to the full row, so a window may reach into the previous rank.
        """
        if main_x.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                f"packed DSpark expected main_x shape {tuple(hidden_states.shape[:2])}, got {tuple(main_x.shape[:2])}"
            )
        attn = self.self_attn
        batch, seq_len, _ = hidden_states.shape
        cos, sin = rotary_emb(position_ids, compressed=False)

        q_residual = attn.q_norm(fake_quant_linear(attn.wq_a, hidden_states))
        query = fake_quant_linear(attn.wq_b, q_residual).unflatten(-1, (attn.n_heads, attn.head_dim))
        query = apply_interleaved_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2))

        kv = self._window_kv(hidden_states, position_ids, rotary_emb)
        main_kv = self._window_kv(main_x, position_ids, rotary_emb)
        if shard is None:
            main_idxs = self._main_window_indices(position_ids)
        else:
            main_kv = shard.gather(main_kv)
            main_idxs = shard.local_rows(self._main_window_indices(shard.position_ids))
        pool = torch.cat([main_kv, kv], dim=1)
        own_idxs = main_kv.shape[1] + torch.arange(seq_len, device=hidden_states.device, dtype=torch.int32)
        topk_idxs = torch.cat([main_idxs, own_idxs.view(1, -1, 1).expand(batch, -1, -1)], dim=-1)

        output = dense_sparse_attn(query, pool, attn.attn_sink, topk_idxs, attn.softmax_scale)
        output = apply_interleaved_rope(output, cos.unsqueeze(-2), -sin.unsqueeze(-2))
        grouped = attn.wo_a(output.reshape(batch, seq_len, attn.n_groups, -1))
        return attn.wo_b(grouped.flatten(2))

    def _main_window_indices(self, position_ids: torch.Tensor) -> torch.Tensor:
        """``[B, S, min(window, S)]`` int32: main rows ``q-width .. q-1`` of the query's own sample.

        ``offset > position_ids[q]`` would reach before the sample start, so
        packed ``position_ids`` restarting at zero doubles as the boundary mask.
        """
        seq_len = position_ids.shape[1]
        device = position_ids.device
        width = min(self.self_attn.sliding_window, seq_len)
        queries = torch.arange(seq_len, device=device, dtype=torch.int32)
        offsets = torch.arange(width, 0, -1, device=device, dtype=torch.int32)
        indices = (queries[None, :, None] - offsets[None, None, :]).expand(position_ids.shape[0], -1, -1)
        return indices.masked_fill(offsets[None, None, :] > position_ids.to(torch.int32)[:, :, None], -1)

    def collapse_draft(self, hidden_streams: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Last-stage ``norm(hc_pre(stream))`` -- the tensor the shared LM head reads."""
        if not hasattr(self, "norm"):
            raise RuntimeError("only the last DSpark stage owns the draft norm")
        return self.norm(mhc_collapse(hidden_streams, pre_mix))


def build_dspark_blocks(config: DeepseekV41LlmConfig) -> nn.ModuleList:
    """``num_nextn_predict_layers`` stages, or empty when the config has none."""
    n_stages = config.num_nextn_predict_layers
    if n_stages <= 0 or config.dspark_block_size <= 0:
        return nn.ModuleList()
    return nn.ModuleList(DeepseekV41DSparkBlock(config, config.num_hidden_layers + stage) for stage in range(n_stages))


__all__ = [
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "DeepseekV41DSparkBlock",
    "build_dspark_blocks",
]
