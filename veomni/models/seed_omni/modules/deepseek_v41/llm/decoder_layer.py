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

"""One V4.1 block: attention and MoE, each between an mHC collapse and expand.

The block takes a ``pre`` and returns one, and the two are not the same site's:
attention collapses with the ``pre`` handed in (the previous block's FFN), the
FFN collapses with the ``pre`` this block's attention site produced, and this
block's FFN ``pre`` goes out for the next block. The chain ends at the model,
which collapses the final stream with the last block's FFN ``pre``.

It reads like an off-by-one and it is not. Each site's coefficients are computed
from the stream *before* its own sublayer runs, so using them at the same site
would make the collapse a function of the very tensor it collapses; deferring by
one site makes each ``pre`` a function of a stream that is already final.

The Engram lookup runs at the top of ``forward`` on layers listed in
``config.engram_layer_ids``, before the DSpark draft-input capture and before
attention. Keeping ``self.engram`` on the decoder layer puts its parameters in
the same FSDP2 unit as the layer itself (preserving strict sequential forward
and prefetch ordering) and matches the released ``layers.{i}.engram.*`` names.
"""

from __future__ import annotations

import torch
from transformers.modeling_layers import GradientCheckpointingLayer

from .attention import (
    CompressionPlan,
    DeepseekV41Attention,
    DeepseekV41RotaryEmbedding,
    IndexerLossTerms,
    SharedAttentionState,
)
from .configuration import DeepseekV41LlmConfig
from .engram_connection import DeepseekV41EngramConnection
from .mhc import DeepseekV41HyperConnection, mhc_collapse, mhc_expand, require_fp32_pre_mix
from .moe import DeepseekV41SparseMoeBlock
from .reuse import DeepseekV4RMSNorm
from .sequence_shard import SequenceShard


class DeepseekV41DecoderLayer(GradientCheckpointingLayer):
    """Attention then MoE, both wrapped in manifold-constrained Hyper-Connections."""

    def __init__(self, config: DeepseekV41LlmConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.engram = DeepseekV41EngramConnection(config, layer_idx) if layer_idx in config.engram_layer_ids else None
        self.self_attn = DeepseekV41Attention(config, layer_idx)
        n_routed, n_activated = config.moe_config_for_layer(layer_idx)
        self.mlp = DeepseekV41SparseMoeBlock(
            config, layer_idx=layer_idx, n_routed_experts=n_routed, num_experts_per_tok=n_activated
        )
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.attn_hc = DeepseekV41HyperConnection(config)
        self.ffn_hc = DeepseekV41HyperConnection(config)
        self.veomni_mhc_post = self.attn_hc.veomni_mhc_post

    def forward(
        self,
        hidden_streams: torch.Tensor,
        pre_mix: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        shared: SharedAttentionState,
        plans: dict[int, CompressionPlan],
        image_mask: torch.Tensor | None = None,
        shard: SequenceShard | None = None,
        ngram_embed: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        capture_draft_input: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, IndexerLossTerms | None, torch.Tensor | None]:
        """``[B, S, hc, D]`` in and out, plus the next ``pre``, router logits, indexer KL and draft input.

        The indexer KL and optional DSpark draft input leave as part of the
        return value and never as an attribute on ``self`` or on the hidden
        states. This method is what gradient checkpointing wraps, and a tensor
        reaching the model loop by any other route carries no graph -- under the
        reentrant implementation the first forward runs inside
        ``torch.no_grad()``, and under either one the backward is built from the
        recomputed forward's tensors. The indexer would get no gradient at all
        while the logged KL fell.
        """
        require_fp32_pre_mix(pre_mix)
        if self.engram is not None and ngram_embed is not None:
            # Before attention and before the DSpark hook: the reference adds
            # the lookup to the stream at the top of the loop body, so this
            # layer's attention (and any DSpark target capture on this layer)
            # is the first thing that sees it.
            hidden_streams = self.engram(hidden_streams, ngram_embed, token_mask)
        draft_input = hidden_streams.mean(dim=2) if capture_draft_input else None
        residual = hidden_streams
        attn_pre, attn_post, attn_comb = self.attn_hc(hidden_streams)
        collapsed = mhc_collapse(hidden_streams, pre_mix)
        attn_out, indexer_loss = self.self_attn(
            self.input_layernorm(collapsed),
            position_ids,
            rotary_emb,
            shared,
            plans,
            shard=shard,
        )
        hidden_streams = mhc_expand(attn_out, residual, attn_post, attn_comb, op=self.veomni_mhc_post)

        residual = hidden_streams
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(hidden_streams)
        collapsed = mhc_collapse(hidden_streams, attn_pre)
        mlp_out, router_logits = self.mlp(self.post_attention_layernorm(collapsed), image_mask)
        hidden_streams = mhc_expand(mlp_out, residual, ffn_post, ffn_comb, op=self.veomni_mhc_post)

        return hidden_streams, ffn_pre, router_logits, indexer_loss, draft_input


__all__ = ["DeepseekV41DecoderLayer"]
