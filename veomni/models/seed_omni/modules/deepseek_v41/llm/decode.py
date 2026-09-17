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

"""The decode-step half of V4.1 attention: one query against cached KV.

``attention.py`` is prefill: it builds every layer's KV from the hidden states in
front of it and throws the result away. Decoding needs the opposite -- one new
query per step against KV that was computed on earlier steps and must survive
them -- so this file holds the state prefill deliberately does not keep.

What makes that tractable is a property of the kernel rather than of the model:
``sparse_attn_tilelang`` takes ``kv`` as an unordered *pool* of rows and
``topk_idxs`` as indices into it. Nothing requires ``kv[0:S]`` to be the current
forward's own tokens. A decode step therefore reuses the prefill call shape
exactly, with the pool swapped for ``cat([window_cache, compress_cache])``; only
the index arithmetic differs, and it is the same arithmetic evaluated at one
query instead of ``S``.

Attention here runs dense rather than through the sparse kernel, which is a
deliberate difference from prefill and not a fallback of convenience:

* at ``S == 1`` the gathered pool is ``[1, 1, window + index_topk, head_dim]`` --
  640 rows at the production shape. Materializing it costs less than launching a
  kernel tiled for thousands of query rows;
* the kernel needs SM90+ and TileLang. Decode is how the model is *inspected* --
  a chat transcript, a logit dump -- and tying that to the training accelerator
  means the cheapest question about a checkpoint needs the scarcest hardware;
* it makes the decode path checkable against the kernel rather than only against
  itself, which is what ``tests/seed_omni/test_deepseek_v41_decode.py`` does.

``dense_sparse_attn`` is therefore held to being the kernel's equal, not its
approximation, and the sink is the part worth naming: it enters the denominator
and contributes no value, so dropping it leaves a distribution that still sums
to one and is wrong by whatever mass the sink held.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from .attention import (
    CompressionPlan,
    DeepseekV41Attention,
    DeepseekV41RotaryEmbedding,
    SharedAttentionState,
    apply_interleaved_rope,
    select_candidate_blocks,
)
from .mhc import expand_to_streams, identity_pre_mix, mhc_collapse, mhc_expand
from .quantization import fake_quant_act, fake_quant_kv, fake_quant_linear


if TYPE_CHECKING:
    from .decoder_layer import DeepseekV41DecoderLayer
    from .modeling import DeepseekV41Backbone


def dense_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
) -> "torch.Tensor | Tuple[torch.Tensor, torch.Tensor]":
    """Sparse MQA by gather-and-softmax, matching ``sparse_attn_tilelang``.

    Args:
        q: ``[B, S, H, D]``.
        kv: ``[B, S_kv, D]`` -- one KV head shared by all ``H`` query heads, and
            the same rows serve as keys and as values.
        attn_sink: ``[H]`` fp32, an additive logit per head that takes softmax
            mass without contributing a value.
        topk_idxs: ``[B, S, topk]`` int32 into ``kv``'s row axis. Negative or
            out-of-range entries are masked, which is how the window's
            not-yet-filled slots and the indexer's misses both arrive.
        sm_scale: softmax scale, defaulting to ``D ** -0.5``.
        return_lse: also return the ``[B, S, H]`` fp32 log-sum-exp, sink
            included. **Base two**, matching the kernel -- see below.

    Returns:
        ``[B, S, H, D]`` in ``q``'s dtype, and the LSE when asked for it.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    kv_len = kv.shape[1]
    # Out-of-range is masked rather than rejected: the kernel treats any index it
    # cannot read as a miss, and a decode step legitimately produces them from a
    # window that is not yet full.
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    safe = topk_idxs.clamp(0, max(kv_len - 1, 0)).long()

    gathered = kv[:, None, :, :].expand(-1, q.shape[1], -1, -1)
    gathered = gathered.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, kv.shape[-1]))

    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * sm_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    # The sink is part of the softmax, so it belongs in the max the exponentials
    # are shifted by. Shifting by the scores' max alone is stable for the output
    # -- 0/inf is 0 -- but sends the LSE to +inf on a fully masked query, where
    # that max is -inf: the shift then becomes exp(sink + 1e30). Folding the sink
    # in leaves such a query with denominator 1 and LSE exactly the sink, which
    # is what a query attending to nothing but the sink means.
    shift = torch.maximum(scores.max(dim=-1).values, attn_sink.view(1, 1, -1))
    exp_scores = (scores - shift.unsqueeze(-1)).exp()
    numerator = torch.einsum("bmhk,bmkd->bmhd", exp_scores, gathered.float())
    denominator = exp_scores.sum(-1) + (attn_sink.view(1, 1, -1) - shift).exp()

    out = (numerator / denominator.unsqueeze(-1)).to(q.dtype)
    if not return_lse:
        return out
    # The kernel exponentiates with exp2 and returns a base-two LSE; its wrapper
    # documents the tensor only as "the log-sum-exp". Returning natural log here
    # would be wrong by log2(e) = 1.4427 -- and wrong in a way that survives every
    # check that does not compare against the kernel, since the consumer
    # (``sparse_mqa_target_fwd``, for the indexer teacher) takes the LSE as a
    # renormalizer and a uniformly scaled one still yields a distribution.
    return out, ((shift + denominator.log()) / math.log(2.0)).float()


@dataclass
class DeepseekV41AttentionCache:
    """One layer's carried state, and nothing a forward could recompute.

    Three tensors with three different growth laws, which is why they are three
    fields rather than one buffer:

    ``window``
        the last ``sliding_window`` KV rows this layer produced, already rotated
        and rounded. Bounded, and the only field every layer has.
    ``compress_kv`` / ``index_k``
        the compressed history, append-only and unbounded, held solely by layers
        in ``kv_source_layer_ids``. Group ``j`` is a function of the hidden states
        at positions ``[j * ratio, (j + 1) * ratio)`` and of nothing later, so
        once written it never changes -- the property the whole scheme rests on.
    ``pending``
        the hidden states of the group still being filled, at most ``ratio - 1``
        of them. Prefill has no counterpart: it sees every token of every group at
        once and a trailing partial group simply contributes no row.
    """

    sliding_window: int
    compress_ratio: int
    window: Optional[torch.Tensor] = None
    compress_kv: Optional[torch.Tensor] = None
    index_k: Optional[torch.Tensor] = None
    pending: Optional[torch.Tensor] = None

    @property
    def window_len(self) -> int:
        return 0 if self.window is None else int(self.window.shape[1])

    @property
    def compressed_len(self) -> int:
        return 0 if self.compress_kv is None else int(self.compress_kv.shape[1])

    def append_window(self, kv: torch.Tensor) -> None:
        """Keep the newest ``sliding_window`` rows, oldest first."""
        self.window = kv if self.window is None else torch.cat([self.window, kv], dim=1)
        if self.window.shape[1] > self.sliding_window:
            self.window = self.window[:, -self.sliding_window :]


@dataclass
class DeepseekV41DecodeCache:
    """Every layer's state, plus the position the next token will occupy."""

    layers: List[DeepseekV41AttentionCache]
    position: int = 0

    @classmethod
    def empty(cls, config) -> "DeepseekV41DecodeCache":
        return cls(
            layers=[
                DeepseekV41AttentionCache(
                    sliding_window=config.sliding_window,
                    compress_ratio=config.compress_ratios[idx],
                )
                for idx in range(config.num_hidden_layers)
            ]
        )


def _group_plan(ratio: int, first_position: int, device: torch.device) -> CompressionPlan:
    """A one-group plan, so the compressor runs its own arithmetic and not a copy.

    ``DeepseekV41Compressor.forward`` reads ``ratio`` and ``window_indices`` and
    nothing else, but reimplementing its gate-softmax pooling here would be a
    second expression of the same thing, free to drift from the trained one.
    """
    return CompressionPlan(
        ratio=ratio,
        window_indices=torch.arange(ratio, device=device, dtype=torch.long).unsqueeze(0),
        window_positions=torch.tensor([first_position], device=device, dtype=torch.long),
        range_starts=torch.zeros(1, device=device, dtype=torch.int32),
        range_ends=torch.ones(1, device=device, dtype=torch.int32),
    )


def _absorb(
    attn: DeepseekV41Attention,
    hidden_states: torch.Tensor,
    position: int,
    rotary_emb: DeepseekV41RotaryEmbedding,
    cache: DeepseekV41AttentionCache,
) -> None:
    """Feed one token to this layer's compressor, emitting a group when it fills.

    Groups are aligned to the sample's first token, so the token at position ``p``
    completes group ``p // ratio`` exactly when ``(p + 1) % ratio == 0``. The
    group's RoPE position is its *first* token's, matching prefill's
    ``positions[window_starts]``.
    """
    ratio = cache.compress_ratio
    cache.pending = hidden_states if cache.pending is None else torch.cat([cache.pending, hidden_states], dim=1)
    if cache.pending.shape[1] < ratio:
        return

    group = cache.pending
    cache.pending = None
    first_position = position - ratio + 1
    plan = _group_plan(ratio, first_position, hidden_states.device)

    latent = attn.compressor(group, plan)
    group_cos, group_sin = rotary_emb(plan.window_positions, compressed=True)
    if attn.indexer is not None:
        keys = attn.indexer.build_keys(latent, group_cos, group_sin)
        cache.index_k = keys if cache.index_k is None else torch.cat([cache.index_k, keys], dim=1)
    rotated = fake_quant_kv(apply_interleaved_rope(latent, group_cos, group_sin), attn.rope_head_dim)
    cache.compress_kv = rotated if cache.compress_kv is None else torch.cat([cache.compress_kv, rotated], dim=1)


def _decode_topk(
    attn: DeepseekV41Attention,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    index_k: torch.Tensor,
    shared: SharedAttentionState,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """This query's ``[B, 1, topk]`` selection into the compressed cache.

    The prefill indexer masks its scores by a visibility mask before ranking.
    Here there is nothing to mask: a query at position ``p`` may see
    ``(p + 1) // ratio`` groups, and that is exactly how many the cache holds once
    this step's ``_absorb`` has run. Every cached group is reachable, so no
    selection can come back a miss and none is marked one.
    """
    indexer = attn.indexer
    compressed_len = int(index_k.shape[1])

    query = indexer.wq_b(q_residual).unflatten(-1, (indexer.n_heads, indexer.index_head_dim))
    query = apply_interleaved_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2))
    query = fake_quant_act(query)
    weights = indexer.weights_proj(hidden_states) * (indexer.softmax_scale * indexer.n_heads**-0.5)

    scores = torch.einsum("bshd,btd->bsht", query, index_k)
    scores = (scores.float().relu() * weights.float().unsqueeze(-1)).sum(dim=2)

    ranked = scores
    if indexer.is_candidate_source:
        range_ends = torch.full((1, 1), compressed_len, device=scores.device, dtype=torch.int32)
        shared.candidates = select_candidate_blocks(
            scores, range_ends, indexer.candidate_topk_blocks, indexer.candidate_block_size
        )
    elif indexer.uses_candidates:
        if shared.candidates is None:
            raise RuntimeError(
                f"Layer {indexer.layer_idx} indexes inside the candidate blocks but none were published; "
                "candidate_source_layer_id must name a layer in index_source_layer_ids"
            )
        ranked = scores.masked_fill(~shared.candidates, -torch.inf)

    topk = min(indexer.index_topk, compressed_len)
    return ranked.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values.to(torch.int32)


def decode_attention_step(
    attn: DeepseekV41Attention,
    hidden_states: torch.Tensor,
    position: int,
    rotary_emb: DeepseekV41RotaryEmbedding,
    shared: SharedAttentionState,
    cache: DeepseekV41AttentionCache,
) -> torch.Tensor:
    """One layer's attention for one new token, against its cached KV.

    A free function rather than a method on :class:`DeepseekV41Attention` so the
    training forward keeps exactly one code path and cannot acquire a decode
    branch it never takes. Every projection, rotation and rounding below is read
    off ``attn``'s own submodules, so the two paths share weights *and* the
    arithmetic applied to them.

    Args:
        hidden_states: ``[B, 1, hidden]``.
        position: the token's absolute position in the sample.
        shared: republished by source layers exactly as in prefill, and carried
            across layers within this one step -- not across steps.
        cache: this layer's state, mutated in place.

    Returns:
        ``[B, 1, hidden]``.
    """
    if hidden_states.shape[1] != 1:
        raise ValueError(f"decode steps carry one token, got {hidden_states.shape[1]}")

    batch = hidden_states.shape[0]
    device = hidden_states.device
    position_ids = torch.tensor([[position]], device=device, dtype=torch.long).expand(batch, 1)
    cos, sin = rotary_emb(position_ids, compressed=attn.rope_is_compressed)

    q_residual = attn.q_norm(fake_quant_linear(attn.wq_a, hidden_states))
    query = fake_quant_linear(attn.wq_b, q_residual).unflatten(-1, (attn.n_heads, attn.head_dim))
    query = apply_interleaved_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2))

    kv = attn.kv_norm(fake_quant_linear(attn.wkv, hidden_states))
    kv = apply_interleaved_rope(kv, cos, sin)
    kv = fake_quant_kv(kv, attn.rope_head_dim)
    cache.append_window(kv)

    # The window rows are all causal and all inside the window by construction,
    # so the selection is every row the cache holds. Prefill pads its width to
    # ``min(window, S)`` with misses; padding to the same width here would add
    # slots that contribute nothing.
    window_len = cache.window_len
    pool = cache.window
    topk_idxs = torch.arange(window_len, device=device, dtype=torch.int32).view(1, 1, -1).expand(batch, 1, -1)

    if attn.compress_ratio:
        if attn.is_kv_source:
            _absorb(attn, hidden_states, position, rotary_emb, cache)
            # An empty tensor, not None, while the first group is still filling:
            # the source layer *did* publish, it has nothing yet. Prefill says the
            # same thing the same way -- ``DeepseekV41Compressor.forward`` returns
            # ``[B, 0, head_dim]`` for an empty plan -- which leaves None meaning
            # only "no source layer ran", the condition worth raising on.
            if cache.compress_kv is None:
                cache.compress_kv = hidden_states.new_zeros((batch, 0, attn.head_dim))
                if attn.indexer is not None:
                    cache.index_k = hidden_states.new_zeros((batch, 0, attn.indexer.index_head_dim))
            shared.compress_kv = cache.compress_kv
            if attn.indexer is not None:
                shared.index_k = cache.index_k
        if shared.compress_kv is None:
            raise RuntimeError(
                f"Layer {attn.layer_idx} attends over compressed KV but no source layer has published any; "
                "kv_source_layer_ids must contain a layer at or before it"
            )
        # Before the first group completes there is nothing compressed to attend
        # to, and the query runs on its window alone -- the decode counterpart of
        # prefill's early positions, whose visibility row is empty.
        if shared.compress_kv.shape[1]:
            if attn.indexer is not None:
                if shared.index_k is None:
                    raise RuntimeError(
                        f"Layer {attn.layer_idx} runs an indexer but no index keys have been published; "
                        "every index source must be at or after a kv source"
                    )
                index_cos, index_sin = rotary_emb(position_ids, compressed=True)
                shared.topk_idxs = _decode_topk(
                    attn, hidden_states, q_residual, shared.index_k, shared, index_cos, index_sin
                )
            elif shared.topk_idxs is None:
                raise RuntimeError(
                    f"Layer {attn.layer_idx} reuses a top-k selection but none has been published; "
                    "index_source_layer_ids must contain a layer at or before it"
                )
            pool = torch.cat([pool, shared.compress_kv], dim=1)
            selection = torch.where(shared.topk_idxs >= 0, shared.topk_idxs + window_len, -1)
            topk_idxs = torch.cat([topk_idxs, selection], dim=-1)

    output = dense_sparse_attn(
        query.contiguous(),
        pool.contiguous(),
        attn.attn_sink.float().contiguous(),
        topk_idxs.contiguous(),
        attn.softmax_scale,
    )
    output = apply_interleaved_rope(output, cos.unsqueeze(-2), -sin.unsqueeze(-2))
    grouped = attn.wo_a(output.reshape(batch, 1, attn.n_groups, -1))
    return attn.wo_b(grouped.flatten(2))


def decode_layer_step(
    layer: "DeepseekV41DecoderLayer",
    hidden_streams: torch.Tensor,
    pre_mix: torch.Tensor,
    position: int,
    rotary_emb: DeepseekV41RotaryEmbedding,
    shared: SharedAttentionState,
    cache: DeepseekV41AttentionCache,
    image_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One decoder layer for one token: ``[B, 1, hc, D]`` in, that and the next ``pre``.

    A line-for-line copy of :meth:`DeepseekV41DecoderLayer.forward` with the
    attention call swapped, because everything else in that method -- both Hyper-
    Connections, both collapses, the expansions and the MoE -- is a function of
    one position and carries nothing between tokens. Only the attention had
    history to keep.

    The copy is deliberate but it is still a copy, and the test that holds it to
    the original is the one that matters: if ``forward`` grows a step this misses,
    the per-position comparison stops agreeing.
    """
    residual = hidden_streams
    attn_pre, attn_post, attn_comb = layer.attn_hc(hidden_streams)
    collapsed = mhc_collapse(hidden_streams, pre_mix)
    attn_out = decode_attention_step(
        layer.self_attn, layer.input_layernorm(collapsed), position, rotary_emb, shared, cache
    )
    hidden_streams = mhc_expand(attn_out, residual, attn_post, attn_comb)

    residual = hidden_streams
    ffn_pre, ffn_post, ffn_comb = layer.ffn_hc(hidden_streams)
    collapsed = mhc_collapse(hidden_streams, attn_pre)
    mlp_out, _ = layer.mlp(layer.post_attention_layernorm(collapsed), image_mask)
    hidden_streams = mhc_expand(mlp_out, residual, ffn_post, ffn_comb)

    # ``attn_pre`` fed the FFN's collapse; ``ffn_pre`` feeds the next site's, which
    # is the V4.1 data flow the mHC rewrite exists for -- a `pre` belongs to the
    # site after the one that produced it.
    return hidden_streams, ffn_pre


def decode_backbone_step(
    backbone: "DeepseekV41Backbone",
    inputs_embeds: torch.Tensor,
    position: int,
    cache: DeepseekV41DecodeCache,
    ngram_embeds: Optional[List[torch.Tensor]] = None,
    image_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The backbone for one token: ``[B, 1, hidden]`` in, normed hidden states out.

    ``expand_to_streams`` and ``identity_pre_mix`` are rebuilt every step rather
    than carried, which is not an oversight: the streams start from this token's
    embedding and the identity ``pre`` is a constant of the shape. Neither is
    history, so neither belongs in a cache.

    ``SharedAttentionState`` likewise is per step. It carries a source layer's
    compressed KV and selection *down* the stack within one token; what carries
    across tokens is the per-layer cache the source layer republishes from.
    """
    if ngram_embeds is None and backbone.engram:
        raise RuntimeError(
            "This backbone has Engram connections at layers "
            f"{sorted(int(idx) for idx in backbone.engram)} but no ngram_embeds were passed. "
            "The deepseek_v41_ngram step must run before this one."
        )

    token_mask = None if image_mask is None else ~image_mask
    shared = SharedAttentionState()
    hidden_streams = expand_to_streams(inputs_embeds, backbone.config.hc_mult)
    pre_mix = identity_pre_mix(hidden_streams, backbone.config.hc_mult)

    for layer_idx, layer in enumerate(backbone.layers):
        if str(layer_idx) in backbone.engram:
            connection = backbone.engram[str(layer_idx)]
            hidden_streams = connection(hidden_streams, ngram_embeds[connection.layer_hash_index], token_mask)
        hidden_streams, pre_mix = decode_layer_step(
            layer,
            hidden_streams,
            pre_mix,
            position,
            backbone.rotary_emb,
            shared,
            cache.layers[layer_idx],
            image_mask,
        )

    return backbone.norm(mhc_collapse(hidden_streams, pre_mix))


__all__ = [
    "DeepseekV41AttentionCache",
    "DeepseekV41DecodeCache",
    "decode_attention_step",
    "decode_backbone_step",
    "decode_layer_step",
    "dense_sparse_attn",
]
