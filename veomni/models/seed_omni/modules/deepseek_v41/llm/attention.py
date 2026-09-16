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

"""V4.1 attention: a sliding window and a compressed history in one sparse call.

Every query attends to two KV sources concatenated into a single sparse
attention call -- the last ``sliding_window`` raw entries, and, on layers whose
``compress_ratio`` is non-zero, up to ``index_topk`` compressed entries chosen by
the indexer. Only ``kv_source_layer_ids`` actually compress; the layers between
one source and the next read what it published.

This is a rewrite rather than a subclass of ``DeepseekV4Attention``, for the
reason set out in ``configuration.py``: V4 dispatches on layer *type* names with
the rate looked up per type, V4.1 on a per-layer integer ratio with the
compression owned by a different layer than the ones using it. The released
implementation is also inference-only, so four things have no counterpart here.

**The caches are gone.** Training is prefill over a whole sequence, i.e. the
reference's ``start_pos == 0`` branch and nothing else. ``window_kv_cache``,
``compress_kv_cache``, ``Indexer.k_cache`` and the compressor's partial-group
``kv_state`` / ``score_state`` all exist to carry state between decode steps.
What survives is their *contents*: the window KV is this forward's own KV, and
the compressed KV is this forward's own latents.

**The shared-state global is threaded explicitly.** The reference publishes
compressed KV, index keys, top-k selections and candidate blocks through a
module-level ``shared_attn`` singleton. Under activation checkpointing a layer's
forward runs twice, and on the second run a global would hold whatever the
*last* source published rather than this layer's own source -- silently
attending to the wrong history. Passing :class:`SharedAttentionState` down the
layer loop costs nothing and cannot go stale.

**Rotation is out of place.** The reference's ``apply_rotary_emb`` ends in
``y.copy_(x)``, which is free during inference and here would either break
autograd or corrupt a saved activation.

**Quantization becomes fake quantization.** The in-place ``act_quant`` /
``fp4_act_quant`` round-trips become the hooks in ``quantization.py``, which
round the same tensors without materializing the low-precision form.

Packed sequences are the one concept this file adds. Dynamic batching hands the
model several samples in one row, so a compression group must not straddle a
boundary and a query must not see another sample's groups. Both fall out of
:class:`CompressionPlan`, built once per forward per distinct ratio; an unpacked
sequence is the single-slice case of the same construction rather than a second
code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ......ops.kernels.deepseek_v4 import sparse_attn_tilelang
from .configuration import DeepseekV41LlmConfig
from .quantization import fake_quant_act, fake_quant_kv, fake_quant_linear
from .reuse import DeepseekV4GroupedLinear, DeepseekV4RMSNorm


#: Query rows scored per chunk by the eager indexer. Its ``[B, S, H, C]``
#: intermediate is by far the largest tensor in the stack -- 32 heads, 32k
#: queries and 32k compressed entries would be 64 GiB in bf16 -- so the query
#: dimension is walked in chunks and reduced over heads inside each one.
INDEXER_QUERY_CHUNK = 512


@dataclass
class SharedAttentionState:
    """What a source layer publishes for the layers that follow it.

    The explicit stand-in for the reference's module-level ``shared_attn``: one
    instance per forward, passed down the layer loop, every field overwritten by
    each new source and read by every layer until the next one.

    ``compress_kv`` and ``index_k`` are republished by ``kv_source_layer_ids``,
    ``topk_idxs`` by ``index_source_layer_ids`` and ``candidates`` once, by
    ``candidate_source_layer_id``. Those three sets are nested but not equal,
    which is why this is three fields and not one bundle.
    """

    compress_kv: Optional[torch.Tensor] = None
    index_k: Optional[torch.Tensor] = None
    topk_idxs: Optional[torch.Tensor] = None
    candidates: Optional[torch.Tensor] = None


@dataclass
class CompressionPlan:
    """Which tokens form each compression group, and which groups a query sees.

    Built once per forward for each distinct non-zero ratio and shared by every
    layer using it, because it depends only on ``position_ids`` and the ratio.

    ``window_indices`` is ``[n_groups, ratio]`` of token indices: gathering with
    it is what keeps a group inside one packed sample, since incomplete tails at
    a sample's end simply have no row. ``window_positions`` is each group's RoPE
    position -- the position of its *first* token, which is what the reference
    means by ``group j takes position j * ratio``.

    ``range_starts`` / ``range_ends`` are the half-open interval of group indices
    query ``t`` may attend to. Packing is entirely contained in these two: a
    sample's groups occupy one contiguous span of the compressed axis, so
    excluding the neighbours and enforcing causality inside the sample are the
    same test.
    """

    ratio: int
    window_indices: torch.Tensor
    window_positions: torch.Tensor
    range_starts: torch.Tensor
    range_ends: torch.Tensor

    @property
    def compressed_len(self) -> int:
        return int(self.window_indices.shape[0])

    def visibility_mask(self, device: torch.device) -> torch.Tensor:
        """``[S, compressed_len]`` bool: may query ``t`` attend to group ``c``?"""
        groups = torch.arange(self.compressed_len, device=device)
        return (groups >= self.range_starts.unsqueeze(-1)) & (groups < self.range_ends.unsqueeze(-1))


def build_compression_plans(
    config: DeepseekV41LlmConfig,
    position_ids: torch.Tensor,
    sequence_slices: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[int, CompressionPlan]:
    """One plan per distinct non-zero ``compress_ratio``, for this forward.

    ``sequence_slices`` are the ``[start, end)`` token spans of the packed
    samples. ``None`` means the batch is not packed and every row is one sample,
    which is the ``((0, seq_len),)`` case -- written that way rather than as a
    separate branch so the packed path is the one that gets exercised by every
    test rather than only by the packed ones.
    """
    if position_ids.ndim != 2:
        raise ValueError(
            f"DeepSeek-V4.1 compression needs [batch, seq_len] position ids, got {position_ids.ndim} dims"
        )
    batch_size, seq_len = position_ids.shape
    if batch_size > 1 and not bool((position_ids == position_ids[:1]).all()):
        # Per-row plans would be a ragged gather and every consumer downstream
        # assumes one shared compressed axis. The packed collator produces batch
        # 1, and the padded collator gives every row the same positions, so a
        # mismatch here means neither -- worth a message rather than a wrong
        # answer computed from row 0.
        raise ValueError(
            "DeepSeek-V4.1 compression requires every batch row to share one position layout; "
            "pack the samples into a single row instead of batching rows with different positions"
        )
    if sequence_slices is None:
        sequence_slices = ((0, seq_len),)

    device = position_ids.device
    positions = position_ids[0]
    plans: Dict[int, CompressionPlan] = {}
    for ratio in sorted({r for r in config.compress_ratios if r > 0}):
        starts: List[int] = []
        for begin, end in sequence_slices:
            starts.extend(range(begin, end - ratio + 1, ratio))
        window_starts = torch.tensor(starts, device=device, dtype=torch.long)
        offsets = torch.arange(ratio, device=device, dtype=torch.long)
        window_indices = window_starts[:, None] + offsets[None, :]

        range_starts = torch.empty(seq_len, device=device, dtype=torch.int32)
        range_ends = torch.empty_like(range_starts)
        compressed_offset = 0
        for begin, end in sequence_slices:
            group_count = (end - begin) // ratio
            visible = torch.clamp((positions[begin:end] + 1) // ratio, max=group_count)
            range_starts[begin:end] = compressed_offset
            range_ends[begin:end] = compressed_offset + visible.to(torch.int32)
            compressed_offset += group_count

        plans[ratio] = CompressionPlan(
            ratio=ratio,
            window_indices=window_indices,
            window_positions=positions[window_starts] if window_starts.numel() else window_starts,
            range_starts=range_starts,
            range_ends=range_ends,
        )
    return plans


def rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """``(-x1, x0, -x3, x2, ...)`` -- the adjacent-pair convention."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_interleaved_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the trailing ``2 * cos.shape[-1]`` channels of ``x``, leaving NoPE alone.

    ``cos`` / ``sin`` arrive half-sized, one entry per rotated pair, and broadcast
    against ``x``'s leading dimensions. Taking adjacent elements as a complex pair
    makes this the same rotation the reference writes with ``view_as_complex``;
    passing ``-sin`` conjugates it, which is how the attention output gets the
    query's rotation removed so the KV can stay in one shared rotated form.
    """
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = (rope.float() * cos + rotate_half_interleaved(rope).float() * sin).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


class DeepseekV41RotaryEmbedding(nn.Module):
    """The two rotary schedules V4.1 uses, selected per layer by compress ratio.

    Written out rather than driven through ``DeepseekV4RotaryEmbedding`` and
    HuggingFace's ``ROPE_INIT_FUNCTIONS``. The schedules agree on the frequency
    ramp, but HF's YaRN also returns an ``attention_scaling`` (mscale) factor it
    folds into cos/sin, and the released ``precompute_freqs_cis`` has no such
    factor -- its output is ``polar(ones, freqs)``, magnitude exactly one. Going
    through the HF initializer would scale every compressed layer's logits by
    ``0.1 * ln(16) + 1`` against the checkpoint.

    The two schedules:

    ``main``
        ``compress_ratio == 0`` layers -- plain rotary at ``rope_theta``, YaRN
        disabled. These attend only inside the sliding window, so no frequency
        ever has to describe a distance beyond it.
    ``compress``
        every compressed layer, and the compressed latents themselves -- YaRN at
        ``compress_rope_theta`` with the config's ``rope_scaling`` band.
    """

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__()
        scaling = config.rope_scaling or {}
        self.register_buffer(
            "main_inv_freq",
            self._inv_freq(config.qk_rope_head_dim, config.rope_theta),
            persistent=False,
        )
        self.register_buffer(
            "compress_inv_freq",
            self._inv_freq(
                config.qk_rope_head_dim,
                config.compress_rope_theta,
                original_seq_len=int(scaling.get("original_max_position_embeddings", 0)),
                factor=float(scaling.get("factor", 1.0)),
                beta_fast=float(scaling.get("beta_fast", 32.0)),
                beta_slow=float(scaling.get("beta_slow", 1.0)),
            ),
            persistent=False,
        )

    @staticmethod
    def _inv_freq(
        dim: int,
        base: float,
        original_seq_len: int = 0,
        factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> torch.Tensor:
        """One frequency per rotated pair, with YaRN's ramp when ``original_seq_len``.

        Dimensions whose wavelength already fits inside the training context keep
        their frequency, those far beyond it are divided by ``factor``, and the
        ``beta_fast``..``beta_slow`` band between is faded linearly across.
        """
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if original_seq_len <= 0:
            return freqs

        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        return freqs / factor * (1 - smooth) + freqs * smooth

    def forward(self, positions: torch.Tensor, compressed: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """``[..., ]`` positions -> half-sized fp32 ``(cos, sin)`` with a trailing pair axis."""
        inv_freq = self.compress_inv_freq if compressed else self.main_inv_freq
        freqs = positions.float().unsqueeze(-1) * inv_freq
        return freqs.cos(), freqs.sin()


class DeepseekV41Compressor(nn.Module):
    """Pool ``compress_ratio`` consecutive tokens into one KV latent.

    Ratio 2 pools with a learned softmax gate; ratio 1 is a plain projection with
    no gate at all, which is the third layer kind V4's two-name taxonomy has no
    slot for. The latent is returned **before** rotation, because the indexer
    needs the unrotated form -- the caller rotates afterwards.

    The released module keeps ``wkv`` and ``wgate`` in fp32 when the ratio
    exceeds one, since the softmax pooling runs in fp32. Holding fp32 parameters
    inside an otherwise bf16 model would give FSDP2 a mixed parameter dtype to
    shard, so the parameters stay in the model dtype here and the *arithmetic* is
    promoted instead. That reproduces the reference's accumulation order and its
    fp32 softmax; what it does not reproduce is fp32 weight storage, which
    matters only to the weights' own rounding.
    """

    def __init__(self, config: DeepseekV41LlmConfig, layer_idx: int) -> None:
        super().__init__()
        self.compress_ratio = config.compress_ratios[layer_idx]
        if self.compress_ratio < 1:
            raise ValueError(f"Layer {layer_idx} has compress_ratio {self.compress_ratio} and cannot compress")
        self.head_dim = config.head_dim
        self.norm = DeepseekV4RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False) if self.compress_ratio > 1 else None

    def forward(self, hidden_states: torch.Tensor, plan: CompressionPlan) -> torch.Tensor:
        """``[B, S, hidden]`` -> ``[B, n_groups, head_dim]``, unrotated."""
        if plan.ratio != self.compress_ratio:
            raise ValueError(
                f"Compression plan is for ratio {plan.ratio}, this layer compresses by {self.compress_ratio}"
            )
        if plan.compressed_len == 0:
            return hidden_states.new_zeros((hidden_states.shape[0], 0, self.head_dim))

        if self.wgate is None:
            # One token per group, so there is nothing to pool: take each group's
            # single token and project it. The gather is not a no-op even at ratio
            # 1 -- a packed sample shorter than the ratio contributes no group.
            return self.norm(self.wkv(hidden_states[:, plan.window_indices[:, 0]]))

        x = hidden_states.float()
        kv = torch.nn.functional.linear(x, self.wkv.weight.float())[:, plan.window_indices]
        gate = torch.nn.functional.linear(x, self.wgate.weight.float())[:, plan.window_indices]
        pooled = (kv * gate.softmax(dim=2)).sum(dim=2)
        return self.norm(pooled.to(hidden_states.dtype))


def select_candidate_blocks(
    scores: torch.Tensor,
    range_ends: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level selection: the best ``topk_blocks`` blocks per query.

    ``scores`` is ``[B, S, C]`` with unreachable entries already at ``-inf``, which
    is what lets a block score of ``-inf`` mean "not reachable yet". Returns a
    bool mask shaped like ``scores``, so the layers consuming it mask and never
    think about blocks again.
    """
    width = scores.shape[-1]
    padded = torch.nn.functional.pad(scores, (0, -width % block_size), value=-torch.inf)
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = block_scores.shape[-1]

    # The block holding a query's newest visible group is only partly filled, so
    # pin it: it carries the most recent history but an older full block would
    # otherwise outscore it.
    last = (range_ends.to(torch.long) - 1) // block_size
    block_ids = torch.arange(num_blocks, device=scores.device)
    block_scores = block_scores.masked_fill(block_ids == last.unsqueeze(-1), torch.inf)

    top = block_scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # Fewer reachable blocks than topk_blocks means the leftover picks came back
    # -inf; dropping them here keeps them out of the mask.
    keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class DeepseekV41Indexer(nn.Module):
    """Pick the ``index_topk`` most promising compressed entries for each query.

    A small side attention: query heads scored against one shared key per
    compressed entry, rectified, then combined across heads by ``weights_proj``.
    With a candidate source in play this is the second of two levels, and
    :func:`select_candidate_blocks` is the first.

    Index keys are derived from the compressor's latent, so only a layer that
    compresses its own KV can produce them; every other indexer reads them off
    the shared state.

    The scoring is eager and chunked rather than the fused ``v4_lighting_indexer``
    kernel, and the reason is the two-level scheme rather than a missing kernel.
    That kernel returns the top-k directly and never exposes the full score row,
    but every layer past ``candidate_source_layer_id`` has to mask the full row by
    the candidate blocks *before* taking its own top-k, and the candidate source
    itself has to reduce the full row into block scores. The chunked loop keeps
    the peak at ``INDEXER_QUERY_CHUNK`` query rows' worth of per-head scores.

    **This module's parameters get no gradient from the attention output.** Its
    result is a set of indices, and ``topk`` does not propagate into what it
    ranked, so ``wq_b`` / ``weights_proj`` / ``wk`` are unreachable from the loss
    no matter how the attention behind them is trained. That is the architecture,
    not an oversight -- DeepSeek trains the indexer against the attention's own
    distribution with a separate KL term, which is what ``deepseek_v4``'s
    ``dsa_indexer_loss`` and ``sparse_mqa_target_fwd`` exist for. Until the
    equivalent is wired up for V4.1 the indexer is effectively frozen at whatever
    the checkpoint provides, which is fine for a fine-tune that does not move the
    attention much and is not fine for a long full-parameter run.
    """

    def __init__(self, config: DeepseekV41LlmConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.owns_keys = layer_idx in config.kv_source_layer_ids
        self.is_candidate_source = layer_idx == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_idx
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.index_topk = config.index_topk
        self.n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = config.index_head_dim**-0.5

        self.wq_b = nn.Linear(config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        if self.owns_keys:
            self.wk = nn.Linear(config.head_dim, config.index_head_dim, bias=False)
            self.k_norm = DeepseekV4RMSNorm(config.index_head_dim, eps=config.rms_norm_eps)

    def build_keys(
        self,
        latent: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Turn this layer's unrotated latents into index keys, ``[B, C, index_head_dim]``.

        Has to run before the caller rotates and rounds the latent for the KV
        cache: the reference writes both out of the same storage, and the index
        key is derived from the *unrotated* value.
        """
        keys = self.k_norm(self.wk(latent))
        keys = apply_interleaved_rope(keys, cos, sin)
        return fake_quant_act(keys)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        index_k: torch.Tensor,
        plan: CompressionPlan,
        shared: SharedAttentionState,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """``[B, S, index_topk]`` int32 group indices, ``-1`` where nothing was reachable."""
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        compressed_len = plan.compressed_len

        query = self.wq_b(q_residual).unflatten(-1, (self.n_heads, self.index_head_dim))
        query = apply_interleaved_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2))
        query = fake_quant_act(query)
        weights = self.weights_proj(hidden_states) * (self.softmax_scale * self.n_heads**-0.5)

        visible = plan.visibility_mask(device)
        scores = torch.empty(batch, seq_len, compressed_len, device=device, dtype=torch.float32)
        for begin in range(0, seq_len, INDEXER_QUERY_CHUNK):
            end = min(begin + INDEXER_QUERY_CHUNK, seq_len)
            chunk = torch.einsum("bshd,btd->bsht", query[:, begin:end], index_k)
            chunk = (chunk.float().relu() * weights[:, begin:end].float().unsqueeze(-1)).sum(dim=2)
            scores[:, begin:end] = chunk.masked_fill(~visible[begin:end], -torch.inf)

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                scores, plan.range_ends, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            if shared.candidates is None:
                raise RuntimeError(
                    f"Layer {self.layer_idx} indexes inside the candidate blocks but none were published; "
                    "candidate_source_layer_id must name a layer in index_source_layer_ids"
                )
            scores = scores.masked_fill(~shared.candidates, -torch.inf)

        topk = min(self.index_topk, compressed_len)
        selected = scores.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        # A row with fewer reachable groups than `topk` has `topk` picked anyway,
        # padded with entries scored at -inf; marking them misses is what keeps
        # the attention from reading a group the query cannot see. The filter is
        # visibility and not the candidate mask, matching the released
        # `torch.where(idxs < compress_lens, ...)`: a query short of *candidates*
        # does pad with merely-visible entries. At the shipped settings that
        # cannot arise -- 16384 candidate positions against an index_topk of 512.
        reachable = visible.unsqueeze(0).expand(batch, -1, -1).gather(-1, selected)
        return selected.to(torch.int32).masked_fill(~reachable, -1)


class DeepseekV41Attention(nn.Module):
    """One layer's attention over the sliding window plus the compressed history.

    ``compress_ratio > 0`` does not mean this layer compresses: only a layer in
    ``kv_source_layer_ids`` owns a compressor, and only one in
    ``index_source_layer_ids`` owns an indexer. Everything else reads the shared
    state, which is why those two flags and the ratio are three independent
    facts about a layer rather than one.
    """

    def __init__(self, config: DeepseekV41LlmConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.sliding_window = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.softmax_scale = config.head_dim**-0.5

        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * config.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.wo_a = DeepseekV4GroupedLinear(
            self.n_heads * config.head_dim // self.n_groups,
            self.n_groups * config.o_lora_rank,
            n_groups=self.n_groups,
        )
        self.wo_b = nn.Linear(self.n_groups * config.o_lora_rank, config.hidden_size, bias=False)

        self.compressor = DeepseekV41Compressor(config, layer_idx) if self.is_kv_source else None
        self.indexer = DeepseekV41Indexer(config, layer_idx) if self.is_index_source else None

    @property
    def rope_is_compressed(self) -> bool:
        """Compressed layers use the YaRN schedule; pure sliding-window layers do not."""
        return self.compress_ratio > 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        shared: SharedAttentionState,
        plans: Dict[int, CompressionPlan],
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        cos, sin = rotary_emb(position_ids, compressed=self.rope_is_compressed)

        q_residual = self.q_norm(fake_quant_linear(self.wq_a, hidden_states))
        query = fake_quant_linear(self.wq_b, q_residual).unflatten(-1, (self.n_heads, self.head_dim))
        query = apply_interleaved_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2))

        kv = self.kv_norm(fake_quant_linear(self.wkv, hidden_states))
        kv = apply_interleaved_rope(kv, cos, sin)
        kv = fake_quant_kv(kv, self.rope_head_dim)

        topk_idxs = self._window_indices(position_ids, seq_len)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compressed_kv(
                hidden_states, q_residual, position_ids, rotary_emb, shared, plans, offset=seq_len
            )
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        output = sparse_attn_tilelang(
            query.contiguous(),
            kv.contiguous(),
            self.attn_sink.float().contiguous(),
            topk_idxs.contiguous(),
            self.softmax_scale,
        )
        # The KV is cached in one shared rotated form, so the query's rotation is
        # removed from the output rather than never applied.
        output = apply_interleaved_rope(output, cos.unsqueeze(-2), -sin.unsqueeze(-2))
        grouped = self.wo_a(output.reshape(batch, seq_len, self.n_groups, -1))
        return self.wo_b(grouped.flatten(2))

    def _window_indices(self, position_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        """``[B, S, min(window, S)]`` int32: this query's own causal window.

        ``position_ids`` restarts at zero on each packed sample, so
        ``offset <= position_ids[q]`` is simultaneously the causality test and the
        packed-boundary test -- an offset larger than the query's position would
        reach into the previous sample.
        """
        device = position_ids.device
        width = min(self.sliding_window, seq_len)
        queries = torch.arange(seq_len, device=device, dtype=torch.int32)
        offsets = torch.arange(width - 1, -1, -1, device=device, dtype=torch.int32)
        indices = (queries[None, :, None] - offsets[None, None, :]).expand(position_ids.shape[0], -1, -1)
        return indices.masked_fill(offsets[None, None, :] > position_ids.to(torch.int32)[:, :, None], -1)

    def _compressed_kv(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        shared: SharedAttentionState,
        plans: Dict[int, CompressionPlan],
        offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The shared compressed KV and this layer's selection into it.

        The ordering here is the one invariant worth stating: a source layer's
        indexer consumes the *unrotated* latent, so the index keys are built
        before the latent is rotated and rounded for the KV side. The reference
        gets this by writing both out of the same buffer in a fixed order; doing
        it out of order would train the indexer against keys the deployed model
        never produces.
        """
        plan = plans[self.compress_ratio]

        if self.is_kv_source:
            latent = self.compressor(hidden_states, plan)
            group_cos, group_sin = rotary_emb(plan.window_positions, compressed=True)
            if self.indexer is not None:
                shared.index_k = self.indexer.build_keys(latent, group_cos, group_sin)
            rotated = apply_interleaved_rope(latent, group_cos, group_sin)
            shared.compress_kv = fake_quant_kv(rotated, self.rope_head_dim)

        if shared.compress_kv is None:
            raise RuntimeError(
                f"Layer {self.layer_idx} attends over compressed KV but no source layer has published any; "
                "kv_source_layer_ids must contain a layer at or before it"
            )

        if self.indexer is not None:
            if shared.index_k is None:
                raise RuntimeError(
                    f"Layer {self.layer_idx} runs an indexer but no index keys have been published; "
                    "every index source must be at or after a kv source"
                )
            cos, sin = rotary_emb(position_ids, compressed=True)
            shared.topk_idxs = self.indexer(hidden_states, q_residual, shared.index_k, plan, shared, cos, sin)
        elif shared.topk_idxs is None:
            raise RuntimeError(
                f"Layer {self.layer_idx} reuses a top-k selection but none has been published; "
                "index_source_layer_ids must contain a layer at or before it"
            )

        selection = torch.where(shared.topk_idxs >= 0, shared.topk_idxs + offset, -1)
        return shared.compress_kv, selection


__all__ = [
    "CompressionPlan",
    "DeepseekV41Attention",
    "DeepseekV41Compressor",
    "DeepseekV41Indexer",
    "DeepseekV41RotaryEmbedding",
    "SharedAttentionState",
    "apply_interleaved_rope",
    "build_compression_plans",
    "select_candidate_blocks",
]
