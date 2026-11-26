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

**Quantization becomes a storage round-trip.** The reference's in-place
``act_quant`` / ``fp4_act_quant`` writes are the hooks in ``quantization.py``:
window KV as FP8/32, compressed KV as FP4/16 with E4M3 scales, indexer Q/K as
FP4/32 with E8M0. Inference always rounds; training rounds only under QAT.

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
from typing import Sequence

import torch
import torch.nn as nn

from ......ops import VeomniOp
from ......ops.kernels.dsa.sparse_mqa_target import sparse_attn_tilelang, sparse_mqa_target_fwd
from ......utils.device import IS_CUDA_AVAILABLE, get_torch_device
from ......utils.import_utils import is_package_available
from .configuration import DeepseekV41LlmConfig
from .quantization import round_compress_kv, round_index_act, round_window_kv
from .quantized import (
    DeepseekV41GroupedLinear,
    DeepseekV41Linear,
    linear_quant_dtype,
    quantized_linear_pair,
)
from .reuse import DeepseekV4RMSNorm, resolve_module_op_impl
from .sequence_shard import SequenceShard


_HAS_TRITON = is_package_available("triton")
if _HAS_TRITON:
    import triton
    import triton.language as tl

if is_package_available("tilelang"):
    from ......ops.kernels.dsa.vendor import (
        tilelang_indexer_fwd as _dsa_tl_indexer_fwd,  # noqa: F401
    )
    from ......ops.kernels.dsa.vendor import (
        tilelang_sparse_mla_fwd as _dsa_tl_sparse_mla_fwd,  # noqa: F401
    )


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

    compress_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    topk_idxs: torch.Tensor | None = None
    candidates: torch.Tensor | None = None
    window_indices: torch.Tensor | None = None
    main_rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None
    compress_rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class IndexerLossTerms:
    """One layer's contribution to the indexer objective.

    ``kl`` is summed over query rows and carries the graph; ``uniform`` is the
    zero-information reference the KL has to be read against, detached because it
    is a metric and must never reach the objective. Both are summed over exactly
    the same rows, so their ratio at the end is a ratio of means -- averaging
    per-row ``kl / uniform`` instead would be dominated by the rows with the
    smallest reference, and would still land in ``[0, 1]`` looking plausible.
    """

    kl: torch.Tensor
    uniform: torch.Tensor


def indexer_kl_terms(index_score: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query ``KL(target || softmax(index_score))``, and its uniform reference.

    Ported from ``deepseek_v4``'s helper of the same name, which is DeepSeek-V3.2
    eq. (4); it is a pure function of two tensors and nothing in it is specific to
    either model's layer schedule.

    Args:
        index_score: ``[B, S, C]`` indexer scores at the selected slots, ``-inf``
            at misses.
        target: ``[B, S, C]`` fp32, zero at misses, and per row either
            L1-normalised or identically zero where the teacher had no mass.

    Returns:
        ``(kl, uniform_kl)``, both ``[B, S]`` fp32, the second detached.
    """
    # A query whose compressed slots are all misses scores every one of them
    # -inf, and log_softmax of such a row is NaN. Masking after the fact is not
    # enough: log_softmax's backward computes `g - softmax * g.sum(-1)` with
    # `softmax = exp(NaN)`, so even the zero gradient such a row receives comes
    # back NaN, and the indexer's backward propagates it because it forms
    # `grad * relu(logits)` and `NaN * 0` is NaN. Neutralise on the way in. This
    # is the common case rather than a corner: the first `ratio - 1` positions of
    # every packed sample have no complete compression group behind them.
    scoreable = torch.isfinite(index_score)
    # Two ways a row has nothing to teach, and both have to leave *both* terms.
    # A row scored entirely -inf is the NaN case above. A row the teacher gave no
    # mass would otherwise contribute 0 to the KL and a full log(n) to the
    # reference -- which is not a student that captured everything, it is a row
    # with nothing to capture, and leaving it in the denominator alone flatters
    # the captured fraction by exactly the rows where the objective did no work.
    nothing_to_learn = ~scoreable.any(-1, keepdim=True) | (target.sum(-1, keepdim=True) <= 0)
    scores = torch.where(nothing_to_learn, 0.0, index_score.float())
    log_q = torch.log_softmax(scores, dim=-1)
    log_target = torch.log(target.clamp_min(torch.finfo(torch.float32).tiny))
    # log_q is -inf exactly where target is 0, and `0 * -inf` is NaN, so the
    # zero-mass slots are masked rather than merely multiplied out.
    contributions = torch.where(target > 0, target * (log_target - log_q), 0.0)
    # The scale to read the KL against: log(n_candidates) - H(target) is what a
    # student would pay knowing the candidate set and nothing about which slot
    # matters. A plateau of 0.021 means one thing against a reference of 0.374
    # and another against 0.02.
    neg_entropy = (target * log_target).sum(-1)
    uniform_kl = torch.where(
        nothing_to_learn.squeeze(-1),
        0.0,
        torch.log(scoreable.sum(-1).clamp_min(1).to(torch.float32)) + neg_entropy,
    )
    return contributions.sum(-1), uniform_kl.detach()


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

    def narrow_queries(self, start: int, end: int) -> CompressionPlan:
        """Keep the full compressed axis, but only the query rows ``[start, end)``.

        CED prefill compresses the encoder on the whole prompt, then replays the
        decoder on the sliding window. The compressor still needs every group;
        the decoder queries only need the visibility of the last ``n_win`` rows.
        """
        return CompressionPlan(
            ratio=self.ratio,
            window_indices=self.window_indices,
            window_positions=self.window_positions,
            range_starts=self.range_starts[start:end],
            range_ends=self.range_ends[start:end],
        )


def sequence_slices_from_positions(position_ids: torch.Tensor) -> tuple[tuple[int, int], ...]:
    """Recover the packed samples' ``[start, end)`` spans from their positions.

    The packed collator restarts ``position_ids`` at zero on each sample, so a
    zero is a boundary and there is nothing else to plumb. Deriving the spans
    here rather than threading them down from the collator keeps the one thing
    that must agree -- which tokens belong to which sample -- reading from a
    single source: the same tensor the causal masks and the sliding window
    already use. A separately-passed span list could disagree with the positions,
    and the model would compress across a boundary while the window did not.

    An unpacked row has one zero, at index 0, and comes back as the single span
    covering it. That is the same construction rather than a special case.
    """
    if position_ids.ndim != 2:
        raise ValueError(f"expected [batch, seq_len] position ids, got {position_ids.ndim} dims")
    seq_len = position_ids.shape[1]
    cached_slices = getattr(position_ids, "_sequence_slices", None)
    if cached_slices is not None and cached_slices and cached_slices[-1][1] == seq_len:
        return cached_slices
    starts = (position_ids[0] == 0).nonzero(as_tuple=True)[0].tolist()
    if not starts or starts[0] != 0:
        raise ValueError(
            "position ids must start a sample at index 0; got a row whose first position is "
            f"{int(position_ids[0, 0])}. Decode-time offsets are not a training input."
        )
    ends = starts[1:] + [seq_len]
    return tuple(zip(starts, ends))


def trailing_partial_group(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    ratio: int,
) -> torch.Tensor | None:
    """Hidden states ``_absorb`` would leave pending after this packed sequence.

    Walks the same packed slices as :func:`build_compression_plans`. A complete
    group contributes no row; each sample's leftover ``length % ratio`` tokens
    are that sample's pending buffer. Decode continues the last sample, so that
    sample's tail is what is returned. A ratio of 0 or 1 never leaves a partial
    group -- 0 does not compress, 1 emits a group on every token.
    """
    if ratio <= 1:
        return None
    pending = None
    for start, end in sequence_slices_from_positions(position_ids):
        remainder = (end - start) % ratio
        pending = hidden_states[:, end - remainder : end] if remainder else None
    return pending


def build_compression_plans(
    config: DeepseekV41LlmConfig,
    position_ids: torch.Tensor,
    sequence_slices: Sequence[tuple[int, int]] | None = None,
) -> dict[int, CompressionPlan]:
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
    slice_begins = torch.tensor([begin for begin, _ in sequence_slices], device=device, dtype=torch.long)
    slice_lengths = torch.tensor([end - begin for begin, end in sequence_slices], device=device, dtype=torch.long)
    plans: dict[int, CompressionPlan] = {}
    for ratio in sorted({r for r in config.compress_ratios if r > 0}):
        group_counts = slice_lengths // ratio
        compressed_offsets = torch.cumsum(group_counts, dim=0) - group_counts
        total_groups = int(sum((end - begin) // ratio for begin, end in sequence_slices))
        if total_groups > 0:
            group_slice_begins = torch.repeat_interleave(slice_begins, group_counts)
            group_compressed_offsets = torch.repeat_interleave(compressed_offsets, group_counts)
            group_idx_in_slice = torch.arange(total_groups, device=device, dtype=torch.long) - group_compressed_offsets
            window_starts = group_slice_begins + group_idx_in_slice * ratio
        else:
            window_starts = torch.empty(0, device=device, dtype=torch.long)
        offsets = torch.arange(ratio, device=device, dtype=torch.long)
        window_indices = window_starts[:, None] + offsets[None, :]

        if seq_len > 0:
            range_starts = torch.repeat_interleave(compressed_offsets.to(torch.int32), slice_lengths)
            max_groups = torch.repeat_interleave(group_counts.to(torch.int32), slice_lengths)
            visible = torch.minimum(((positions + 1) // ratio).to(torch.int32), max_groups)
            range_ends = range_starts + visible
        else:
            range_starts = torch.empty(0, device=device, dtype=torch.int32)
            range_ends = torch.empty(0, device=device, dtype=torch.int32)

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


if _HAS_TRITON:

    @triton.jit
    def _triton_interleaved_rope_inplace_kernel(
        X_ptr,
        Cos_ptr,
        Sin_ptr,
        STRIDE_M,
        H,
        D: tl.constexpr,
        ROPE_HALF: tl.constexpr,
        CONJUGATE: tl.constexpr,
    ):
        m = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1).to(tl.int64)
        p = tl.arange(0, ROPE_HALF)
        c = tl.load(Cos_ptr + m * ROPE_HALF + p).to(tl.float32)
        s = tl.load(Sin_ptr + m * ROPE_HALF + p).to(tl.float32)
        if CONJUGATE:
            s = -s
        base = X_ptr + m * STRIDE_M + h * D + (D - 2 * ROPE_HALF)
        even = tl.load(base + 2 * p).to(tl.float32)
        odd = tl.load(base + 2 * p + 1).to(tl.float32)
        rot_even = (even * c + (-odd) * s).to(X_ptr.dtype.element_ty)
        rot_odd = (odd * c + even * s).to(X_ptr.dtype.element_ty)
        tl.store(base + 2 * p, rot_even)
        tl.store(base + 2 * p + 1, rot_odd)


class _InterleavedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, conjugate: bool) -> torch.Tensor:
        r_half = cos.shape[-1]
        d = x.shape[-1]
        h = x.shape[2] if x.ndim == 4 else 1
        m = x.numel() // (h * d)
        out = x.contiguous().clone()
        cos_c = cos if cos.is_contiguous() else cos.contiguous()
        sin_c = sin if sin.is_contiguous() else sin.contiguous()
        stride_m = h * d
        with get_torch_device().device(x.device):
            _triton_interleaved_rope_inplace_kernel[(m, h)](
                out,
                cos_c,
                sin_c,
                stride_m,
                h,
                D=d,
                ROPE_HALF=r_half,
                CONJUGATE=conjugate,
                num_warps=1,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(cos_c, sin_c)
        ctx.conjugate = conjugate
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        cos_c, sin_c = ctx.saved_tensors
        r_half = cos_c.shape[-1]
        d = grad_output.shape[-1]
        h = grad_output.shape[2] if grad_output.ndim == 4 else 1
        m = grad_output.numel() // (h * d)
        dx = grad_output.contiguous().clone()
        stride_m = h * d
        with get_torch_device().device(grad_output.device):
            _triton_interleaved_rope_inplace_kernel[(m, h)](
                dx,
                cos_c,
                sin_c,
                stride_m,
                h,
                D=d,
                ROPE_HALF=r_half,
                CONJUGATE=not ctx.conjugate,
                num_warps=1,
                enable_fp_fusion=False,
            )
        return dx, None, None, None


def apply_interleaved_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Rotate the trailing ``2 * cos.shape[-1]`` channels of ``x``, leaving NoPE alone.

    ``cos`` / ``sin`` arrive half-sized, one entry per rotated pair, and broadcast
    against ``x``'s leading dimensions. Taking adjacent elements as a complex pair
    makes this the same rotation the reference writes with ``view_as_complex``;
    passing ``-sin`` (or ``conjugate=True``) conjugates it, which is how the
    attention output gets the query's rotation removed so the KV can stay in one
    shared rotated form.
    """
    if (
        IS_CUDA_AVAILABLE
        and _HAS_TRITON
        and x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.ndim in (2, 3, 4)
        and not cos.requires_grad
        and not sin.requires_grad
    ):
        r_half = cos.shape[-1]
        d = x.shape[-1]
        if r_half > 0 and (r_half & (r_half - 1)) == 0 and 2 * r_half <= d:
            h = x.shape[2] if x.ndim == 4 else 1
            m = x.numel() // (h * d)
            if m > 0 and cos.numel() == m * r_half and sin.numel() == m * r_half:
                if torch.is_grad_enabled() and x.requires_grad:
                    return _InterleavedRoPEFunction.apply(x, cos, sin, conjugate)
                can_strided_inplace = (
                    inplace
                    and not torch.is_grad_enabled()
                    and x.stride(-1) == 1
                    and (h == 1 or x.stride(-2) == d)
                    and (x.is_contiguous() or x.ndim == 2 or (x.ndim in (3, 4) and x.shape[0] == 1))
                )
                out = x if can_strided_inplace else x.contiguous().clone()
                if out.is_contiguous() or m == 1:
                    stride_m = h * d
                elif out.ndim == 2:
                    stride_m = out.stride(0)
                else:
                    stride_m = out.stride(1)
                with get_torch_device().device(x.device):
                    _triton_interleaved_rope_inplace_kernel[(m, h)](
                        out,
                        cos if cos.is_contiguous() else cos.contiguous(),
                        sin if sin.is_contiguous() else sin.contiguous(),
                        stride_m,
                        h,
                        D=d,
                        ROPE_HALF=r_half,
                        CONJUGATE=conjugate,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
                return out

    cos = cos.repeat_interleave(2, dim=-1)
    sin = (-sin if conjugate else sin).repeat_interleave(2, dim=-1)
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

    def forward(self, positions: torch.Tensor, compressed: bool) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.norm = DeepseekV4RMSNorm(config.head_dim, eps=config.rms_norm_eps, config=config)
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False) if self.compress_ratio > 1 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        plan: CompressionPlan,
        shard: SequenceShard | None = None,
    ) -> torch.Tensor:
        """``[B, S, hidden]`` -> ``[B, n_groups, head_dim]``, unrotated.

        Under ``shard`` the hidden states are this rank's rows and ``plan``
        indexes the full row: the projections run on the local rows and are
        gathered before the groups are pooled, so the pooling arithmetic is the
        unsharded one and a group may straddle two ranks.
        """
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
            if shard is None:
                return self.norm(self.wkv(hidden_states[:, plan.window_indices[:, 0]]))
            return self.norm(shard.gather(self.wkv(hidden_states))[:, plan.window_indices[:, 0]])

        x = hidden_states.float()
        kv = torch.nn.functional.linear(x, self.wkv.weight.float())
        gate = torch.nn.functional.linear(x, self.wgate.weight.float())
        if shard is not None:
            kv, gate = shard.gather(kv), shard.gather(gate)
        kv, gate = kv[:, plan.window_indices], gate[:, plan.window_indices]
        pooled = (kv * gate.softmax(dim=2)).sum(dim=2)
        return self.norm(pooled.to(hidden_states.dtype))


def select_candidate_blocks(
    scores: torch.Tensor,
    range_ends: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    *,
    return_block_mask: bool = False,
) -> torch.Tensor:
    """Level one of the two-level selection: the best ``topk_blocks`` blocks per query.

    ``scores`` is ``[B, S, C]`` with unreachable entries already at ``-inf``, which
    is what lets a block score of ``-inf`` mean "not reachable yet". Returns a
    bool mask shaped like ``scores`` (or ``[B, S, num_blocks]`` when
    ``return_block_mask=True`` to avoid expanding ``block_size``-times in memory
    on long contexts).
    """
    width = scores.shape[-1]
    pad_w = -width % block_size
    padded = torch.nn.functional.pad(scores, (0, pad_w), value=-torch.inf) if pad_w else scores
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
    if return_block_mask:
        return keep
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _mask_scores_by_candidates(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    block_size: int,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Mask ``scores`` (``[B, S, width]``) by either a full or block-level candidate mask."""
    width = scores.shape[-1]
    if candidates.shape[-1] == width:
        return scores.masked_fill_(~candidates, -torch.inf) if inplace else scores.masked_fill(~candidates, -torch.inf)
    num_blocks = candidates.shape[-1]
    if width == num_blocks * block_size:
        target = scores if inplace else scores.clone()
        target.view(*scores.shape[:-1], num_blocks, block_size).masked_fill_(~candidates.unsqueeze(-1), -torch.inf)
        return target
    expanded = candidates.repeat_interleave(block_size, dim=-1)[..., :width]
    return scores.masked_fill_(~expanded, -torch.inf) if inplace else scores.masked_fill(~expanded, -torch.inf)


def _resolve_kv_round_impl(config: DeepseekV41LlmConfig):
    """Read ``kv_round_implementation``, defaulting unset to ``'none'`` (no-op in training)."""
    impl = resolve_module_op_impl(config, "kv_round_implementation")
    return "none" if impl is None else impl


class _V41TileLangIndexerStudentFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        index_k: torch.Tensor,
        weights_fp32: torch.Tensor,
        topk_idxs: torch.Tensor,
        range_starts: torch.Tensor,
        range_ends: torch.Tensor,
        student_scores: torch.Tensor,
    ) -> torch.Tensor:
        q_sbhd = query.transpose(0, 1).contiguous()
        k_sbd = index_k.transpose(0, 1).contiguous()
        w_sbh = weights_fp32.transpose(0, 1).contiguous()
        ctx.save_for_backward(q_sbhd, k_sbd, w_sbh, topk_idxs, range_starts, range_ends)
        return student_scores

    @staticmethod
    def backward(ctx, grad_student: torch.Tensor):
        from ......ops.kernels.dsa.vendor.tilelang_indexer_bwd import batched_indexer_bwd

        q_sbhd, k_sbd, w_sbh, topk_idxs, range_starts, range_ends = ctx.saved_tensors
        grad_q_sbhd, grad_w_sbh, grad_k_sbd = batched_indexer_bwd(
            q_sbhd,
            w_sbh,
            k_sbd,
            topk_idxs,
            grad_student.contiguous(),
            cu_seqlen_ks=range_starts,
            cu_seqlen_ke=range_ends,
        )
        grad_q = grad_q_sbhd.transpose(0, 1).contiguous()
        grad_k = grad_k_sbd.transpose(0, 1).to(index_k_dtype := q_sbhd.dtype).contiguous()
        grad_w = grad_w_sbh.transpose(0, 1).contiguous()
        del index_k_dtype
        return grad_q, grad_k, grad_w, None, None, None, None


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
        # Named ``head_dim`` (not ``index_head_dim``) so it is the attribute Muon's
        # head-split resolution reads: ``_HEAD_STRIDE_ATTRS`` looks for
        # ``head_dim``/``qk_head_dim``, and this is the only indexer in the repo
        # that spelled its per-head stride differently. Without it, ``wq_b`` stayed
        # on full-matrix Muon and took a ``sqrt(4096/1280)``-larger step than
        # ``self_attn.wq_b`` alongside it.
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5

        # `wq_b` is the only quantized projection in here. `weights_proj` and
        # `wk` are left in the precision the release left them: they do not
        # scale an output slightly, they decide which blocks the sparse
        # attention reads at all.
        self.wq_b = DeepseekV41Linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
            quant_dtype=linear_quant_dtype(config),
            config=config,
            site=f"layers.{layer_idx}.indexer_q_b_proj",
        )
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        if self.owns_keys:
            self.wk = nn.Linear(config.head_dim, config.index_head_dim, bias=False)
            self.k_norm = DeepseekV4RMSNorm(config.index_head_dim, eps=config.rms_norm_eps, config=config)
        self.veomni_kv_round_fp4 = VeomniOp(
            "kv_round",
            "fp4",
            _resolve_kv_round_impl(config),
            site=f"layers.{layer_idx}.indexer",
        )
        self.dsa_indexer_implementation = resolve_module_op_impl(config, "dsa_indexer_implementation")

    def build_keys(
        self,
        latent: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Turn this layer's unrotated latents into index keys, ``[B, C, head_dim]``.

        Has to run before the caller rotates and rounds the latent for the KV
        cache: the reference writes both out of the same storage, and the index
        key is derived from the *unrotated* value.
        """
        # Detach the compressor latent so the auxiliary indexer KL objective
        # trains only the indexer's own projections (DeepSeek-V3.2 §2.1).
        latent = latent.detach()
        keys = self.k_norm(self.wk(latent))
        keys = apply_interleaved_rope(keys, cos, sin)
        return round_index_act(
            keys,
            training=self.training,
            op=self.veomni_kv_round_fp4,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        index_k: torch.Tensor,
        plan: CompressionPlan,
        shared: SharedAttentionState,
        cos: torch.Tensor,
        sin: torch.Tensor,
        return_scores: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``[B, S, index_topk]`` int32 group indices, ``-1`` where nothing was reachable.

        With ``return_scores``, also the student distribution the indexer loss
        trains: this layer's own score at each selected slot.
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        compressed_len = plan.compressed_len

        # Detach trunk activations so the auxiliary indexer KL objective moves
        # only the indexer parameters and never leaks into the LM trunk
        # (DeepSeek-V3.2 §2.1, matching DeepseekV4Indexer.forward).
        hidden_states = hidden_states.detach()
        q_residual = q_residual.detach()

        recompute_cache = getattr(self, "_recompute_indexer_cache", None)
        if recompute_cache is not None and not return_scores:
            topk_idxs, _, cached_candidates = recompute_cache
            self._recompute_indexer_cache = None
            if self.is_candidate_source:
                shared.candidates = cached_candidates
            return topk_idxs, None

        with torch.set_grad_enabled(return_scores and torch.is_grad_enabled()):
            query = self.wq_b(q_residual).unflatten(-1, (self.n_heads, self.head_dim))
            query = apply_interleaved_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2))
            query = round_index_act(
                query,
                training=self.training,
                op=self.veomni_kv_round_fp4,
            )
            weights = self.weights_proj(hidden_states) * (self.softmax_scale * self.n_heads**-0.5)

        topk = min(self.index_topk, compressed_len)
        use_tl_indexer = (
            self.dsa_indexer_implementation == "tilelang"
            and IS_CUDA_AVAILABLE
            and query.is_cuda
            and query.dtype == torch.bfloat16
            and index_k.dtype == torch.bfloat16
            and compressed_len > 0
            and self.n_heads <= 64
            and self.n_heads % 8 == 0
            and self.head_dim >= 32
        )
        if use_tl_indexer:
            weights_fp32 = weights.float()
            if recompute_cache is not None:
                topk_idxs, student_scores, cached_candidates = recompute_cache
                self._recompute_indexer_cache = None
                if self.is_candidate_source:
                    shared.candidates = cached_candidates
            else:
                with torch.no_grad():
                    q_sbhd = query.detach().transpose(0, 1).contiguous()
                    k_sbd = index_k.detach().transpose(0, 1).contiguous()
                    w_sbh = weights_fp32.detach().transpose(0, 1).contiguous()
                    if self.uses_candidates and shared.candidates is None:
                        raise RuntimeError(
                            f"Layer {self.layer_idx} indexes inside the candidate blocks but none were published; "
                            "candidate_source_layer_id must name a layer in index_source_layer_ids"
                        )
                    tl_chunk = 1024 if compressed_len > 32768 else seq_len
                    if seq_len <= tl_chunk:
                        scores = torch.ops.DSA.lighting_indexer_fwd(
                            q_sbhd, k_sbd, w_sbh, plan.range_starts, plan.range_ends
                        )
                        ranked = scores
                        if self.is_candidate_source:
                            shared.candidates = select_candidate_blocks(
                                scores,
                                plan.range_ends,
                                self.candidate_topk_blocks,
                                self.candidate_block_size,
                                return_block_mask=True,
                            )
                        elif self.uses_candidates:
                            assert shared.candidates is not None
                            ranked = _mask_scores_by_candidates(
                                scores,
                                shared.candidates,
                                self.candidate_block_size,
                                inplace=not return_scores,
                            )

                        selected = ranked.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
                        reachable = (selected >= plan.range_starts.view(1, -1, 1)) & (
                            selected < plan.range_ends.view(1, -1, 1)
                        )
                        topk_idxs = selected.to(torch.int32).masked_fill(~reachable, -1)
                        student_scores = (
                            scores.gather(-1, selected).masked_fill(~reachable, -torch.inf) if return_scores else None
                        )
                        del scores, ranked
                    else:
                        topk_idxs = torch.empty((batch, seq_len, topk), dtype=torch.int32, device=device)
                        student_scores = (
                            torch.empty((batch, seq_len, topk), dtype=torch.float32, device=device)
                            if return_scores
                            else None
                        )
                        num_blocks = (compressed_len + self.candidate_block_size - 1) // self.candidate_block_size
                        candidates_buf = (
                            torch.empty((batch, seq_len, num_blocks), dtype=torch.bool, device=device)
                            if self.is_candidate_source
                            else None
                        )
                        for begin in range(0, seq_len, tl_chunk):
                            end = min(begin + tl_chunk, seq_len)
                            rs_chunk = plan.range_starts[begin:end]
                            re_chunk = plan.range_ends[begin:end]
                            scores_chunk = torch.ops.DSA.lighting_indexer_fwd(
                                q_sbhd[begin:end], k_sbd, w_sbh[begin:end], rs_chunk, re_chunk
                            )
                            ranked_chunk = scores_chunk
                            if self.is_candidate_source:
                                assert candidates_buf is not None
                                candidates_buf[:, begin:end] = select_candidate_blocks(
                                    scores_chunk,
                                    re_chunk,
                                    self.candidate_topk_blocks,
                                    self.candidate_block_size,
                                    return_block_mask=True,
                                )
                            elif self.uses_candidates:
                                assert shared.candidates is not None
                                ranked_chunk = _mask_scores_by_candidates(
                                    scores_chunk,
                                    shared.candidates[:, begin:end],
                                    self.candidate_block_size,
                                    inplace=not return_scores,
                                )
                            sel_chunk = ranked_chunk.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
                            reach_chunk = (sel_chunk >= rs_chunk.view(1, -1, 1)) & (
                                sel_chunk < re_chunk.view(1, -1, 1)
                            )
                            topk_idxs[:, begin:end] = sel_chunk.to(torch.int32).masked_fill(~reach_chunk, -1)
                            if student_scores is not None:
                                student_scores[:, begin:end] = scores_chunk.gather(-1, sel_chunk).masked_fill(
                                    ~reach_chunk, -torch.inf
                                )
                            del scores_chunk, ranked_chunk, sel_chunk, reach_chunk
                        if self.is_candidate_source:
                            shared.candidates = candidates_buf
                if getattr(self, "_enable_recompute_cache", False):
                    self._recompute_indexer_cache = (
                        topk_idxs,
                        student_scores,
                        shared.candidates if self.is_candidate_source else None,
                    )
            if not return_scores:
                return topk_idxs, None
            student = _V41TileLangIndexerStudentFunction.apply(
                query,
                index_k,
                weights_fp32,
                topk_idxs,
                plan.range_starts,
                plan.range_ends,
                student_scores,
            )
            return topk_idxs, student

        visible = plan.visibility_mask(device)
        with torch.no_grad():
            scores = torch.empty(batch, seq_len, compressed_len, device=device, dtype=torch.float32)
            for begin in range(0, seq_len, INDEXER_QUERY_CHUNK):
                end = min(begin + INDEXER_QUERY_CHUNK, seq_len)
                chunk = torch.einsum("bshd,btd->bsht", query[:, begin:end], index_k)
                chunk = (chunk.float().relu() * weights[:, begin:end].float().unsqueeze(-1)).sum(dim=2)
                scores[:, begin:end] = chunk.masked_fill(~visible[begin:end], -torch.inf)

            ranked = scores
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
                ranked = scores.masked_fill(~shared.candidates, -torch.inf)

            selected = ranked.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
            reachable = visible.unsqueeze(0).expand(batch, -1, -1).gather(-1, selected)
            topk_idxs = selected.to(torch.int32).masked_fill(~reachable, -1)
            del scores, ranked
        if not return_scores:
            return topk_idxs, None
        # Differentiable student scores only at the `topk` selected compressed slots
        # (`[B, S, topk]`), avoiding retaining `[B, S, H, T_c]` autograd activations.
        student_chunks = []
        for begin in range(0, seq_len, INDEXER_QUERY_CHUNK):
            end = min(begin + INDEXER_QUERY_CHUNK, seq_len)
            sel_chunk = selected[:, begin:end]
            k_sel = index_k.gather(1, sel_chunk.reshape(batch, -1, 1).expand(-1, -1, index_k.size(-1))).unflatten(
                1, (end - begin, topk)
            )
            q_chunk = query[:, begin:end]
            w_chunk = weights[:, begin:end].float().unsqueeze(-1)
            s_chunk = (torch.einsum("bshd,bskd->bshk", q_chunk, k_sel).float().relu() * w_chunk).sum(dim=2)
            student_chunks.append(s_chunk)
        student = torch.cat(student_chunks, dim=1).masked_fill(~reachable, -torch.inf)
        return topk_idxs, student


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
        quant_dtype = linear_quant_dtype(config)
        self.wq_a = DeepseekV41Linear(
            config.hidden_size,
            config.q_lora_rank,
            bias=False,
            quant_dtype=quant_dtype,
            config=config,
            site=f"layers.{layer_idx}.q_a_proj",
        )
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps, config=config)
        self.q_norm.fuse_fp8_quant = quant_dtype is not None
        self.wq_b = DeepseekV41Linear(
            config.q_lora_rank,
            self.n_heads * config.head_dim,
            bias=False,
            quant_dtype=quant_dtype,
            config=config,
            site=f"layers.{layer_idx}.q_b_proj",
        )
        self.wkv = DeepseekV41Linear(
            config.hidden_size,
            config.head_dim,
            bias=False,
            quant_dtype=quant_dtype,
            config=config,
            site=f"layers.{layer_idx}.kv_proj",
        )
        self.kv_norm = DeepseekV4RMSNorm(config.head_dim, eps=config.rms_norm_eps, config=config)
        self.wo_a = DeepseekV41GroupedLinear(
            self.n_heads * config.head_dim // self.n_groups,
            self.n_groups * config.o_lora_rank,
            n_groups=self.n_groups,
            quant_dtype=quant_dtype,
            config=config,
            site=f"layers.{layer_idx}.o_a_proj",
        )
        self.wo_b = DeepseekV41Linear(
            self.n_groups * config.o_lora_rank,
            config.hidden_size,
            bias=False,
            quant_dtype=quant_dtype,
            config=config,
            site=f"layers.{layer_idx}.o_b_proj",
        )

        self.compressor = DeepseekV41Compressor(config, layer_idx) if self.is_kv_source else None
        self.indexer = DeepseekV41Indexer(config, layer_idx) if self.is_index_source else None
        self.compress_kv_dtype = getattr(config, "compress_kv_dtype", "fp4")
        kv_round_impl = _resolve_kv_round_impl(config)
        self.veomni_kv_round_fp8 = VeomniOp(
            "kv_round", "standard", kv_round_impl, site=f"layers.{layer_idx}.window_kv"
        )
        self.veomni_kv_round_compress_fp8 = VeomniOp(
            "kv_round", "standard", kv_round_impl, site=f"layers.{layer_idx}.compress_kv"
        )
        self.veomni_kv_round_fp4 = VeomniOp("kv_round", "fp4", kv_round_impl, site=f"layers.{layer_idx}.compress_kv")

    @property
    def rope_is_compressed(self) -> bool:
        """Compressed layers use the YaRN schedule; pure sliding-window layers do not."""
        return self.compress_ratio > 0

    @property
    def builds_indexer_loss(self) -> bool:
        """Whether this layer produces an indexer KL this forward.

        Only a layer owning an indexer has a student to train. A non-positive
        coefficient counts as off rather than as a zero-weighted term: ``loss +
        0.0 * kl`` is the right value but still builds the graph, so the backward
        writes a zero ``grad`` onto every indexer parameter -- and an optimizer
        that decays whatever it steps would then apply weight decay to parameters
        the objective deliberately left alone, at the full price of the teacher
        kernel.
        """
        if self.indexer is None or not getattr(self.config, "dsa_indexer_loss", False):
            return False
        return getattr(self.config, "dsa_indexer_loss_coef", 1.0) > 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        shared: SharedAttentionState,
        plans: dict[int, CompressionPlan],
        shard: SequenceShard | None = None,
    ) -> tuple[torch.Tensor, IndexerLossTerms | None]:
        """``[B, S, hidden]``, and this layer's indexer KL when it builds one.

        The second value is ``None`` for every layer that owns no indexer and for
        every forward with the objective off, but the arity does not change --
        the alternative is three call sites each re-deriving which layers return
        what, and a stale copy in any one of them is an unpacking error.

        ``shard`` is sequence parallelism: ``hidden_states`` and ``position_ids``
        are this rank's rows, ``plans`` cover the full row, and the queries
        attend over the gathered KV. See :mod:`.sequence_shard`.
        """
        batch, seq_len, _ = hidden_states.shape
        cached_rope = shared.compress_rope_cos_sin if self.rope_is_compressed else shared.main_rope_cos_sin
        if cached_rope is not None and cached_rope[0].shape[1] == position_ids.shape[1]:
            cos, sin = cached_rope
        else:
            cos, sin = rotary_emb(position_ids, compressed=self.rope_is_compressed)
        can_inplace = not torch.is_grad_enabled()

        wq_a_out, wkv_out = quantized_linear_pair(hidden_states, self.wq_a, self.wkv)
        q_residual = self.q_norm(wq_a_out)
        query = self.wq_b(q_residual).unflatten(-1, (self.n_heads, self.head_dim))
        query = apply_interleaved_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2), inplace=can_inplace)

        kv = self.kv_norm(wkv_out)
        kv = apply_interleaved_rope(kv, cos, sin, inplace=can_inplace)
        kv = round_window_kv(
            kv,
            training=self.training,
            op=self.veomni_kv_round_fp8,
        )

        student = None
        if shard is None:
            if shared.window_indices is not None and shared.window_indices.shape[1] == seq_len:
                topk_idxs = shared.window_indices
            else:
                topk_idxs = self._window_indices(position_ids, seq_len)
        else:
            win_start = max(0, shard.start - self.sliding_window + 1)
            kv = shard.gather_kv(kv)[:, win_start : shard.end]
            if shared.window_indices is not None and shared.window_indices.shape[1] == seq_len:
                topk_idxs = shared.window_indices
            else:
                topk_idxs = self._local_window_indices(shard)
        if self.compress_ratio:
            compress_kv, compress_idxs, student = self._compressed_kv(
                hidden_states,
                q_residual,
                position_ids,
                rotary_emb,
                shared,
                plans,
                offset=kv.shape[1],
                shard=shard,
            )
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        query = query.contiguous()
        kv = kv.contiguous()
        topk_idxs = topk_idxs.contiguous()
        sink = self.attn_sink.float().contiguous()

        indexer_loss = None
        if student is None:
            output = sparse_attn_tilelang(query, kv, sink, topk_idxs, self.softmax_scale)
        else:
            output, lse = sparse_attn_tilelang(query, kv, sink, topk_idxs, self.softmax_scale, return_lse=True)
            valid_rows = None if shard is None else shard.valid_rows(query.device)
            indexer_loss = self._indexer_loss(query, kv, topk_idxs, lse, student, valid_rows)

        # The KV is cached in one shared rotated form, so the query's rotation is
        # removed from the output rather than never applied.
        output = apply_interleaved_rope(
            output, cos.unsqueeze(-2), sin.unsqueeze(-2), inplace=can_inplace, conjugate=True
        )
        grouped = self.wo_a(output.reshape(batch, seq_len, self.n_groups, -1))
        return self.wo_b(grouped.flatten(2)), indexer_loss

    def _indexer_loss(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
        lse: torch.Tensor,
        student: torch.Tensor,
        valid_rows: torch.Tensor | None = None,
    ) -> IndexerLossTerms:
        """Score the indexer against the attention it feeds.

        The teacher is the attention's own mass on each selected compressed slot,
        read off the log-sum-exp the forward already computed. That LSE is the
        true denominator only because ``topk_idxs`` spans the sliding window as
        well as the compressed entries and the kernel folds the sink into the
        same sum: a second forward over the compressed slice alone would produce
        a plausible, decreasing loss that trains the indexer toward the wrong
        distribution.

        The compressed entries are the *trailing* range of the index tensor,
        because the forward above concatenates window then compressed. The width
        is read off the student so that slot ``j`` of the teacher pairs with
        ``student[..., j]``, and it is checked rather than asserted -- ``python
        -O`` strips an assert, and this is the only thing between the teacher's
        slice and the sliding-window slots.

        In V4.1 the indexer is shared, so this trains it on the *source* layer's
        attention; the layers reusing the selection have their own distributions
        and no say. That is inherent to one indexer serving several layers.

        ``valid_rows`` drops sequence-parallel padding rows: the pad is its own
        packed sample and has compressed groups to attend to, so it would
        otherwise contribute a KL of its own.
        """
        width = student.shape[-1]
        if width > topk_idxs.shape[-1]:
            raise RuntimeError(
                f"the indexer scored {width} slots but the attention selected {topk_idxs.shape[-1]} in total; "
                "the compressed slice cannot be wider than the concatenation holding it"
            )
        target = sparse_mqa_target_fwd(query, kv, topk_idxs[:, :, -width:].contiguous(), lse, self.softmax_scale)
        # A row the teacher gave no mass goes out as exactly zero rather than as
        # `0 / tiny`. Dividing by the clamp raises the denominator instead of the
        # numerator, so a row whose mass is denormal rather than zero comes back
        # summing to something in (0, 1) -- neither a distribution nor an absence
        # of one, and the KL would weight it as though it were the former.
        mass = target.sum(-1, keepdim=True)
        tiny = torch.finfo(torch.float32).tiny
        target = torch.where(mass > tiny, target / mass.clamp_min(tiny), 0.0)
        kl, uniform = indexer_kl_terms(student, target)
        if valid_rows is not None:
            kl = torch.where(valid_rows, kl, 0.0)
            uniform = torch.where(valid_rows, uniform, 0.0)
        return IndexerLossTerms(kl=kl.sum(), uniform=uniform.sum())

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

    def _local_window_indices(self, shard: SequenceShard) -> torch.Tensor:
        """``[B, local_len, min(window, full_len)]`` int32 relative to ``kv[:, win_start : shard.end]``."""
        device = shard.position_ids.device
        width = min(self.sliding_window, shard.full_len)
        win_start = max(0, shard.start - self.sliding_window + 1)
        local_pos = shard.local_rows(shard.position_ids)
        queries = torch.arange(shard.start - win_start, shard.end - win_start, device=device, dtype=torch.int32)
        offsets = torch.arange(width - 1, -1, -1, device=device, dtype=torch.int32)
        indices = (queries[None, :, None] - offsets[None, None, :]).expand(local_pos.shape[0], -1, -1)
        return indices.masked_fill(offsets[None, None, :] > local_pos.to(torch.int32)[:, :, None], -1)

    def _compressed_kv(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        shared: SharedAttentionState,
        plans: dict[int, CompressionPlan],
        offset: int,
        shard: SequenceShard | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """The shared compressed KV and this layer's selection into it.

        The ordering here is the one invariant worth stating: a source layer's
        indexer consumes the *unrotated* latent, so the index keys are built
        before the latent is rotated and rounded for the KV side. The reference
        gets this by writing both out of the same buffer in a fixed order; doing
        it out of order would train the indexer against keys the deployed model
        never produces.

        Under ``shard`` the compressed KV and index keys cover the full row and
        are identical on every rank, while the top-k selection and the
        candidate mask stay on this rank's query rows: only this rank's
        queries ever read them.
        """
        plan = plans[self.compress_ratio]

        if self.is_kv_source:
            latent = self.compressor(hidden_states, plan, shard)
            group_cos, group_sin = rotary_emb(plan.window_positions, compressed=True)
            if self.indexer is not None:
                shared.index_k = self.indexer.build_keys(latent, group_cos, group_sin)
            rotated = apply_interleaved_rope(latent, group_cos, group_sin)
            shared.compress_kv = round_compress_kv(
                rotated,
                self.compress_kv_dtype,
                training=self.training,
                op_fp4=self.veomni_kv_round_fp4,
                op_fp8=self.veomni_kv_round_compress_fp8,
            )

        if shared.compress_kv is None:
            raise RuntimeError(
                f"Layer {self.layer_idx} attends over compressed KV but no source layer has published any; "
                "kv_source_layer_ids must contain a layer at or before it"
            )

        student: torch.Tensor | None = None
        if self.indexer is not None:
            if shared.index_k is None:
                raise RuntimeError(
                    f"Layer {self.layer_idx} runs an indexer but no index keys have been published; "
                    "every index source must be at or after a kv source"
                )
            if (
                shared.compress_rope_cos_sin is not None
                and shared.compress_rope_cos_sin[0].shape[1] == position_ids.shape[1]
            ):
                cos, sin = shared.compress_rope_cos_sin
            else:
                cos, sin = rotary_emb(position_ids, compressed=True)
            shared.topk_idxs, student = self.indexer(
                hidden_states,
                q_residual,
                shared.index_k,
                plan if shard is None else plan.narrow_queries(shard.start, shard.end),
                shared,
                cos,
                sin,
                return_scores=self.builds_indexer_loss,
            )
        elif shared.topk_idxs is None:
            raise RuntimeError(
                f"Layer {self.layer_idx} reuses a top-k selection but none has been published; "
                "index_source_layer_ids must contain a layer at or before it"
            )

        selection = torch.where(shared.topk_idxs >= 0, shared.topk_idxs + offset, -1)
        return shared.compress_kv, selection, student


__all__ = [
    "CompressionPlan",
    "DeepseekV41Attention",
    "DeepseekV41Compressor",
    "DeepseekV41Indexer",
    "DeepseekV41RotaryEmbedding",
    "IndexerLossTerms",
    "SharedAttentionState",
    "apply_interleaved_rope",
    "build_compression_plans",
    "indexer_kl_terms",
    "select_candidate_blocks",
    "sequence_slices_from_positions",
    "trailing_partial_group",
]
