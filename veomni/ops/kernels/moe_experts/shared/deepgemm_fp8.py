# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Hopper/Blackwell FP8 preparation for DeepGEMM grouped linear training.

Activations use 1x128 scales for forward/dgrad and independent token-axis
128x1 scales for wgrad. Weights use 128x128 scales. Quantization always reads
the BF16 source; transposed weights reuse the same quantized square tiles.
This module is imported lazily by the CUDA-only backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


# SM90 K-grouped GEMM requires FP32 output. Bound its temporary independently
# of the local expert count; a single expert is the minimum indivisible unit.
_WGRAD_WORKSPACE_BYTES = 256 * 1024**2
_PROCESS_DEVICE: int | None = None


def _require_process_device(device: torch.device) -> None:
    # The pinned DeepGEMM JIT caches CUfunction handles without a device key.
    # A handle loaded on another GPU fails even when both GPUs have the same CC.
    global _PROCESS_DEVICE
    index = device.index if device.index is not None else torch.cuda.current_device()
    if _PROCESS_DEVICE is None:
        _PROCESS_DEVICE = index
    elif _PROCESS_DEVICE != index:
        raise RuntimeError(
            "DeepGEMM FP8 MoE requires one CUDA device per process; "
            f"this process already uses cuda:{_PROCESS_DEVICE}, but received cuda:{index}. "
            "Use one distributed worker per GPU."
        )


def _uses_ue8m0(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (10, 0)


@triton.jit
def _quant_scale(amax, USE_UE8M0: tl.constexpr):
    scale = tl.maximum(amax, 1.0e-4) / 448.0
    if USE_UE8M0:
        # Match DeepGEMM ceil_to_ue8m0 before quantizing the payload. Rounding
        # only the GEMM scale would change the values represented by that payload.
        bits = scale.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (tl.minimum(tl.maximum(exponent, 1), 254) << 23).to(tl.float32, bitcast=True)
    return scale


@triton.jit
def _layout_kernel(Meta, Rows, Unpad, Groups, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    start = tl.load(Meta + block * 6)
    count = tl.load(Meta + block * 6 + 1)
    expert = tl.load(Meta + block * 6 + 2)
    valid = r < count
    row = block * BLOCK + r
    tl.store(Rows + row, tl.where(valid, start + r, -1))
    tl.store(Groups + row, tl.where(valid, expert, -1))
    tl.store(Unpad + start + r, row, mask=valid)


@dataclass
class GroupedLayout:
    """Per-expert aligned padding and host K sizes for grouped GEMM launches.

    ``from_counts`` performs one bounded host read when counts reside on CUDA.
    The SM90 K-grouped API requires these host sizes; CUDA graph capture is not
    supported. The resulting metadata is reused by both expert projections and
    all their backward GEMMs.
    """

    counts: tuple[int, ...]
    padded_counts: list[int]
    metadata: torch.Tensor
    rows: torch.Tensor
    unpad: torch.Tensor
    groups: torch.Tensor
    ks: torch.Tensor
    use_ue8m0: bool

    @classmethod
    def from_counts(cls, counts: torch.Tensor, device: torch.device) -> GroupedLayout:
        """Create group-local padding without mixing tokens from different experts."""
        logical = tuple(int(n) for n in counts.tolist())
        if any(n < 0 for n in logical):
            raise ValueError("Expert token counts must be nonnegative.")
        import deep_gemm

        _require_process_device(device)
        with torch.cuda.device(device):
            alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
        if alignment < 128 or alignment % 128:
            raise ValueError("Blockwise FP8 MoE requires a DeepGEMM group alignment divisible by 128.")
        padded = [triton.cdiv(n, alignment) * alignment for n in logical]
        blocks = []
        source, target = 0, 0
        for expert, (n, p) in enumerate(zip(logical, padded)):
            for offset in range(0, p, 128):
                blocks.append([source + offset, max(0, min(128, n - offset)), expert, target, p, offset])
            source += n
            target += p
        metadata = torch.tensor(blocks, dtype=torch.int32, device=device).reshape(-1, 6)
        rows = torch.empty(target, dtype=torch.int32, device=device)
        unpad = torch.empty(source, dtype=torch.int64, device=device)
        groups = torch.empty_like(rows)
        ks = torch.tensor(padded, dtype=torch.int32, device=device)
        if target:
            with torch.cuda.device(device):
                _layout_kernel[(len(blocks),)](metadata, rows, unpad, groups, BLOCK=128)
        return cls(logical, padded, metadata, rows, unpad, groups, ks, _uses_ue8m0(device))


@triton.jit
def _row_quant_kernel(
    X,
    Q,
    S,
    Rows,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    H: tl.constexpr,
    P,
    BM: tl.constexpr,
    BK: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    # Long-context routed payloads can exceed 2^31 elements on one Hopper GPU.
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    cols = tl.program_id(1) * BK + tl.arange(0, BK)
    source = tl.load(Rows + rows, mask=rows < P, other=-1).to(tl.int64)
    x = tl.load(
        X + source[:, None] * SX0 + cols[None, :] * SX1, mask=(source[:, None] >= 0) & (cols[None, :] < H), other=0
    ).to(tl.float32)
    scale = _quant_scale(tl.max(tl.abs(x), 1), USE_UE8M0)
    tl.store(Q + rows[:, None] * H + cols[None, :], x / scale[:, None], mask=(rows[:, None] < P) & (cols[None, :] < H))
    tl.store(S + rows + tl.program_id(1) * P, scale, mask=rows < P)


def quantize_rows(x: torch.Tensor, layout: GroupedLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 rows with FP32 scales, rounded to powers of two on SM100."""
    p, h = layout.rows.numel(), x.shape[1]
    q = torch.empty((p, h), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty_strided((p, h // 128), (1, p), dtype=torch.float32, device=x.device)
    if p:
        with torch.cuda.device(x.device):
            _row_quant_kernel[(triton.cdiv(p, 16), h // 128)](
                x,
                q,
                scales,
                layout.rows,
                x.stride(0),
                x.stride(1),
                h,
                p,
                16,
                128,
                layout.use_ue8m0,
            )
    return q, scales


@triton.jit
def _weight_quant_kernel(
    W,
    Q,
    QT,
    S,
    ST,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    STORE_NORMAL: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    mb, kb, expert = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = mb * BLOCK + tl.arange(0, BLOCK)
    k = kb * BLOCK + tl.arange(0, BLOCK)
    # The expert base is always int64. Only widen intra-expert offsets when
    # their compile-time bound requires it, preserving the small-tile layout.
    if N * K > 2147483648:
        m = m.to(tl.int64)
        k = k.to(tl.int64)
    offsets = expert.to(tl.int64) * N * K + m[:, None] * K + k[None, :]
    w = tl.load(W + offsets).to(tl.float32)
    if ROUND_BF16:
        w = w.to(tl.bfloat16).to(tl.float32)
    scale = _quant_scale(tl.max(tl.max(tl.abs(w), 1), 0), USE_UE8M0)
    if STORE_NORMAL:
        tl.store(Q + offsets, w / scale)
        tl.store(S + (expert * (N // BLOCK) + mb) * (K // BLOCK) + kb, scale)
    if STORE_TRANSPOSE:
        transposed = expert.to(tl.int64) * N * K + k[None, :] * N + m[:, None]
        tl.store(QT + transposed, w / scale)
        tl.store(ST + (expert * (K // BLOCK) + kb) * (N // BLOCK) + mb, scale)


def quantize_weights(
    w: torch.Tensor,
    *,
    use_ue8m0: bool | None = None,
    store_normal: bool = True,
    store_transpose: bool = True,
    round_bf16: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Quantize 128x128 weight tiles into the requested GEMM orientations.

    ``round_bf16`` preserves the FSDP BF16-cast recipe when the source is an
    FP32 master shard, without materializing a separate BF16 temporary.
    """
    if not (store_normal or store_transpose):
        raise ValueError("At least one quantized weight orientation must be requested.")
    e, n, k = w.shape
    q = torch.empty_like(w, dtype=torch.float8_e4m3fn) if store_normal else None
    qt = torch.empty((e, k, n), dtype=torch.float8_e4m3fn, device=w.device) if store_transpose else None
    scales = torch.empty((e, n // 128, k // 128), dtype=torch.float32, device=w.device) if store_normal else None
    scales_t = torch.empty((e, k // 128, n // 128), dtype=torch.float32, device=w.device) if store_transpose else None
    if use_ue8m0 is None:
        use_ue8m0 = _uses_ue8m0(w.device)
    with torch.cuda.device(w.device):
        _weight_quant_kernel[(n // 128, k // 128, e)](
            w, q, qt, scales, scales_t, n, k, 128, use_ue8m0, store_normal, store_transpose, round_bf16, num_warps=8
        )
    return q, scales, qt, scales_t


@triton.jit
def _weight_transpose_kernel(Q, QT, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    nb, kb, expert = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = nb * BLOCK + tl.arange(0, BLOCK)
    k = kb * BLOCK + tl.arange(0, BLOCK)
    base = expert.to(tl.int64) * N * K
    if N * K > 2147483648:
        n = n.to(tl.int64)
        k = k.to(tl.int64)
    values = tl.load(Q + base + n[:, None] * K + k[None, :])
    tl.store(QT + base + k[None, :] * N + n[:, None], values)


def transpose_fp8_weight(qdata: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose square-block FP8 weights without dequantization or requantization."""
    e, n, k = qdata.shape
    qt = torch.empty((e, k, n), device=qdata.device, dtype=qdata.dtype)
    with torch.cuda.device(qdata.device):
        _weight_transpose_kernel[(n // 128, k // 128, e)](
            qdata.view(torch.uint8), qt.view(torch.uint8), n, k, 128, num_warps=8
        )
    return qt, scale.transpose(1, 2).contiguous()


@triton.jit
def _column_quant_kernel(
    X,
    Q,
    S,
    Meta,
    H: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    block, cb = tl.program_id(0), tl.program_id(1)
    source = tl.load(Meta + block * 6).to(tl.int64)
    count = tl.load(Meta + block * 6 + 1)
    group_start = tl.load(Meta + block * 6 + 3).to(tl.int64)
    group_size = tl.load(Meta + block * 6 + 4)
    group_offset = tl.load(Meta + block * 6 + 5)
    r = tl.arange(0, BK)
    c = cb * BN + tl.arange(0, BN)
    x = tl.load(
        X + (source + r[:, None]) * SX0 + c[None, :] * SX1, mask=(r[:, None] < count) & (c[None, :] < H), other=0
    ).to(tl.float32)
    scale = _quant_scale(tl.max(tl.abs(x), 0), USE_UE8M0)
    if USE_UE8M0:
        offsets = (group_start + group_offset + r[:, None]) * H + c[None, :]
    else:
        offsets = group_start * H + c[None, :].to(tl.int64) * group_size + group_offset + r[:, None]
    tl.store(Q + offsets, x / scale[None, :], mask=c[None, :] < H)
    tl.store(S + block.to(tl.int64) * H + c, scale, mask=c < H)


def quantize_columns(x: torch.Tensor, layout: GroupedLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize along tokens into the device's native K-grouped wgrad layout.

    The flat FP8 payload concatenates ``[H,padded_tokens_e]`` matrices. Scales
    have logical shape ``[H,sum(padded_tokens_e)/128]`` with MN-major strides.
    Every padding value is zero and every padding scale is finite and nonzero.
    On SM100 the payload is contiguous ``[sum(padded_tokens_e),H]`` for TN,
    with contiguous ``[sum(padded_tokens_e)/128,H]`` power-of-two scales.
    """
    p, h = layout.rows.numel(), x.shape[1]
    q = torch.empty(p * h, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty_strided((h, p // 128), (1, h), dtype=torch.float32, device=x.device)
    if p:
        with torch.cuda.device(x.device):
            _column_quant_kernel[(p // 128, triton.cdiv(h, 32))](
                x,
                q,
                scales,
                layout.metadata,
                h,
                x.stride(0),
                x.stride(1),
                128,
                32,
                layout.use_ue8m0,
            )
    if layout.use_ue8m0:
        return q.view(p, h), scales.T
    return q, scales


def _weight_gradient(x, grad_output, layout, shape, dtype):
    return _weight_gradient_quantized_input(*quantize_columns(x, layout), grad_output, layout, shape, dtype)


def _weight_gradient_quantized_input(qb, sb, grad_output, layout, shape, dtype):
    """Write SM100 gradients directly; bound the mandatory SM90 FP32 workspace.

    Chunk only the expert dimension, never an expert's token reduction. The
    independently quantized column operands are shared by all chunks as views.
    SM100 accumulates internally in FP32 and casts once in the GEMM epilogue.
    """
    import deep_gemm

    experts, n, k = shape
    dw = torch.empty(shape, device=grad_output.device, dtype=dtype)
    if not layout.rows.numel():
        return dw.zero_()
    qa, sa = quantize_columns(grad_output, layout)
    if layout.use_ue8m0:
        # The SM100 scale-packing kernel supports at most 128 groups per call.
        begin = 0
        for expert in range(0, experts, 128):
            end_expert = min(expert + 128, experts)
            ks = layout.padded_counts[expert:end_expert]
            end = begin + sum(ks)
            output = dw[expert:end_expert]
            if end > begin:
                deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
                    (qa[begin:end], sa[begin // 128 : end // 128]),
                    (qb[begin:end], sb[begin // 128 : end // 128]),
                    output,
                    ks,
                    layout.ks[expert:end_expert],
                    None,
                    recipe=(1, 1, 128),
                    use_psum_layout=False,
                )
            else:
                output.zero_()
            begin = end
        return dw

    chunk_size = min(experts, max(1, _WGRAD_WORKSPACE_BYTES // (n * k * 4)))
    workspace = torch.empty((chunk_size, n, k), device=grad_output.device, dtype=torch.float32)
    begin = 0
    for expert in range(0, experts, chunk_size):
        end_expert = min(expert + chunk_size, experts)
        ks = layout.padded_counts[expert:end_expert]
        end = begin + sum(ks)
        accumulator = workspace[: end_expert - expert]
        accumulator.zero_()
        if end > begin:
            deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
                (qa[begin * n : end * n], sa[:, begin // 128 : end // 128]),
                (qb[begin * k : end * k], sb[:, begin // 128 : end // 128]),
                accumulator,
                ks,
                layout.ks[expert:end_expert],
                accumulator,
                recipe=(1, 1, 128),
                use_psum_layout=False,
            )
        dw[expert:end_expert].copy_(accumulator)
        begin = end
    return dw


@triton.jit
def _swiglu_kernel(
    X, Y, DY, DX, SIZE, I: tl.constexpr, LIMIT: tl.constexpr, BACKWARD: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    source = offsets // I * (2 * I) + offsets % I
    gate = tl.load(X + source, offsets < SIZE, 0).to(tl.float32)
    up = tl.load(X + source + I, offsets < SIZE, 0).to(tl.float32)
    gc, uc = gate, up
    if LIMIT is not None:
        limit = tl.full((), LIMIT, tl.float32).to(X.dtype.element_ty).to(tl.float32)
        gc = tl.minimum(gate, limit)
        uc = tl.minimum(tl.maximum(up, -limit), limit)
    denominator = 1.0 + libdevice.exp(-gc)
    # Eager SiLU materializes BF16 before multiplication. Keep that rounding.
    silu = tl.div_rn(gc, denominator).to(X.dtype.element_ty).to(tl.float32)
    if BACKWARD:
        dy = tl.load(DY + offsets, offsets < SIZE, 0).to(tl.float32)
        # Eager mul backward also rounds before calling aten.silu_backward.
        ds = (dy * uc).to(X.dtype.element_ty).to(tl.float32)
        sigmoid = tl.div_rn(1.0, denominator)
        dg = (ds * sigmoid) * tl.fma(gc, 1.0 - sigmoid, 1.0)
        du = dy * silu
        if LIMIT is not None:
            dg = tl.where(gate <= limit, dg, 0.0)
            du = tl.where((up >= -limit) & (up <= limit), du, 0.0)
        tl.store(DX + source, dg, offsets < SIZE)
        tl.store(DX + source + I, du, offsets < SIZE)
    else:
        tl.store(Y + offsets, silu * uc, offsets < SIZE)


class RecomputedSwiGLU(torch.autograd.Function):
    """Save only the BF16 projection and recompute cheap pointwise intermediates.

    Fuse pointwise work while preserving eager BF16 rounding and SiLU backward,
    including the inclusive clamp-boundary derivative. Inputs are never mutated.
    """

    @staticmethod
    def forward(ctx, projected, limit):
        """Compute the clamped activation without retaining its intermediate copies."""
        projected = projected.contiguous()
        ctx.save_for_backward(projected)
        ctx.limit = limit
        intermediate = projected.shape[-1] // 2
        output = projected.new_empty((*projected.shape[:-1], intermediate))
        if output.numel():
            with torch.cuda.device(projected.device):
                _swiglu_kernel[(triton.cdiv(output.numel(), 1024),)](
                    projected,
                    output,
                    None,
                    None,
                    output.numel(),
                    intermediate,
                    limit,
                    False,
                    1024,
                    enable_fp_fusion=False,
                )
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """Reconstruct SiLU and clamp values from the original projection."""
        (projected,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad = torch.empty_like(projected)
        if grad_output.numel():
            with torch.cuda.device(projected.device):
                _swiglu_kernel[(triton.cdiv(grad_output.numel(), 1024),)](
                    projected,
                    None,
                    grad_output,
                    grad,
                    grad_output.numel(),
                    projected.shape[-1] // 2,
                    ctx.limit,
                    True,
                    1024,
                    enable_fp_fusion=False,
                )
        return grad, None


class GroupedLinear(torch.autograd.Function):
    """BF16-facing linear with SM90/SM100 FP8 GEMMs in forward, dgrad and wgrad.

    Quantization uses a straight-through training approximation. Weight gradients
    accumulate in FP32: SM90 uses a bounded workspace, while SM100 casts directly
    to the parameter dtype in the GEMM epilogue. Persistent BF16 parameters are
    requantized for dgrad; temporary weights retain an invocation-local FP8
    transpose instead. Neither path uses a module-level quantized cache.
    Gathered FP8 parameters use a byte-exact tiled transpose for dgrad.
    Higher-order gradients are unsupported.
    """

    @staticmethod
    def forward(ctx, x, weight, layout, recompute_weight=True):
        """Return BF16 output and save independently quantized token-axis inputs."""
        with torch.cuda.device(x.device):
            import deep_gemm

            from .fp8_weight import FP8ExpertWeight

            if isinstance(weight, FP8ExpertWeight):
                qw, sw, qwt, swt = weight.qdata, weight.scale, None, None
                recompute_weight = True
            else:
                qw, sw, qwt, swt = quantize_weights(
                    weight, store_transpose=ctx.needs_input_grad[0] and not recompute_weight
                )
            ctx.layout = layout
            ctx.weight_shape = weight.shape
            ctx.weight_dtype = weight.dtype
            ctx.input_shape = x.shape
            ctx.input_dtype = x.dtype
            output = x.new_empty((layout.rows.numel(), weight.shape[1]))
            if output.shape[0]:
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    quantize_rows(x, layout),
                    (qw, sw),
                    output,
                    layout.groups,
                    recipe=(1, 128, 128),
                    disable_ue8m0_cast=not layout.use_ue8m0,
                )
            del qw, sw
            unpadded = output.index_select(0, layout.unpad)
            del output
            # Wgrad needs token-axis quantization, not a transpose of the forward
            # row quantization. Produce it from BF16 now so autograd need not
            # retain that larger source activation until backward.
            # Sparse expert groups can expand substantially when padded. Keep
            # BF16 instead when it is smaller than the FP8 payload plus scales.
            column_bytes = layout.rows.numel() * (x.shape[1] + 4 * (x.shape[1] // 128))
            ctx.quantized_input = column_bytes < x.numel() * x.element_size()
            qx, sx = None, None
            if ctx.needs_input_grad[1]:
                qx, sx = quantize_columns(x, layout) if ctx.quantized_input else (x, None)
            # Existing BF16 parameters already occupy storage. Requantize them
            # at dgrad rather than retaining an extra FP8 copy across forward.
            # Callers retain FP8 instead for temporary BF16 weight copies.
            source_weight = weight if ctx.needs_input_grad[0] and recompute_weight else None
            ctx.save_for_backward(qx, sx, qwt, swt, source_weight)
            return unpadded

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """Compute input and weight gradients using independently quantized operands."""
        with torch.cuda.device(grad_output.device):
            import deep_gemm

            qx, sx, qwt, swt, source_weight = ctx.saved_tensors
            # Unlike clear_saved_tensors_on_access, this honors retain_graph.
            # Local references keep operands alive until their last use below.
            ctx.maybe_clear_saved_tensors()
            layout = ctx.layout
            dx = dw = None
            if ctx.needs_input_grad[0]:
                if source_weight is not None:
                    from .fp8_weight import FP8ExpertWeight

                    if isinstance(source_weight, FP8ExpertWeight):
                        # Square blocks preserve the exact quantization under transpose.
                        qwt, swt = transpose_fp8_weight(source_weight.qdata, source_weight.scale)
                    else:
                        _, _, qwt, swt = quantize_weights(source_weight, store_normal=False)
                padded_dx = torch.empty(
                    (layout.rows.numel(), ctx.input_shape[1]), device=grad_output.device, dtype=ctx.input_dtype
                )
                if padded_dx.shape[0]:
                    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                        quantize_rows(grad_output, layout),
                        (qwt, swt),
                        padded_dx,
                        layout.groups,
                        recipe=(1, 128, 128),
                        disable_ue8m0_cast=not layout.use_ue8m0,
                    )
                dx = padded_dx.index_select(0, layout.unpad)
                del padded_dx
            # Dgrad is the last consumer of the weight transpose. Do not overlap
            # it with wgrad's output, column operands and FP32 workspace.
            del qwt, swt, source_weight
            if ctx.needs_input_grad[1]:
                if ctx.quantized_input:
                    dw = _weight_gradient_quantized_input(
                        qx, sx, grad_output, layout, ctx.weight_shape, ctx.weight_dtype
                    )
                else:
                    dw = _weight_gradient(qx, grad_output, layout, ctx.weight_shape, ctx.weight_dtype)
            return dx, dw, None, None
