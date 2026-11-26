"""Internal BF16 combine kernels for validated physical/logical permutations."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["count"])
def _alias_maps(
    logical_order,
    logical_to_physical,
    physical_by_occurrence,
    occurrence_by_physical,
    logical_rank,
    count,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    occurrence = tl.load(logical_order + index, index < count, other=0)
    physical = tl.load(logical_to_physical + index, index < count, other=0)
    tl.store(physical_by_occurrence + occurrence, physical, index < count)
    tl.store(occurrence_by_physical + physical, occurrence, index < count)
    tl.store(logical_rank + occurrence, index, index < count)


@triton.jit
def _weighted_combine_forward(X, W, RANKS, MAP, Y, H: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    tiles: tl.constexpr = triton.cdiv(H, BLOCK)
    program = tl.program_id(0).to(tl.int64)
    token = program // tiles
    columns = (program % tiles) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    slots = tl.arange(0, TOPK)
    logical = tl.load(RANKS + token * TOPK + slots)
    # The wrapper bounds count*TOPK so packing cannot overflow int64.
    keys = tl.sort(logical * TOPK + slots, descending=False)
    result = tl.full((BLOCK,), 0, tl.float32)
    for ordinal in tl.static_range(TOPK):
        slot = tl.sum(tl.where(slots == ordinal, keys % TOPK, 0), axis=0)
        occurrence = token * TOPK + slot
        physical = tl.load(MAP + occurrence)
        weight = tl.load(W + occurrence).to(tl.float32)
        values = tl.load(X + physical * H + columns, columns < H, other=0).to(tl.float32)
        weighted = (values * weight).to(X.dtype.element_ty).to(tl.float32)
        result = result + weighted
    tl.store(Y + token * H + columns, result.to(Y.dtype.element_ty), columns < H)


@triton.jit
def _weighted_combine_backward(
    X,
    W,
    DY,
    ORDER,
    DX,
    PRODUCT,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    NEED_INPUT: tl.constexpr,
    NEED_WEIGHTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(H, BLOCK)
    program = tl.program_id(0).to(tl.int64)
    physical = program // tiles
    column = (program % tiles) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    occurrence = tl.load(ORDER + physical)
    gradient = tl.load(DY + (occurrence // TOPK) * H + column, column < H, other=0).to(tl.float32)
    if NEED_INPUT:
        weight = tl.load(W + occurrence).to(tl.float32)
        dx = (gradient * weight).to(DX.dtype.element_ty)
        # Match the original unique-index accumulation into a +0 buffer.
        dx = tl.where(dx == 0, 0, dx)
        tl.store(DX + physical * H + column, dx, column < H)
    if NEED_WEIGHTS:
        value = tl.load(X + physical * H + column, column < H, other=0).to(tl.float32)
        product = (gradient * value).to(PRODUCT.dtype.element_ty)
        tl.store(PRODUCT + physical * H + column, product, column < H)


def build_alias_maps(logical_order, logical_to_physical, direct, inverse, ranks):
    count = logical_order.numel()
    _alias_maps[(triton.cdiv(count, 256),)](logical_order, logical_to_physical, direct, inverse, ranks, count, 256)


def weighted_combine_forward(value, weights, ranks, direct, output):
    rows, hidden = output.shape
    _weighted_combine_forward[(rows * triton.cdiv(hidden, 256),)](
        value,
        weights,
        ranks,
        direct,
        output,
        hidden,
        weights.shape[1],
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )


def weighted_combine_backward(value, weights, gradient, inverse, dx, product):
    rows, hidden = value.shape
    _weighted_combine_backward[(rows * triton.cdiv(hidden, 256),)](
        value,
        weights,
        gradient,
        inverse,
        dx if dx is not None else value,
        product if product is not None else value,
        hidden,
        weights.shape[1],
        dx is not None,
        product is not None,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
