"""Fused physical-alias combine with unchanged low-precision rounding points."""

import torch
from torch.autograd.function import once_differentiable

from ...utils.device import get_torch_device
from .moe_utils import compact_unpermute


class _WeightedCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weights, logical_order, logical_to_physical):
        from ...ops.kernels.moe.ep_combine import build_alias_maps, weighted_combine_forward

        direct = torch.empty_like(logical_order)
        inverse = torch.empty_like(logical_order)
        ranks = torch.empty_like(logical_order)
        output = torch.empty((weights.shape[0], value.shape[1]), dtype=value.dtype, device=value.device)
        with get_torch_device().device(value.device):
            if logical_order.numel():
                build_alias_maps(logical_order, logical_to_physical, direct, inverse, ranks)
            if output.numel():
                weighted_combine_forward(value, weights, ranks, direct, output)
        ctx.save_for_backward(value, weights, inverse)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        from ...ops.kernels.moe.ep_combine import weighted_combine_backward

        value, weights, inverse = ctx.saved_tensors
        dx = torch.empty_like(value) if ctx.needs_input_grad[0] else None
        product = torch.empty_like(value) if ctx.needs_input_grad[1] else None
        with get_torch_device().device(value.device):
            if value.numel():
                weighted_combine_backward(value, weights, gradient.contiguous(), inverse, dx, product)
            dw = None
            if product is not None:
                # Preserve the original BF16 product and ATen hidden-dimension
                # reduction; only row order is changed, not the reduction tree.
                reduced = product.sum(dim=1, keepdim=True).squeeze(1)
                dw = torch.ops.aten.index_select_backward(reduced, [weights.numel()], 0, inverse).reshape_as(weights)
        return dx, dw, None, None


def weighted_combine(value, weights, shape, logical_order, logical_to_physical):
    """Combine validated alias rows in the original logical occurrence order.

    Maps must be permutations produced by the EP dispatcher (not arbitrary
    user indices). Metadata shape/device checks do not add GPU value scans.
    The CUDA BF16 path supports first-order gradients for hidden widths aligned
    to 16 elements; other widths, dtypes, layouts, devices and top-k sizes retain
    the original PyTorch implementation.
    """
    if value.ndim != 2 or weights.ndim != 2 or weights.shape[1] <= 0:
        raise ValueError("Expected expert rows and nonempty top-k weights as matrices.")
    count = weights.numel()
    if value.shape[0] != count or tuple(shape) != (weights.shape[0], value.shape[1]):
        raise ValueError("Value, weight and output shapes do not match.")
    if value.device != weights.device or value.layout != torch.strided or weights.layout != torch.strided:
        raise ValueError("Values and weights must be dense and on the same device.")
    for mapping in (logical_order, logical_to_physical):
        if (
            mapping.ndim != 1
            or mapping.numel() != count
            or mapping.dtype != torch.long
            or mapping.device != value.device
        ):
            raise ValueError("Alias maps must be matching one-dimensional int64 tensors on the value device.")
    compatible = (
        value.is_cuda
        and value.dtype == weights.dtype == torch.bfloat16
        # ATen's vectorized sum handles a misaligned row head separately.
        # Reordering rows is bit-preserving only when every row starts at the
        # same vector alignment. Both products are fresh contiguous buffers.
        and value.shape[1] % 16 == 0
        and value.data_ptr() % 32 == weights.data_ptr() % 32 == 0
        and weights.shape[1] in (1, 2, 4, 8, 16)
        and count <= ((1 << 63) - 1) // weights.shape[1]
        # The backward launches one block per physical row/hidden tile; its
        # grid is at least as large as the corresponding forward grid.
        and count * ((value.shape[1] + 255) // 256) <= (1 << 31) - 1
        and all(tensor.is_contiguous() for tensor in (value, weights, logical_order, logical_to_physical))
    )
    if not compatible:
        return compact_unpermute(value.index_select(0, logical_to_physical), weights, shape, logical_order)
    return _WeightedCombine.apply(value, weights, logical_order, logical_to_physical)
