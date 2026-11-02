"""One-to-one token layout changes without a split/cat autograd graph."""

import torch

from ...utils.device import get_torch_device


def _integers(value, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional CPU tensor or integer sequence.")
        value = value.tolist()
    value = tuple(value)
    if any(type(item) is not int for item in value):
        raise TypeError(f"{name} must contain integers.")
    return value


class _PermuteRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, direct, inverse):
        ctx.save_for_backward(inverse)
        return value.index_select(0, direct)

    @staticmethod
    def backward(ctx, gradient):
        (inverse,) = ctx.saved_tensors
        return gradient.index_select(0, inverse), None, None


class ChunkPermutation:
    """Reuse a validated layout and its inverse for dispatch and combine.

    The maps are permutations, not reductions. Forward and backward only copy
    values and therefore do not change floating-point accumulation order.
    CPU is a reference implementation; CUDA builds both maps in one kernel.
    """

    def __init__(self, split_sizes, sorted_idxs, device):
        sizes = _integers(split_sizes, "split_sizes")
        order = _integers(sorted_idxs, "sorted_idxs")
        if any(size < 0 for size in sizes):
            raise ValueError("split_sizes must be non-negative.")
        if sorted(order) != list(range(len(sizes))):
            raise ValueError("sorted_idxs must be a permutation of every chunk.")
        self.rows = sum(sizes)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("ChunkPermutation supports CUDA and its CPU reference only.")

        starts = []
        offset = 0
        for size in sizes:
            starts.append(offset)
            offset += size
        ends = []
        offset = 0
        for index in order:
            offset += sizes[index]
            ends.append(offset)
        self._direct = torch.empty(self.rows, device=self.device, dtype=torch.long)
        self._inverse = torch.empty_like(self._direct)
        # Resolve an unspecified CUDA index to the actual allocated device.
        self.device = self._direct.device
        if not self.rows:
            return
        if self.device.type == "cuda":
            from ...ops.kernels.moe.ep_permutation import fill_chunk_row_maps

            with get_torch_device().device(self.device):
                metadata = torch.tensor(
                    [ends, [starts[index] for index in order]], device=self.device, dtype=torch.long
                )
                fill_chunk_row_maps(metadata, self._direct, self._inverse, len(sizes), self.rows)
        else:
            chunks = torch.arange(self.rows).split(sizes)
            self._direct = torch.cat([chunks[index] for index in order])
            self._inverse.scatter_(0, self._direct, torch.arange(self.rows))

    def apply(self, value, *, inverse=False):
        if value.ndim != 2 or value.layout != torch.strided or value.shape[0] != self.rows:
            raise ValueError("Input must be a dense matrix matching the permutation row count.")
        if value.device != self.device:
            raise ValueError("Input and permutation must be on the same device.")
        direct, backward = (self._inverse, self._direct) if inverse else (self._direct, self._inverse)
        return _PermuteRows.apply(value, direct, backward)
