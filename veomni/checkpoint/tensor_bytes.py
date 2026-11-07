# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Bounded, logical-order tensor bytes for checkpoint identity checks."""

import torch


def tensor_byte_chunks(tensor, *, chunk_bytes):
    """Yield CPU byte chunks without materializing a full non-contiguous tensor."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    tensor = tensor.detach()
    count = max(1, chunk_bytes // max(1, tensor.element_size()))

    def blocks(view):
        if view.numel() == 0:
            return
        if view.numel() <= count:
            yield view
            return
        if view.is_contiguous():
            flat = view.reshape(-1)
            for start in range(0, flat.numel(), count):
                yield flat[start : start + count]
            return
        # If a complete trailing slice fits, copy several leading slices at
        # once. Otherwise descend one dimension without materializing a pool
        # of indices or a full contiguous copy of the original tensor.
        trailing = view.numel() // view.shape[0]
        if trailing <= count:
            rows = max(1, count // trailing)
            for start in range(0, view.shape[0], rows):
                yield view[start : start + rows]
        else:
            for index in range(view.shape[0]):
                yield from blocks(view[index])

    for block in blocks(tensor):
        flat = block.contiguous().view(-1)
        if flat.stride(0) != 1:
            # PyTorch can report a one-element strided view as contiguous,
            # while dtype reinterpretation still requires unit stride.
            flat = torch.empty(flat.numel(), dtype=flat.dtype, device=flat.device).copy_(flat)
        yield flat.view(torch.uint8).cpu().numpy().tobytes()
