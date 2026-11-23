# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Reshape-then-swap layout helper. Not a config-selected VeomniOp."""

from __future__ import annotations

from torch import Size, Tensor


def confusion_transpose(
    hidden_states: Tensor,
    *,
    flatten_dim: int = 2,
    transpose_dims: tuple[int, int] = (1, 2),
) -> Tensor:
    """Flatten from ``flatten_dim``, then swap ``transpose_dims``.

    The defaults are Wan's patch embedding, ``[B, C, F, H, W]`` to
    ``[B, S, C]``. NPU tensors pass the same axes to
    ``npu_confusion_transpose`` as a reshape shape and a permutation.
    Every other device uses ``flatten(flatten_dim).transpose(*transpose_dims)``.
    Host availability of ``torch_npu`` does not change the CPU path.
    """
    if hidden_states.device.type == "npu":
        return _npu_confusion_transpose(hidden_states, flatten_dim, transpose_dims)
    dim0, dim1 = transpose_dims
    return hidden_states.flatten(flatten_dim).transpose(dim0, dim1)


def _confusion_transpose_args(
    shape: Size,
    flatten_dim: int,
    transpose_dims: tuple[int, int],
) -> tuple[list[int], list[int]]:
    """Return ``(perm, shape)`` for ``transpose_first=False``.

    ``shape`` is the flattened view. ``perm`` swaps ``transpose_dims`` on that
    view, so it matches ``flatten(flatten_dim).transpose(*transpose_dims)``.
    """
    flat_shape = _flattened_shape(shape, flatten_dim)
    dim0, dim1 = (_normalize_dim(dim, len(flat_shape)) for dim in transpose_dims)
    perm = list(range(len(flat_shape)))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm, flat_shape


def _flattened_shape(shape: Size, flatten_dim: int) -> list[int]:
    """Sizes after ``Tensor.flatten(flatten_dim)``."""
    rank = len(shape)
    flatten_dim = _normalize_dim(flatten_dim, rank)
    tail = 1
    for size in shape[flatten_dim:]:
        tail *= int(size)
    return [int(size) for size in shape[:flatten_dim]] + [tail]


def _normalize_dim(dim: int, rank: int) -> int:
    """Match ``Tensor.flatten`` / ``Tensor.transpose`` negative-dim wrapping."""
    original = dim
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {original})"
        )
    return dim


def _npu_confusion_transpose(
    hidden_states: Tensor,
    flatten_dim: int,
    transpose_dims: tuple[int, int],
) -> Tensor:
    import torch_npu

    perm, shape = _confusion_transpose_args(hidden_states.shape, flatten_dim, transpose_dims)
    return torch_npu.npu_confusion_transpose(
        hidden_states,
        perm,
        shape,
        transpose_first=False,
    )
