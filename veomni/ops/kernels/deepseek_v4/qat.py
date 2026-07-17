# Copyright 2026 the Miles contributors and ByteDance Ltd. and/or its affiliates
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

"""DeepSeek-V4 FP8 activation quantization-aware training helpers.

Adapted from radixark/miles' DeepSeek-V4 QAT implementation.
"""

from __future__ import annotations

import torch


def fp8_simulate(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Simulate E4M3 quantization with per-token UE8M0 block scales."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"DeepSeek-V4 FP8 activation QAT expects BF16 inputs, got {x.dtype}")
    if x.shape[-1] % block_size != 0:
        raise ValueError(f"The last dimension ({x.shape[-1]}) must be divisible by block_size ({block_size})")

    from tile_kernels.quant import per_token_cast_back

    from .act_quant import act_quant

    contiguous = x.contiguous()
    quantized, scale = act_quant(contiguous, block_size, "ue8m0")
    width = contiguous.shape[-1]
    quantized_flat = quantized.view(-1, width)
    scale_flat = scale.reshape(quantized_flat.shape[0], width // block_size).contiguous()
    output = per_token_cast_back((quantized_flat, scale_flat), "bf16", block_size)
    return output.view_as(contiguous)


class _DeepseekV4FP8ActivationQAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
        return fp8_simulate(x, block_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def fp8_simulate_qat(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Apply FP8 fake quantization with a straight-through gradient."""
    return _DeepseekV4FP8ActivationQAT.apply(x, block_size)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply the normalized Hadamard rotation used by the V4 indexer QAT path."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"DeepSeek-V4 Hadamard rotation expects BF16 inputs, got {x.dtype}")
    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek-V4 FP8 activation QAT requires fast-hadamard-transform; "
            "install it after syncing the GPU environment as documented in "
            "docs/design/kernel_selection.md"
        ) from exc
    return hadamard_transform(x.contiguous(), scale=x.shape[-1] ** -0.5)


__all__ = ["fp8_simulate", "fp8_simulate_qat", "rotate_activation"]
