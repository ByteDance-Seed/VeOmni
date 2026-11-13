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

"""Hopper/Blackwell DeepGEMM FP8 SwiGLU MLP forward and backward."""

from __future__ import annotations

import torch
from torch import Tensor

from .....utils.device import get_gpu_compute_capability, is_nvidia_cuda_device
from ...linear._module import require_deepgemm_block_size


#: The activations this row's recomputed kernel implements, in both directions.
_FUSED_ACTIVATIONS = frozenset({"silu", "swish"})


def wrapper(
    x: Tensor,
    gate_w: Tensor,
    gate_b: Tensor,
    up_w: Tensor,
    up_b: Tensor,
    down_w: Tensor,
    down_b: Tensor,
    *,
    swiglu_limit: float | None = None,
    block_size: int = 128,
    activation: str = "silu",
) -> Tensor:
    """Run FP8 gated-MLP GEMMs with BF16 inputs, outputs, and first-order gradients.

    Uses DeepGEMM blockwise FP8 grouped GEMMs (single group) for the gate/up and
    down projections and a recomputed Triton SwiGLU activation with optional
    clamping. Hidden and intermediate dimensions must be positive multiples of
    128. Empty bias tensors denote unused biases. ``block_size`` is fixed by
    DeepGEMM's blockwise recipe at 128; see the ``linear`` rows.

    ``activation`` is part of the family's call contract (#1462): every
    ``swiglu_mlp`` row takes it so a model can pass ``config.hidden_act`` without
    branching on which row the config selected. This row's recomputed kernel
    implements SiLU only -- it calls ``aten.silu`` in the forward and
    ``aten.silu_backward`` in the backward, and both are fused into the same
    Triton launch as the clamp -- so anything else is refused here rather than
    silently computed as SiLU. ``row_supports_activation`` already keeps a
    non-SiLU model off this row; this check is what makes a direct call honest.
    """
    if activation not in _FUSED_ACTIVATIONS:
        raise ValueError(
            f"swiglu_mlp_implementation='deepgemm_fp8' implements {sorted(_FUSED_ACTIVATIONS)} only, "
            f"got activation={activation!r}. Its Triton kernel fuses the activation into the "
            "grouped GEMM's forward and backward, so another activation needs its own kernel."
        )
    require_deepgemm_block_size(block_size)
    if not is_nvidia_cuda_device(x.device) or get_gpu_compute_capability(x.device) not in (90, 100):
        raise RuntimeError("deepgemm_fp8 requires NVIDIA Hopper SM90 or Blackwell SM100.")
    if x.ndim < 2 or x.dtype != torch.bfloat16:
        raise ValueError("deepgemm_fp8 requires BF16 input with at least 2 dimensions.")
    if gate_w.ndim != 2 or up_w.ndim != 2 or down_w.ndim != 2:
        raise ValueError("SwiGLU MLP weights must be 2D tensors.")
    intermediate, hidden = gate_w.shape
    if up_w.shape != (intermediate, hidden) or down_w.shape != (hidden, intermediate) or x.shape[-1] != hidden:
        raise ValueError("Incompatible SwiGLU MLP weight shapes or hidden dimension.")
    if min(hidden, intermediate) == 0 or hidden % 128 or intermediate % 128:
        raise ValueError("FP8 SwiGLU MLP requires positive dimensions and hidden/intermediate multiples of 128.")
    for weight in (gate_w, up_w, down_w):
        if weight.dtype != torch.bfloat16 or weight.device != x.device:
            raise ValueError("FP8 SwiGLU MLP weights must be BF16 and on the same device as input.")
    for bias, expected in ((gate_b, intermediate), (up_b, intermediate), (down_b, hidden)):
        if bias.numel() != 0 and (bias.shape != (expected,) or bias.device != x.device):
            raise ValueError("Non-empty SwiGLU MLP bias must match its projection output width and device.")

    from ...moe_experts.shared.deepgemm_fp8 import GroupedLinear, RecomputedSwiGLU, get_uniform_layout

    x_2d = x.reshape(-1, hidden)
    layout = get_uniform_layout(x_2d.shape[0], 1, x.device)
    # gate and up share one GEMM: their operands are the same activation, so
    # concatenating them along the output axis halves the launch count and the
    # activation quantization.
    fc1 = torch.cat((gate_w, up_w), dim=0).unsqueeze(0).contiguous()
    fc2 = down_w.contiguous().unsqueeze(0)
    recompute_fc2 = isinstance(down_w, torch.nn.Parameter) and down_w.is_contiguous()

    projected = GroupedLinear.apply(x_2d, fc1, layout, False)
    if gate_b.numel() > 0 or up_b.numel() > 0:
        gb = gate_b.to(projected.dtype) if gate_b.numel() > 0 else projected.new_zeros(intermediate)
        ub = up_b.to(projected.dtype) if up_b.numel() > 0 else projected.new_zeros(intermediate)
        projected = projected + torch.cat((gb, ub), dim=0)
    activated = RecomputedSwiGLU.apply(projected, swiglu_limit)
    out_2d = GroupedLinear.apply(activated, fc2, layout, recompute_fc2)
    if down_b.numel() > 0:
        out_2d = out_2d + down_b.to(out_2d.dtype)
    return out_2d.reshape(*x.shape[:-1], hidden)
