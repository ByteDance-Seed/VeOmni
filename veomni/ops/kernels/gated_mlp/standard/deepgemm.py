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

"""Hopper/Blackwell DeepGEMM FP8 gated-MLP forward and backward."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .....utils.device import get_gpu_compute_capability, is_nvidia_cuda_device


#: Formats this row can compute. DeepGEMM's blockwise recipe is FP8 with 128-wide
#: tiles on both the reduction and the output axis; it does not cover FP4 yet.
SUPPORTED_DTYPES = ("fp8",)
#: The one block width that recipe supports. Not a tuning knob: the GEMM indexes a
#: weight's scale by the same block counter as the activation's.
_BLOCK = 128
#: The activations :class:`RecomputedSwiGLU` implements, in both directions.
_FUSED_ACTIVATIONS = frozenset({"silu", "swish"})


def wrapper(
    mlp: nn.Module,
    x: Tensor,
    *,
    swiglu_limit: float | None = None,
    dtype: str = "fp8",
    block_size: int | None = None,
    activation: str = "silu",
) -> Tensor:
    """Run gated-MLP GEMMs on the FP8 tensor cores: forward, dgrad and wgrad.

    One fused FP8 grouped GEMM produces gate and up together, the clamp and
    ``act_fn(gate) * up`` run in the model dtype inside
    :class:`RecomputedSwiGLU` -- which fuses the activation into its own backward
    rather than calling :func:`~veomni.ops.activation.apply_activation` -- and a
    second FP8 GEMM produces the down projection. Hidden and intermediate sizes
    must be positive multiples of 128.

    ``mlp`` is the model's MLP block. This row quantizes the weights it multiplies,
    so :func:`.._module.render` refuses a wrapped projection or an ``nn.Linear``
    subclass rather than quantizing something other than what the projection
    computes -- and because it quantizes, it cannot fall back to the projection
    modules.

    ``dtype`` and ``block_size`` are validated rather than ignored: ``deepgemm``
    has exactly one recipe, and accepting another value would let one field mean
    two formats depending on which row the config selected. ``block_size=None``
    means "the row's own recipe" and is the default, matching the ``linear`` rows
    and ``torchao``; the ``qat`` row defaults it to 128 and honors the value.
    """
    from .._module import render

    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"gated_mlp/standard/deepgemm supports dtype={SUPPORTED_DTYPES}, got {dtype!r}. "
            "DeepGEMM ships blockwise FP8 kernels only."
        )
    if block_size is not None and block_size != _BLOCK:
        raise ValueError(
            f"gated_mlp/standard/deepgemm supports only block_size={_BLOCK}, got {block_size}. "
            "The block width is part of DeepGEMM's blockwise FP8 recipe, not a per-call choice. "
            "``None`` means the row's own recipe."
        )
    gate_w, gate_b, up_w, up_b, down_w, down_b = render(mlp, impl="deepgemm", activation=activation).weights

    if not is_nvidia_cuda_device(x.device) or get_gpu_compute_capability(x.device) not in (90, 100):
        raise RuntimeError("deepgemm requires NVIDIA Hopper SM90 or Blackwell SM100.")
    if x.ndim < 2 or x.dtype != torch.bfloat16:
        raise ValueError("deepgemm requires BF16 input with at least 2 dimensions.")
    intermediate, hidden = gate_w.shape
    if up_w.shape != (intermediate, hidden) or down_w.shape != (hidden, intermediate) or x.shape[-1] != hidden:
        raise ValueError("Incompatible gated-MLP weight shapes or hidden dimension.")
    if min(hidden, intermediate) == 0 or hidden % _BLOCK or intermediate % _BLOCK:
        raise ValueError("FP8 gated-MLP requires positive dimensions and hidden/intermediate multiples of 128.")
    for weight in (gate_w, up_w, down_w):
        if weight.dtype != torch.bfloat16 or weight.device != x.device:
            raise ValueError("FP8 gated-MLP weights must be BF16 and on the same device as input.")
    for bias, expected in ((gate_b, intermediate), (up_b, intermediate), (down_b, hidden)):
        if bias.numel() != 0 and (bias.shape != (expected,) or bias.device != x.device):
            raise ValueError("Non-empty gated-MLP bias must match its projection output width and device.")

    from ...moe_experts.shared.deepgemm import GroupedLinear, RecomputedSwiGLU, get_uniform_layout

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
    # ``RecomputedSwiGLU`` fuses the clamp with ``aten.silu`` in both directions,
    # which is why this row is SiLU-only; ``render`` already kept a non-SiLU model
    # off it. The exact spelling is checked here so a direct call cannot slip a
    # ``gelu`` request into a kernel that would compute SiLU.
    #
    # This is a plain membership check, and that is sufficient: ``ActivationType``
    # is a ``str`` enum, so ``ActivationType.SILU in _FUSED_ACTIVATIONS`` is ``True``
    # because ``__hash__`` is ``str.__hash__``. Only a non-``str`` enum would need
    # ``is_silu``'s ``.value`` unwrapping, and no config field carries one today.
    if activation not in _FUSED_ACTIVATIONS:
        raise ValueError(
            f"gated_mlp/standard/deepgemm implements {sorted(_FUSED_ACTIVATIONS)} only, got "
            f"activation={activation!r}. Its Triton kernel fuses the activation into the grouped "
            "GEMM's forward and backward, so another activation needs its own kernel."
        )
    activated = RecomputedSwiGLU.apply(projected, swiglu_limit)
    out_2d = GroupedLinear.apply(activated, fc2, layout, recompute_fc2)
    if down_b.numel() > 0:
        out_2d = out_2d + down_b.to(out_2d.dtype)
    return out_2d.reshape(*x.shape[:-1], hidden)
