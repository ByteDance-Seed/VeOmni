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

"""Nabla fused SiLU-mul (Triton forward + backward) and SwiGLU MLP.

Ported from the Nabla kernel-optimization project (Apache-2.0), where the
activation core was validated on H100 against liger-kernel v0.7.0 with an
fp32 oracle: output, grad_gate and grad_up all pass bf16 tolerances
(atol=7e-2, rtol=2e-2); fwd+bwd latency is 1.10-1.13x liger at training
shapes (M in {2048, 4096, 8192}, intermediate=768, bf16).

Only the fusable activation core ``silu(gate) * up`` is fused; the
gate/up/down projections stay on cuBLAS, exactly like
``LigerSiLUMulFunction`` / ``LigerSwiGLUMLP``:

    out = silu(gate) * up        # silu(x) = x * sigmoid(x)

The Triton path requires contiguous CUDA tensors; anything else (CPU,
missing Triton, non-contiguous / mismatched inputs) transparently falls back
to an fp32 pure-torch implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

_BLOCK = 1024


def _torch_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    orig_dtype = gate.dtype
    return (F.silu(gate.float()) * up.float()).to(orig_dtype)


if _HAS_TRITON:

    @triton.jit
    def _silu_mul_fwd_kernel(GATE, UP, OUT, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.sigmoid(g)
        out = g * s * u
        tl.store(OUT + offs, out.to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _silu_mul_bwd_kernel(GATE, UP, DOUT, DGATE, DUP, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)
        do = tl.load(DOUT + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.sigmoid(g)
        silu = g * s
        dsilu = s * (1.0 + g * (1.0 - s))
        dgate = do * u * dsilu
        dup = do * silu
        tl.store(DGATE + offs, dgate.to(DGATE.dtype.element_ty), mask=mask)
        tl.store(DUP + offs, dup.to(DUP.dtype.element_ty), mask=mask)


def _grid(n_elements: int) -> tuple[int, ...]:
    return (triton.cdiv(n_elements, _BLOCK),)


def _silu_mul_fwd_launch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(gate)
    n = gate.numel()
    _silu_mul_fwd_kernel[_grid(n)](gate, up, out, n, BLOCK=_BLOCK)
    return out


def _silu_mul_bwd_launch(
    gate: torch.Tensor, up: torch.Tensor, grad_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_out = grad_out.contiguous()
    dgate = torch.empty_like(gate)
    dup = torch.empty_like(up)
    n = gate.numel()
    _silu_mul_bwd_kernel[_grid(n)](gate, up, grad_out, dgate, dup, n, BLOCK=_BLOCK)
    return dgate, dup


class NablaSiLUMulFunction(torch.autograd.Function):
    """Fused silu(gate) * up forward + backward (Triton)."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = _silu_mul_fwd_launch(gate, up)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        gate, up = ctx.saved_tensors
        dgate, dup = _silu_mul_bwd_launch(gate, up, grad_output)
        return dgate, dup


def _can_use_triton(gate: torch.Tensor, up: torch.Tensor) -> bool:
    return (
        _HAS_TRITON
        and gate.is_cuda
        and up.is_cuda
        and gate.is_contiguous()
        and up.is_contiguous()
        and gate.shape == up.shape
    )


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``silu(gate) * up``. Triton fwd+bwd when possible, torch fallback otherwise."""
    if _can_use_triton(gate, up):
        return NablaSiLUMulFunction.apply(gate, up)
    return _torch_silu_mul(gate, up)


def nabla_swiglu_forward(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """OpSlot-shaped SwiGLU MLP forward: replaces ``{Model}MLP.forward(self, x)``.

    Same shape as the registered liger kernel: the module supplies
    ``gate_proj`` / ``up_proj`` / ``down_proj``; only the activation core is
    fused, the projections stay on cuBLAS.
    """
    return module.down_proj(silu_mul(module.gate_proj(x), module.up_proj(x)))


class NablaSwiGLUMLP(torch.nn.Module):
    """Drop-in SwiGLU MLP module for the ``swiglu_mlp_implementation="nabla"`` backend.

    ABI-compatible with ``LigerSwiGLUMLP``: constructed as
    ``NablaSwiGLUMLP(config)`` where ``config`` exposes ``hidden_size``,
    ``intermediate_size`` and ``hidden_act``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))
