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
"""
LigerKernel-based kernel registrations (SwiGLU).

Registrations are executed at import time via ``veomni.ops.__init__``.
RMSNorm and RoPE live in ``veomni.kernels``.
"""

from __future__ import annotations

from ..kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


# ── Liger SwiGLU MLP ─────────────────────────────────────────────────────────


def _liger_swiglu_factory():
    """Return a functional SwiGLU MLP kernel using LigerSiLUMulFunction.

    Matches LigerSwiGLUMLP.forward in:
    https://github.com/linkedin/Liger-Kernel/blob/v0.7.0/src/liger_kernel/transformers/swiglu.py
    """
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    def liger_swiglu_forward(self, x):
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))

    return liger_swiglu_forward


KERNEL_REGISTRY.register(
    KernelSpec(
        name="liger_kernel",
        op_name="swiglu_mlp",
        variant="standard",
        factory=_liger_swiglu_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="LigerKernel fused SwiGLU MLP",
    )
)
