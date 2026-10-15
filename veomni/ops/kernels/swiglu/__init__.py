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

"""SwiGLU MLP kernel registry entry.

Default per-model backend:
    - ``liger_kernel``: ``liger_kernel.transformers.swiglu.LigerSwiGLUMLP``

Opt-in GPU backend:
    - ``nabla``: Triton fwd+bwd SiLU-mul activation core (projections stay on
      cuBLAS), validated on H100 against liger v0.7.0.
"""

from ...config.registry import BackendSpec, OpScope, OpSpec, register_op
from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


register_op(
    OpSpec(
        name="swiglu_mlp",
        config_field="swiglu_mlp_implementation",
        label="SwiGLU",
        scope=OpScope.PER_MODEL,
        default="liger_kernel",
        backends={
            "liger_kernel": BackendSpec(
                entry="liger_kernel.transformers.swiglu:LigerSwiGLUMLP",
                requires=("liger_kernel",),
            ),
            "nabla": BackendSpec(
                entry="veomni.ops.kernels.swiglu.nabla:NablaSwiGLUMLP",
                requires=("triton",),
            ),
        },
    )
)


def _nabla_swiglu_factory():
    """Return the OpSlot-shaped Nabla SwiGLU forward (fused SiLU-mul core)."""

    from .nabla import nabla_swiglu_forward

    return nabla_swiglu_forward


KERNEL_REGISTRY.register(
    KernelSpec(
        name="nabla",
        op_name="swiglu_mlp",
        variant="standard",
        factory=_nabla_swiglu_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="Nabla fused SwiGLU MLP (Triton SiLU-mul activation core)",
    )
)
