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

"""RMSNorm kernel registry entry.

Default per-model backends:
    - ``liger_kernel``: ``liger_kernel.transformers.rms_norm.LigerRMSNorm``
    - ``npu``: ``torch_npu.npu_rms_norm`` via ``veomni.ops.kernels.rms_norm.npu``
Models can register a ``triton`` backend via ``extra_backends`` in their
``device_patch.py`` (e.g. DeepSeek V3 batch-invariant kernel).

Opt-in GPU backend:
    - ``nabla``: single-kernel Triton fwd+bwd (grad_weight accumulated in one
      fp32 buffer via atomics), validated on H100 against liger v0.7.0.
"""

from ...config.registry import BackendSpec, OpScope, OpSpec, register_op
from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


register_op(
    OpSpec(
        name="rms_norm",
        config_field="rms_norm_implementation",
        label="RMSNorm",
        scope=OpScope.PER_MODEL,
        default="liger_kernel",
        backends={
            "liger_kernel": BackendSpec(
                entry="liger_kernel.transformers.rms_norm:LigerRMSNorm",
                requires=("liger_kernel",),
            ),
            "nabla": BackendSpec(
                entry="veomni.ops.kernels.rms_norm.nabla:NablaRMSNorm",
                requires=("triton",),
            ),
            "npu": BackendSpec(
                entry="veomni.ops.kernels.rms_norm.npu:rms_norm_forward_npu",
                requires=("torch_npu",),
                replace_forward=True,
            ),
        },
    )
)


def _nabla_rms_norm_factory():
    """Return the functional Nabla fused RMSNorm kernel (standard formulation)."""

    from .nabla import rms_norm

    return rms_norm


KERNEL_REGISTRY.register(
    KernelSpec(
        name="nabla",
        op_name="rms_norm",
        variant="standard",
        factory=_nabla_rms_norm_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="Nabla fused RMSNorm (single-kernel Triton fwd+bwd)",
    )
)

# ── rms_norm (Torch_npu) ───────────────────────────────────


def _npu_standard_rms_norm_factory():
    from .npu import standard_rms_norm_forward_npu

    return standard_rms_norm_forward_npu


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="rms_norm",
        variant="standard",
        factory=_npu_standard_rms_norm_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="standard fused RMSNorm on NPU",
    )
)


def _npu_unweighted_rms_norm_factory():
    from .npu import unweighted_rms_norm_forward_npu

    return unweighted_rms_norm_forward_npu


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="rms_norm",
        variant="unweighted",
        factory=_npu_unweighted_rms_norm_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="unweighted RMSNorm eager fallback on NPU",
    )
)


def _npu_qwen3_5_rms_norm_factory():
    from .npu import qwen3_5_rms_norm_forward_npu

    return qwen3_5_rms_norm_forward_npu


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="rms_norm",
        variant="qwen3_5",
        factory=_npu_qwen3_5_rms_norm_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="Qwen3.5 fused RMSNorm on NPU",
    )
)
