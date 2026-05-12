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

"""Rotary positional embedding kernel registry entry.

Default per-model backends:
    - ``liger_kernel``: ``liger_kernel.transformers.rope.liger_rotary_pos_emb``
    - ``npu``: ``torch_npu.npu_rotary_mul`` via ``veomni.ops.kernels.rotary.npu``
Models can register a ``triton`` (deterministic bmm / Wan DiT) backend via
``extra_backends`` in their ``device_patch.py``.
"""

from ...config.registry import BackendSpec, OpScope, OpSpec, register_op
from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


register_op(
    OpSpec(
        name="rotary_pos_emb",
        config_field="rotary_pos_emb_implementation",
        label="RoPE",
        scope=OpScope.PER_MODEL,
        default="liger_kernel",
        backends={
            "liger_kernel": BackendSpec(
                entry="liger_kernel.transformers.rope:liger_rotary_pos_emb",
                requires=("liger_kernel",),
            ),
            "npu": BackendSpec(
                entry="veomni.ops.kernels.rotary.npu:apply_rotary_pos_emb_npu",
                requires=("torch_npu",),
            ),
        },
    )
)


# ── rotary_pos_emb (Torch_npu) ───────────────────────────────────


def _npu_full_apply_rotary_pos_emb_factory():
    from .npu import apply_rotary_pos_emb_npu

    return apply_rotary_pos_emb_npu


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="rotary_pos_emb",
        variant="full",
        factory=_npu_full_apply_rotary_pos_emb_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="full apply_rotary_pos_emb on NPU",
    )
)


def _npu_apply_rotary_pos_emb_vision_factory():
    from .npu import apply_rotary_pos_emb_vision_npu

    return apply_rotary_pos_emb_vision_npu


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="rotary_pos_emb_vision",
        variant="full",
        factory=_npu_apply_rotary_pos_emb_vision_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="full apply_rotary_pos_emb_vision on NPU",
    )
)


def _npu_partial_apply_rotary_pos_emb_factory():
    from .npu import partial_apply_rotary_pos_emb_npu

    return partial_apply_rotary_pos_emb_npu


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="rotary_pos_emb",
        variant="partial",
        factory=_npu_partial_apply_rotary_pos_emb_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="partial apply_rotary_pos_emb on NPU",
    )
)
