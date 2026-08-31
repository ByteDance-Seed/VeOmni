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

"""Gated delta-rule kernels used by Qwen3.5 GatedDeltaNet.

Three independent families share this package. They are not a compound
kernel and are not ``rms_norm`` variants.

``rms_norm_gated`` is ``weight * rms_norm(x) * silu(gate)``. Functional extra
is *weight*. FLA's unused norm bias and ``activation`` are accepted.
``causal_conv1d`` is depthwise causal conv on ``[B, S, D]``. ``None`` or empty
bias / ``cu_seqlens`` are unused. FLA ``seq_idx`` / ``backend`` are accepted.
Eager has no varlen path.
``chunk_gated_delta_rule`` takes FLA ``[B, T, H, D]``. ``None`` or empty
``initial_state`` / ``cu_seqlens`` are unused. NPU varlen tables are accepted.
Eager has no varlen path.
"""

from ...registry import register_kernel
from ...requirement import CudaKernelRequirement, NpuKernelRequirement
from .causal_conv1d.standard import eager as conv_eager
from .causal_conv1d.standard import fla as conv_fla
from .causal_conv1d.standard import npu as conv_npu
from .chunk_gated_delta_rule.standard import eager as chunk_eager
from .chunk_gated_delta_rule.standard import fla as chunk_fla
from .chunk_gated_delta_rule.standard import flash_qla as chunk_flash_qla
from .chunk_gated_delta_rule.standard import npu as chunk_npu
from .chunk_gated_delta_rule.standard import npu_ascendc as chunk_ascendc
from .rms_norm_gated.standard import eager as rms_eager
from .rms_norm_gated.standard import fla as rms_fla
from .rms_norm_gated.standard import npu as rms_npu


register_kernel("rms_norm_gated", "standard", "eager", wrapper=rms_eager.wrapper)

register_kernel(
    "rms_norm_gated",
    "standard",
    "fla",
    wrapper=rms_fla.wrapper,
    requirement=CudaKernelRequirement(),
)

register_kernel(
    "rms_norm_gated",
    "standard",
    "npu",
    wrapper=rms_npu.wrapper,
    requirement=NpuKernelRequirement(),
)

register_kernel("causal_conv1d", "standard", "eager", wrapper=conv_eager.wrapper)

register_kernel(
    "causal_conv1d",
    "standard",
    "fla",
    wrapper=conv_fla.wrapper,
    requirement=CudaKernelRequirement(),
)

register_kernel(
    "causal_conv1d",
    "standard",
    "npu",
    conv_npu.forward,
    conv_npu.backward,
    requirement=NpuKernelRequirement(),
)

register_kernel("chunk_gated_delta_rule", "standard", "eager", wrapper=chunk_eager.wrapper)

register_kernel(
    "chunk_gated_delta_rule",
    "standard",
    "fla",
    wrapper=chunk_fla.wrapper,
    requirement=CudaKernelRequirement(),
)

register_kernel(
    "chunk_gated_delta_rule",
    "standard",
    "flash_qla",
    wrapper=chunk_flash_qla.wrapper,
    requirement=CudaKernelRequirement(min_cc=90, max_cc=90),
)

register_kernel(
    "chunk_gated_delta_rule",
    "standard",
    "npu",
    chunk_npu.forward,
    chunk_npu.backward,
    requirement=NpuKernelRequirement(),
)

register_kernel(
    "chunk_gated_delta_rule",
    "standard",
    "npu_ascendc",
    chunk_ascendc.forward,
    chunk_ascendc.backward,
    requirement=NpuKernelRequirement(),
)
