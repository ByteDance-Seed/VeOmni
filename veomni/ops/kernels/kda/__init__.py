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

"""chunk_kda kernel registry entry (Kimi-K3 KDA linear attention).

Backends:
    - ``ascendc``: fused AscendC forward+backward from the
      flash-linear-attention-npu package
      (https://github.com/flashserve/flash-linear-attention-npu), vendored
      from MindSpeed-MM's kimi_k3 ``chunk_kda_ascendc_kimi`` wrapper (which
      in turn derives from flash-linear-attention, MIT). Supports the full
      Kimi-K3 training contract (varlen cu_seqlens, use_gate_in_kernel,
      beta-sigmoid) with initial_state=None / output_final_state=False --
      the training path after source fix #16. Only small l2norm triton
      kernels remain on the triton runtime.
The eager fallback lives in the Kimi-K3 modeling itself
(``source/kda_fallback.py`` ``chunk_kda_naive``).

History: the standalone ``npu`` (triton fused) entry was removed on
2026-10-10 -- EP>1 NaN race with the MoE alltoall (stage-6/7 finding, deep
cause in HCCL/kernel concurrency) plus an 8-card OOM in the vision conv
backward before the first step; the ascendc path covers production with no
upstream dependency.
"""

from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


def _chunk_kda_ascendc_factory():
    from .ascendc.chunk_kda_ascendc_kimi import chunk_kda_ascendc

    return chunk_kda_ascendc


KERNEL_REGISTRY.register(
    KernelSpec(
        name="ascendc",
        op_name="chunk_kda",
        variant="kimi_k3",
        factory=_chunk_kda_ascendc_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="Kimi-K3 KDA chunked linear attention fused AscendC kernel "
        "(flash-linear-attention-npu / MindSpeed-MM wrapper) on NPU; avoids the "
        "triton chunk kernels entirely on the training path",
    ),
)
