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

"""situ_and_mul kernel registry entry (Kimi-K3 SituGLU).

Backends:
    - ``npu``: vendored Triton-ascend kernel (MindSpeed-MM upstream), fused
      forward + backward, fp32 compute inside the kernel with fp16/bf16
      input/output.
The eager fallback lives in the Kimi-K3 modeling itself (``SituAndMul``).
"""

from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


def _situ_and_mul_npu_factory():
    from .situ_triton import situ_and_mul

    return situ_and_mul


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="situ_and_mul",
        variant="kimi_k3",
        factory=_situ_and_mul_npu_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="Kimi-K3 SituGLU (beta*tanh(gate/beta)*sigmoid(gate) * up) fused Triton kernel on NPU",
    ),
)
