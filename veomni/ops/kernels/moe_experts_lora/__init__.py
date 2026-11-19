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

"""MoE experts LoRA kernel family.

``shared`` is one LoRA pair per logical spec across experts. ``independent``
is a per-expert pair. Eager is the per-expert loop. ``fused_triton`` and
``fused_npu`` wrap the local fused Functions. ``fused_quack`` / ``fused_mlu`` are not registered; callers remap
those ``moe_implementation`` values to eager.

All three rows serve expert parallelism through ``dispatch_to_ep_class``, so the
all-to-all plumbing is the same one ``moe_experts`` uses. Only the per-rank math
differs: ``fused_npu`` runs Ascend grouped GEMMs in ``helper.npu_ep_forward``, while
``eager`` walks the local experts with plain PyTorch -- which makes it the reference
for the fused rows rather than a training choice. Only the base weights are sharded.
"""

from __future__ import annotations

from torch import Tensor

from ...platform import NVIDIA_SM70_PLUS, ROCM_GPU, GpuKernelRequirement, NpuKernelRequirement
from ...registry import register_op
from .independent import eager as independent_eager
from .independent import npu as independent_npu
from .shared import eager as shared_eager
from .shared import npu as shared_npu


_GPU_SM70_OR_ROCM = GpuKernelRequirement(platforms=(NVIDIA_SM70_PLUS, ROCM_GPU))


def _shared_triton_wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_a_down: Tensor,
    lora_b_down: Tensor,
    *,
    num_experts: int,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
) -> Tensor:
    """Lazy-load and call the shared Triton MoE-LoRA implementation."""
    from .shared.triton import wrapper

    return wrapper(
        hidden_states,
        routing_weights,
        selected_experts,
        fc1_1_2_weight,
        fc2_weight,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        num_experts=num_experts,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
    )


def _independent_triton_wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_a_down: Tensor,
    lora_b_down: Tensor,
    *,
    num_experts: int,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
) -> Tensor:
    """Lazy-load and call the independent Triton MoE-LoRA implementation."""
    from .independent.triton import wrapper

    return wrapper(
        hidden_states,
        routing_weights,
        selected_experts,
        fc1_1_2_weight,
        fc2_weight,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        num_experts=num_experts,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
    )


register_op(
    "moe_experts_lora",
    "shared",
    "eager",
    description="PyTorch routed MoE experts with LoRA weights shared across experts",
    wrapper=shared_eager.wrapper,
)

register_op(
    "moe_experts_lora",
    "shared",
    "fused_triton",
    description="Triton routed MoE experts with LoRA weights shared across experts",
    wrapper=_shared_triton_wrapper,
    requirement=_GPU_SM70_OR_ROCM,
    requires=("triton",),
)

register_op(
    "moe_experts_lora",
    "shared",
    "fused_npu",
    description="torch_npu routed MoE experts with LoRA weights shared across experts",
    wrapper=shared_npu.wrapper,
    requirement=NpuKernelRequirement(),
)

register_op(
    "moe_experts_lora",
    "independent",
    "eager",
    description="PyTorch routed MoE experts with independent per-expert LoRA weights",
    wrapper=independent_eager.wrapper,
)

register_op(
    "moe_experts_lora",
    "independent",
    "fused_triton",
    description="Triton routed MoE experts with independent per-expert LoRA weights",
    wrapper=_independent_triton_wrapper,
    requirement=_GPU_SM70_OR_ROCM,
    requires=("triton",),
)

register_op(
    "moe_experts_lora",
    "independent",
    "fused_npu",
    description="torch_npu routed MoE experts with independent per-expert LoRA weights",
    wrapper=independent_npu.wrapper,
    requirement=NpuKernelRequirement(),
)
