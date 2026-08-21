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
from ....utils.device import IS_NPU_AVAILABLE
from ...loader import MODELING_REGISTRY


def _create_qwen3_5_moe_checkpoint_tensor_converter(model):
    """Create the shared Qwen3 MoE converter for trunk and MTP expert weights."""
    from ..qwen3_moe.checkpoint_tensor_converter import Qwen3MoeCheckpointTensorConverter

    text_config = getattr(model.config, "text_config", model.config)
    return Qwen3MoeCheckpointTensorConverter(num_experts=text_config.num_experts)


# Qwen3.5 GatedDeltaNet's three fused ops (FusedRMSNormGated, causal_conv1d,
# chunk_gated_delta_rule) currently only ship GPU backends via FLA / FlashQLA.
# Setting any of these to a non-eager value on NPU raises at OpSlot.bind via
# KERNEL_REGISTRY.resolve's HardwareRequirement check; the varlen
# (dyn_bsz=True) caveat is documented in docs/usage/arguments.md and on the
# OpsImplementationConfig field metadata.
#
# NPU branch is opt-in; everything else (CUDA, CPU-only) falls back to the GPU
# generated file. The GPU generated module imports cleanly without an active
# CUDA device, so a CPU-only environment (e.g. CI lint, doc build) can still
# register the class.


@MODELING_REGISTRY.register("qwen3_5_moe")
def register_qwen3_5_moe_modeling(architecture: str):
    """Register Qwen3.5-MoE classes with checkpoint conversion support."""
    from ..qwen3_moe.checkpoint_tensor_converter import convert_qwen3_moe_fqn_to_index_mapping

    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_5_moe_npu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )
    else:
        from .generated.patched_modeling_qwen3_5_moe_gpu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )

    for model_cls in (Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration):
        model_cls._create_checkpoint_tensor_converter = staticmethod(_create_qwen3_5_moe_checkpoint_tensor_converter)
        model_cls._convert_fqn_to_index_mapping = staticmethod(convert_qwen3_moe_fqn_to_index_mapping)

    if "ForCausalLM" in architecture:
        return Qwen3_5MoeForCausalLM
    elif "ForConditionalGeneration" in architecture:
        return Qwen3_5MoeForConditionalGeneration
    else:
        return Qwen3_5MoeForCausalLM


@MODELING_REGISTRY.register("qwen3_5_moe_text")
def register_qwen3_5_moe_text_modeling(architecture: str):
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_5_moe_npu import Qwen3_5MoeForCausalLM
    else:
        from .generated.patched_modeling_qwen3_5_moe_gpu import Qwen3_5MoeForCausalLM

    return Qwen3_5MoeForCausalLM
