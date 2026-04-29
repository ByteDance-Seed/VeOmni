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
from ....utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE
from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODELING_REGISTRY.register("qwen3_moe")
def register_qwen3_moe_modeling(architecture: str):
    if is_transformers_version_greater_or_equal_to("5.0.0"):
        # Register runtime checkpoint tensor converter for v5+ only.
        # For transformers < v5, users should continue using the offline
        # merge script to produce pre-fused expert weights.
        from .checkpoint_tensor_converter import create_qwen3_moe_checkpoint_tensor_converter

        if IS_CUDA_AVAILABLE:
            from .generated.patched_modeling_qwen3_moe_gpu import (
                Qwen3MoeForCausalLM,
                Qwen3MoeForQuestionAnswering,
                Qwen3MoeModel,
            )
        elif IS_NPU_AVAILABLE:
            from .generated.patched_modeling_qwen3_moe_npu import (
                Qwen3MoeForCausalLM,
                Qwen3MoeForQuestionAnswering,
                Qwen3MoeModel,
            )

        for model_cls in (Qwen3MoeForCausalLM, Qwen3MoeForQuestionAnswering, Qwen3MoeModel):
            model_cls._create_checkpoint_tensor_converter = staticmethod(create_qwen3_moe_checkpoint_tensor_converter)
    else:
        from transformers import (
            Qwen3MoeForCausalLM,
            Qwen3MoeForQuestionAnswering,
            Qwen3MoeModel,
        )

        from .checkpoint_tensor_converter_v4 import create_qwen3_moe_v4_checkpoint_tensor_converter
        from .modeling_qwen3_moe import apply_veomni_qwen3_moe_patch

        apply_veomni_qwen3_moe_patch()

        # Stack per-expert HF weights into v4's three 3-D nn.Parameters at load time
        # (PatchQwen3MoeExperts.gate_proj/up_proj/down_proj). Without this, users had
        # to pre-merge with scripts/moe_ckpt_merge/moe_merge.py before training.
        for model_cls in (Qwen3MoeForCausalLM, Qwen3MoeForQuestionAnswering, Qwen3MoeModel):
            model_cls._create_checkpoint_tensor_converter = staticmethod(
                create_qwen3_moe_v4_checkpoint_tensor_converter
            )

    if "ForCausalLM" in architecture:
        return Qwen3MoeForCausalLM
    elif "ForQuestionAnswering" in architecture:
        return Qwen3MoeForQuestionAnswering
    elif "Model" in architecture:
        return Qwen3MoeModel
    else:
        return Qwen3MoeForCausalLM
