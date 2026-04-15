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
from ....utils.env import get_env
from ....utils.import_utils import is_liger_kernel_available, is_transformers_version_greater_or_equal_to
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("deepseek_v3")
def register_deepseek_v3_modeling(architecture: str):
    if is_transformers_version_greater_or_equal_to("5.2.0"):
        from .checkpoint_tensor_converter import create_deepseek_v3_checkpoint_tensor_converter

        if IS_NPU_AVAILABLE:
            from .generated.patched_modeling_deepseek_v3_npu import (
                DeepseekV3ForCausalLM,
                DeepseekV3ForSequenceClassification,
                DeepseekV3Model,
            )
        else:
            from .generated.patched_modeling_deepseek_v3_gpu import (
                DeepseekV3ForCausalLM,
                DeepseekV3ForSequenceClassification,
                DeepseekV3Model,
            )

            # When liger is disabled (e.g. for VeRL rollout parity), swap in the
            # deterministic Triton RoPE + batch-invariant RMSNorm kernels. The
            # generated GPU file bakes in LigerRMSNorm / liger rotary as the
            # happy path; this branch restores the v4 behavior where
            # ``VEOMNI_USE_LIGER_KERNEL != "1"`` opts into batch-invariant ops.
            if IS_CUDA_AVAILABLE and not (is_liger_kernel_available() and get_env("VEOMNI_USE_LIGER_KERNEL") == "1"):
                from ....ops.batch_invariant_ops import batch_invariant_rms_norm
                from .generated import patched_modeling_deepseek_v3_gpu as gen
                from .gpu_patch import _make_deterministic_rope_forward

                def _fused_rms_norm_forward(self, hidden_states):
                    return batch_invariant_rms_norm(hidden_states, self.weight, self.variance_epsilon)

                gen.DeepseekV3RotaryEmbedding.forward = _make_deterministic_rope_forward()
                gen.DeepseekV3RMSNorm.forward = _fused_rms_norm_forward

        for model_cls in (DeepseekV3ForCausalLM, DeepseekV3ForSequenceClassification, DeepseekV3Model):
            model_cls._create_checkpoint_tensor_converter = staticmethod(
                create_deepseek_v3_checkpoint_tensor_converter
            )
    else:
        from transformers import (
            DeepseekV3ForCausalLM,
            DeepseekV3ForSequenceClassification,
            DeepseekV3Model,
        )

        from .modeling_deepseek_v3 import apply_veomni_deepseek_v3_patch

        apply_veomni_deepseek_v3_patch()

    if "ForCausalLM" in architecture:
        return DeepseekV3ForCausalLM
    elif "ForSequenceClassification" in architecture:
        return DeepseekV3ForSequenceClassification
    elif "Model" in architecture:
        return DeepseekV3Model
    else:
        return DeepseekV3ForCausalLM
