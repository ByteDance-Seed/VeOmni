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
"""
Patch configuration for Qwen3-Omni-Moe.

Regen command:
patchgen veomni.models_kernel.transformers.qwen3_omni_moe.qwen3_omni_moe_npu_patch_gen_config -o veomni/models_kernel/transformers/qwen3_omni_moe/generated --diff

NPU keeps torch_npu fused rope/rmsnorm. Those were never OpSlot. GPU consume already swapped MoE/CE/LB.
"""

import torch

from veomni.models_kernel.transformers.qwen3_omni_moe.qwen3_omni_moe_gpu_patch_gen_config import config


config.target_file = "patched_modeling_qwen3_omni_moe_npu.py"
config.description = "Qwen3OmniMoe with NPU"

config.add_import("torch_npu", names=["npu_rotary_mul"])
config.add_import("torch_npu", names=["npu_rms_norm"])


@config.replace_function("apply_rotary_pos_emb_vision", description="Replace with the fusion operator on Ascend.")
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()

    q_embed = npu_rotary_mul(q.float(), cos, sin, rotary_mode="half").to(q.dtype)
    k_embed = npu_rotary_mul(k.float(), cos, sin, rotary_mode="half").to(k.dtype)

    return q_embed, k_embed


@config.override_method(
    "Qwen3OmniMoeThinkerTextRMSNorm.forward",
    description="NPU fused RMSNorm -- reduces pow+mean+rsqrt to single npu_rms_norm call",
)
def qwen3_omni_moe_thinker_text_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """NPU optimized implementation for RMSNorm"""
    input_dtype = hidden_states.dtype
    out_fp32 = npu_rms_norm(hidden_states.float(), self.weight.float(), epsilon=self.variance_epsilon)[0]
    return out_fp32.to(input_dtype)
