# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
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

import transformers.models.qwen3_moe.modeling_qwen3_moe as hf_qwen3_moe

from ....ops.config.registry import apply_per_model_patches


def apply_veomni_qwen3_moe_device_patch():
    apply_per_model_patches(
        hf_module=hf_qwen3_moe,
        model_name="Qwen3-MoE",
        targets={
            "rotary_pos_emb": "apply_rotary_pos_emb",
            "rms_norm": "Qwen3MoeRMSNorm",
            "swiglu_mlp": "Qwen3MoeMLP",
        },
    )
