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

import transformers.models.qwen2.modeling_qwen2 as hf_qwen2

from ....ops.device_patch_utils import ImplSpec, apply_device_patches, rms_norm_patch, rope_patch, swiglu_patch


PATCHES = [
    rope_patch(
        "apply_rotary_pos_emb",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.rope", "liger_rotary_pos_emb"),
            "npu": ImplSpec("veomni.ops.npu_patch.npu_fused_operator", "apply_rotary_pos_emb_npu"),
        },
    ),
    rms_norm_patch(
        "Qwen2RMSNorm",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.rms_norm", "LigerRMSNorm"),
            "npu": ImplSpec("veomni.ops.npu_patch.npu_fused_operator", "rms_norm_forward_npu", replace_forward=True),
        },
    ),
    swiglu_patch(
        "Qwen2MLP",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.swiglu", "LigerSwiGLUMLP"),
        },
    ),
]


def apply_veomni_qwen2_device_patch():
    apply_device_patches(hf_qwen2, PATCHES, "Qwen2")
