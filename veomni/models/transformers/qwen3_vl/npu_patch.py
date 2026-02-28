# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

import transformers.models.qwen3_vl.modeling_qwen3_vl as hf_qwen3vl

from ....ops.npu_patch import npu_fused_operator


def apply_qwen3vl_npu_patch():
    # Patches for Qwen3VL Model
    hf_qwen3vl.Qwen3VLTextRMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
    hf_qwen3vl.apply_rotary_pos_emb = npu_fused_operator.apply_rotary_pos_emb_npu
    hf_qwen3vl.apply_rotary_pos_emb_vision = npu_fused_operator.apply_rotary_pos_emb_vision_npu
