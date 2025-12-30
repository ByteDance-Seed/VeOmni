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

import torch_npu

from . import modeling_qwen3_vl
from ....ops.npu_patch import npu_fused_operator

def apply_rotary_pos_emb_vision_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    orig_dtype = q.dtype
    q_4d = q.unsqueeze(0).float().contiguous()
    k_4d = k.unsqueeze(0).float().contiguous()
    cos_4d = cos.unsqueeze(0).unsqueeze(2).float()
    sin_4d = sin.unsqueeze(0).unsqueeze(2).float()

    q_embed_4d = torch_npu.npu_rotary_mul(q_4d, cos_4d, sin_4d)
    k_embed_4d = torch_npu.npu_rotary_mul(k_4d, cos_4d, sin_4d)

    q_embed = q_embed_4d.transpose(1, 2).to(orig_dtype)
    k_embed = k_embed_4d.transpose(1, 2).to(orig_dtype)

    return q_embed, k_embed

def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    orig_dtype = q.dtype
    cos_4d = cos.unsqueeze(unsqueeze_dim).float()
    sin_4d = sin.unsqueeze(unsqueeze_dim).float()

    q_contig = q.float().contiguous()
    k_contig = k.float().contiguous()
    q_embed = torch_npu.npu_rotary_mul(q_contig, cos_4d, sin_4d)
    k_embed = torch_npu.npu_rotary_mul(k_contig, cos_4d, sin_4d)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

def apply_qwen3vl_npu_patch():
    # Patches for Qwen3VL Model
    modeling_qwen3_vl.Qwen3VLTextRMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
    modeling_qwen3_vl.apply_rotary_pos_emb = apply_rotary_pos_emb_npu
    modeling_qwen3_vl.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision_npu
