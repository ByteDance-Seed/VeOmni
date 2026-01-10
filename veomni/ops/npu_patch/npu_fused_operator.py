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

import torch_npu


# This api can improve performance on ASCEND NPU
def rms_norm_forward_npu(self, x):
    """NPU optimized implementation for RMSNorm."""
    if x.dtype != self.weight.dtype:
        x = x.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """NPU optimized implementation for RoPE."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def apply_rotary_pos_emb_vision_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    orig_q_shape = q.shape
    orig_k_shape = k.shape
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q_4d = q.unsqueeze(0).float().contiguous()
    k_4d = k.unsqueeze(0).float().contiguous()
    cos_4d = cos.unsqueeze(0).unsqueeze(2).float()
    sin_4d = sin.unsqueeze(0).unsqueeze(2).float()
    q_embed_4d = torch_npu.npu_rotary_mul(q_4d, cos_4d, sin_4d)
    k_embed_4d = torch_npu.npu_rotary_mul(k_4d, cos_4d, sin_4d)
    q_embed = q_embed_4d.squeeze(0).to(orig_q_dtype)
    k_embed = k_embed_4d.squeeze(0).to(orig_k_dtype)
    q_embed = q_embed.reshape(orig_q_shape)
    k_embed = k_embed.reshape(orig_k_shape)
    return q_embed, k_embed
