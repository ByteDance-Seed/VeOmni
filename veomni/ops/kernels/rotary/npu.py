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

"""Ascend NPU optimised rotary positional embedding kernels, used as the
``npu`` backend for the ``rotary_pos_emb`` op in the kernel registry.

Two variants are exposed:

* ``apply_rotary_pos_emb_npu`` – standard text-token RoPE (``[B, H, S, D]``).
* ``apply_rotary_pos_emb_vision_npu`` – vision tower RoPE (``[S, H, D]``)
  used by Qwen3-VL and friends.
* ``partial_apply_rotary_pos_emb_npu`` – partial RoPE (``[B, H, S, D]``).
  used by Qwen3.5 and friends.
"""

import torch
import torch_npu


def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """NPU optimized implementation for RoPE."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def apply_rotary_pos_emb_vision_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    q, k = q.unsqueeze(0), k.unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(2).float()
    sin = sin.unsqueeze(0).unsqueeze(2).float()
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    q_embed, k_embed = q_embed.squeeze(0), k_embed.squeeze(0)
    return q_embed, k_embed


def partial_apply_rotary_pos_emb_npu(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


__all__ = ["apply_rotary_pos_emb_npu", "apply_rotary_pos_emb_vision_npu", "partial_apply_rotary_pos_emb_npu"]
