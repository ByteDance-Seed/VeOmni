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

import torch
import transformers.models.deepseek_v3.modeling_deepseek_v3 as hf_deepseek_v3

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
        "DeepseekV3RMSNorm",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.rms_norm", "LigerRMSNorm"),
            "npu": ImplSpec("veomni.ops.npu_patch.npu_fused_operator", "rms_norm_forward_npu", replace_forward=True),
        },
    ),
    swiglu_patch(
        "DeepseekV3MLP",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.swiglu", "LigerSwiGLUMLP"),
        },
    ),
]


def _make_deterministic_rope_forward():
    """Build a RotaryEmbedding.forward that uses a deterministic Triton bmm kernel.

    The default ``inv_freq @ position_ids`` dispatches to cuBLAS bmm which is
    non-deterministic on the first call for certain GPU architectures.  Replacing
    it with an explicit Triton batched-GEMM kernel eliminates this issue.
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import dynamic_rope_update

    from ....ops.batch_invariant_ops import triton_bmm

    @torch.no_grad()
    @dynamic_rope_update
    def _deterministic_rope_forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = triton_bmm(
                inv_freq_expanded.float().contiguous(),
                position_ids_expanded.float().contiguous(),
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    return _deterministic_rope_forward


def _patch_triton_rms_norm():
    """Replace DeepseekV3RMSNorm.forward with fused Triton kernel."""
    from ....ops.batch_invariant_ops import batch_invariant_rms_norm

    def _fused_rms_norm_forward(self, hidden_states):
        return batch_invariant_rms_norm(hidden_states, self.weight, self.variance_epsilon)

    hf_deepseek_v3.DeepseekV3RMSNorm.forward = _fused_rms_norm_forward


def _custom_deepseek_v3(ops_config, applied):
    if ops_config.rotary_pos_emb_implementation == "triton":
        hf_deepseek_v3.DeepseekV3RotaryEmbedding.forward = _make_deterministic_rope_forward()
        applied.append("RoPE (triton)")
    if ops_config.rms_norm_implementation == "triton":
        _patch_triton_rms_norm()
        applied.append("RMSNorm (triton)")


def apply_veomni_deepseek_v3_device_patch():
    apply_device_patches(hf_deepseek_v3, PATCHES, "DeepSeek-V3", custom_patches=_custom_deepseek_v3)
