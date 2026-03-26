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

from ....ops.batch_invariant_ops import batch_invariant_rms_norm, triton_bmm
from ....utils import logging
from ....utils.env import get_env
from ....utils.import_utils import is_liger_kernel_available


logger = logging.get_logger(__name__)


def _make_deterministic_rope_forward():
    """Build a RotaryEmbedding.forward that uses a deterministic Triton bmm kernel.

    The default ``inv_freq @ position_ids`` dispatches to cuBLAS bmm which is
    non-deterministic on the first call for certain GPU architectures.  Replacing
    it with an explicit Triton batched-GEMM kernel eliminates this issue.
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import dynamic_rope_update

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


def _patch_rms_norm():
    """Replace DeepseekV3RMSNorm.forward with fused Triton kernel.

    The fused kernel computes pow2+mean+rsqrt+weight in one pass per row,
    which is both faster and batch-invariant.  Supports backward for training.
    """

    def _fused_rms_norm_forward(self, hidden_states):
        return batch_invariant_rms_norm(hidden_states, self.weight, self.variance_epsilon)

    hf_deepseek_v3.DeepseekV3RMSNorm.forward = _fused_rms_norm_forward


def apply_veomni_deepseek_v3_gpu_patch():
    if is_liger_kernel_available() and get_env("VEOMNI_USE_LIGER_KERNEL") == "1":
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.rope import liger_rotary_pos_emb
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        hf_deepseek_v3.apply_rotary_pos_emb = liger_rotary_pos_emb
        hf_deepseek_v3.DeepseekV3RMSNorm = LigerRMSNorm
        hf_deepseek_v3.DeepseekV3MLP = LigerSwiGLUMLP
        logger.info_rank0("Apply liger kernel to deepseek_v3.")
    else:
        # Fused Triton kernels: deterministic RoPE bmm + batch-invariant RMSNorm.
        # Faster than native PyTorch and ensures numerical reproducibility
        # across different batch compositions (required for VeRL rollout matching).
        hf_deepseek_v3.DeepseekV3RotaryEmbedding.forward = _make_deterministic_rope_forward()
        _patch_rms_norm()
        logger.info_rank0("Apply Triton RoPE and RMSNorm to deepseek_v3.")
