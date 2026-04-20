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

from ....ops.config.registry import BackendSpec, apply_per_model_patches


def _make_deterministic_rope_forward():
    """Build a RotaryEmbedding.forward that uses a deterministic Triton bmm kernel.

    The default ``inv_freq @ position_ids`` dispatches to cuBLAS bmm which is
    non-deterministic on the first call for certain GPU architectures.
    Replacing it with an explicit Triton batched-GEMM kernel eliminates this
    issue.
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import dynamic_rope_update

    from ....ops.kernels.rotary.triton_deterministic import triton_bmm

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


def _make_batch_invariant_rms_norm_forward():
    """Build an RMSNorm.forward that uses the batch-invariant Triton kernel."""
    from ....ops.kernels.rms_norm.triton_batch_invariant import batch_invariant_rms_norm

    def _fused_rms_norm_forward(self, hidden_states):
        return batch_invariant_rms_norm(hidden_states, self.weight, self.variance_epsilon)

    return _fused_rms_norm_forward


def apply_veomni_deepseek_v3_device_patch():
    apply_per_model_patches(
        hf_module=hf_deepseek_v3,
        model_name="DeepSeek-V3",
        targets={
            "rotary_pos_emb": "apply_rotary_pos_emb",
            "rms_norm": "DeepseekV3RMSNorm",
            "swiglu_mlp": "DeepseekV3MLP",
        },
        extra_backends={
            "rotary_pos_emb": {
                # The Triton backend replaces ``DeepseekV3RotaryEmbedding.forward``
                # rather than the module-level ``apply_rotary_pos_emb``.
                "triton": BackendSpec(
                    entry="veomni.models.transformers.deepseek_v3.device_patch:_make_deterministic_rope_forward",
                    entry_is_factory=True,
                    replace_forward=True,
                    target_override="DeepseekV3RotaryEmbedding",
                ),
            },
            "rms_norm": {
                "triton": BackendSpec(
                    entry="veomni.models.transformers.deepseek_v3.device_patch:_make_batch_invariant_rms_norm_forward",
                    entry_is_factory=True,
                    replace_forward=True,
                ),
            },
        },
    )
