import torch_npu

from . import modeling_qwen3


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


def apply_qwen3_npu_patch():
    # Patches for Qwen3 Model
    modeling_qwen3.Qwen3RMSNorm.forward = rms_norm_forward_npu
    modeling_qwen3.apply_rotary_pos_emb = apply_rotary_pos_emb_npu
