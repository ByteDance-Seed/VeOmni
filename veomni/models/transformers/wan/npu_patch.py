import torch
import torch_npu
from einops import rearrange

from . import modeling_wan


def rms_norm_forward_npu(self, x):
    """NPU optimized implementation for RMSNorm."""
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]


def rope_apply_fused(x, **kwargs):
    """NPU optimized implementation for RoPE."""
    freqs = kwargs.pop("freqs")
    cos = freqs.real.to(torch.float32).unsqueeze(0).repeat_interleave(2, dim=-1).contiguous()
    sin = freqs.imag.to(torch.float32).unsqueeze(0).repeat_interleave(2, dim=-1).contiguous()
    head_dim = kwargs.pop("head_dim")
    x = rearrange(x, "b s (n d) -> b s n d", d=head_dim)
    x_float = x.to(torch.float32)
    x_out = torch_npu.npu_rotary_mul(x_float, cos, sin, rotary_mode="interleave").flatten(-2)
    return x_out.to(x.dtype)


def apply_wan_npu_patch():
    # Patches for wan2.1 Model
    modeling_wan.RMSNorm.forward = rms_norm_forward_npu
    modeling_wan.rope_apply = rope_apply_fused
