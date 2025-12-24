import torch
import torch_npu
import torch.nn as nn

class RMSNorm_npu(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]

def fused_apply_rotary_pos_emb_vision(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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

def fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    orig_dtype = q.dtype
    cos_4d = cos.unsqueeze(unsqueeze_dim).float()
    sin_4d = sin.unsqueeze(unsqueeze_dim).float()

    q_contig = q.float().contiguous()
    k_contig = k.float().contiguous()
    q_embed = torch_npu.npu_rotary_mul(q_contig, cos_4d, sin_4d)
    k_embed = torch_npu.npu_rotary_mul(k_contig, cos_4d, sin_4d)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)