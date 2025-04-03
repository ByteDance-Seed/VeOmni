import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from veomni.distributed.sequence_parallel import gather_heads_scatter_seq, gather_seq_scatter_heads
from veomni.distributed.sequence_parallel.async_ulysses import (
    async_ulysses_output_projection,
    async_ulysses_qkv_projection,
)

from .normalization import get_layernorm


"""
This Attention module is adapted from an open-source project, OpenDiT
For more details, see: https://github.com/NUS-HPC-AI-Lab/OpenDiT/blob/master/opendit/modules/attn.py
"""


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sp_async: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.sp_async = sp_async

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = get_layernorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = get_layernorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_o = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
        else:
            self.rope = False
            self.rotary_emb = nn.Identity()

    def forward(self, x: torch.Tensor, unpadded_seq_len: int) -> torch.Tensor:
        if not self.sp_async:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, unpadded_dim_size=unpadded_seq_len)
            k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2, unpadded_dim_size=unpadded_seq_len)
            v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2, unpadded_dim_size=unpadded_seq_len)
            q = rearrange(q, "B N (h d) -> B N h d", d=self.head_dim).contiguous()
            k = rearrange(k, "B N (h d) -> B N h d", d=self.head_dim).contiguous()
            v = rearrange(v, "B N (h d) -> B N h d", d=self.head_dim).contiguous()
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k, v = async_ulysses_qkv_projection(
                hidden_states=x,
                seq_dimension=1,
                head_dimension=2,
                q_weight=self.q_proj.weight,
                q_bias=self.q_proj.bias,
                k_weight=self.k_proj.weight,
                k_bias=self.k_proj.bias,
                v_weight=self.v_proj.weight,
                v_bias=self.v_proj.bias,
                norm_type="layernorm",
                norm_q_weight=self.q_norm.weight,
                norm_q_bias=self.q_norm.bias,
                norm_k_weight=self.k_norm.weight,
                norm_k_bias=self.k_norm.bias,
                normalized_shape=self.head_dim,
                eps=self.q_norm.eps,
                unpadded_dim_size=unpadded_seq_len,
                head_dim=self.head_dim,
            )

        if self.rope:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        x = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x = x.transpose(1, 2)
        x = rearrange(x, "B N h d -> B N (h d)", d=self.head_dim)

        if not self.sp_async:
            x = gather_heads_scatter_seq(x, head_dim=2, seq_dim=1)
            x = self.proj_o(x)
        else:
            x = async_ulysses_output_projection(
                hidden_states=x,
                seq_dimension=1,
                head_dimension=2,
                proj_weight=self.proj_o.weight,
                proj_bias=self.proj_o.bias,
                unpadded_dim_size=unpadded_seq_len,
            )
        x = self.proj_drop(x)
        return x
