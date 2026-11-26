"""The Cosmos3 mixture-of-towers decoder block.

The reasoner (``und``) and generation (``gen``) towers share one attention softmax
but have their own norms, projections and feed-forwards. Parameter names follow the
released checkpoint; the generation tower's MLP and norms carry the ``_moe_gen``
suffix.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from transformers.activations import ACT2FN

from .mot_attention import MotAttention, MotRMSNorm
from .packed_layout import PackedLayout


class MotMLP(nn.Module):
    """SwiGLU feed-forward used by both towers."""

    def __init__(self, hidden_size: int, intermediate_size: int, *, hidden_act: str = "silu") -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden))


class MoTDecoderLayer(nn.Module):
    """One mixture-of-towers block: per-tower pre-norm, joint attention, per-tower FFN."""

    def __init__(self, config, *, layer_idx: int = 0) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        eps = config.rms_norm_eps
        self.layer_idx = layer_idx
        self.self_attn = MotAttention(config, layer_idx=layer_idx)
        self.mlp = MotMLP(hidden_size, config.intermediate_size, hidden_act=config.hidden_act)
        self.mlp_moe_gen = MotMLP(hidden_size, config.intermediate_size, hidden_act=config.hidden_act)
        self.input_layernorm = MotRMSNorm(hidden_size, eps=eps, config=config)
        self.post_attention_layernorm = MotRMSNorm(hidden_size, eps=eps, config=config)
        self.input_layernorm_moe_gen = MotRMSNorm(hidden_size, eps=eps, config=config)
        self.post_attention_layernorm_moe_gen = MotRMSNorm(hidden_size, eps=eps, config=config)

    def forward(
        self,
        hidden_und: Tensor,
        hidden_gen: Tensor,
        *,
        und_rope: tuple[Tensor, Tensor],
        gen_rope: tuple[Tensor, Tensor],
        layout: PackedLayout | None = None,
    ) -> tuple[Tensor, Tensor]:
        attn_und, attn_gen = self.self_attn(
            self.input_layernorm(hidden_und),
            self.input_layernorm_moe_gen(hidden_gen),
            und_rope=und_rope,
            gen_rope=gen_rope,
            layout=layout,
        )
        residual_und = hidden_und + attn_und
        out_und = residual_und + self.mlp(self.post_attention_layernorm(residual_und))
        residual_gen = hidden_gen + attn_gen
        out_gen = residual_gen + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))
        return out_und, out_gen
