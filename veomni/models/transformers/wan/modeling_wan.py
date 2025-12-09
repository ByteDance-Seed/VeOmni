# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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

import math
import os
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.modeling_utils import PreTrainedModel

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor_scale_grad,
)

from ....utils import logging
from ....utils.import_utils import is_liger_kernel_available
from .config_wan import WanConfig


if is_liger_kernel_available():
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from veomni.distributed.sequence_parallel.async_ulysses_dit import (
    async_ulysses_output_projection,
    async_ulysses_qkv_projection,
)


logger = logging.get_logger(__name__)

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False


try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def stochastic_round_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Perform stochastic rounding on a tensor
    Args:
        x (torch.Tensor): Input tensor
    Returns:
        torch.Tensor: Stochastically rounded integer tensor
    """
    floor_x = torch.floor(x)
    frac = x - floor_x
    rand_vals = torch.rand_like(x)
    round_up = rand_vals < frac
    result = floor_x + round_up.to(x.dtype)
    return result


def symmetric_quantize(x, dtype=torch.float8_e4m3fn):
    """
    Dynamic symmetric quantization that supports block-wise quantization under multi-head attention mechanism
    Args:
        x: Input tensor [batch_size, seq_len, head_count, head_dim]
        dtype: Target quantization type, defaults to torch.float8_e4m3fn
    Returns:
        x_quantized: Quantized tensor [batch_size, seq_len, head_count, head_dim]
        scales: Scaling factors for each head [batch_size, head_count]
    """
    batch_size, seq_len, head_count, head_dim = x.shape
    x = x.to(torch.float32)
    max_vals = x.abs().amax(dim=(1, 3), keepdim=True)  # [batch, 1, head, 1]
    finfo = torch.finfo(dtype)
    eps = 1e-12  # Smaller epsilon for better stability
    scales = (max_vals + eps) / finfo.max  # Ensure non-zero denominator
    scales = scales.clamp(min=eps)
    x_scaled = x / scales
    is_round = True
    if is_round:
        x_rounded = stochastic_round_tensor(x_scaled)
        x_clamped = x_rounded.clamp(min=finfo.min, max=finfo.max)
    else:
        x_clamped = x_scaled.clamp(min=finfo.min, max=finfo.max)
    x_quantized = x_clamped.to(dtype)
    scales = scales.squeeze((1, 3)).to(torch.float32)
    return x_quantized, scales


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def rearrange_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rerange_type: str, head_dim: int):
    q = rearrange(q, rerange_type, d=head_dim)
    k = rearrange(k, rerange_type, d=head_dim)
    v = rearrange(v, rerange_type, d=head_dim)
    return q, k, v


def gather_seq_scatter_heads_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_dim: int, head_dim: int):
    q = gather_seq_scatter_heads(q, seq_dim, head_dim)
    k = gather_seq_scatter_heads(k, seq_dim, head_dim)
    v = gather_seq_scatter_heads(v, seq_dim, head_dim)
    return q, k, v


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    **kwargs,
):
    head_dim = query.shape[-1]
    scaling = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.bfloat16).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def wrapped_sageattention(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    **kwargs,
):
    assert SAGE_ATTN_AVAILABLE
    head_dim = query_states.shape[-1]
    rerange_type_head_seq = "b n s d -> b s n d"
    attn_output = sageattn(query_states, key_states, value_states)
    attn_output = rearrange(attn_output, rerange_type_head_seq, d=head_dim)
    return attn_output


def wrapped_flash_attention_3(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    last_loss=None,
    isSelfAttn=False,
    **kwargs,
):
    assert FLASH_ATTN_3_AVAILABLE
    head_dim = query_states.shape[-1]
    rerange_type_seq_head = "b n s d -> b s n d"

    q, k, v = rearrange_qkv(query_states, key_states, value_states, rerange_type_seq_head, head_dim=head_dim)

    if isSelfAttn and last_loss is not None:
        if math.isnan(last_loss):
            attn_output = flash_attn_interface.flash_attn_func(q, k, v)
        else:
            original_q = q
            original_k = k
            original_v = v
            q, qscale = symmetric_quantize(q, dtype=torch.float8_e4m3fn)
            k, kscale = symmetric_quantize(k, dtype=torch.float8_e4m3fn)
            v, vscale = symmetric_quantize(v, dtype=torch.float8_e4m3fn)
            attn_output = flash_attn_interface.flash_attn_func(
                q,
                k,
                v,
                q_descale=qscale,
                k_descale=kscale,
                v_descale=vscale,
                original_q=original_q,
                original_k=original_k,
                original_v=original_v,
            )
    else:
        attn_output = flash_attn_interface.flash_attn_func(q, k, v)

    return attn_output


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, **kwargs):
    freqs = kwargs.pop("freqs")
    head_dim = kwargs.pop("head_dim")
    x = rearrange(x, "b s (n d) -> b s n d", d=head_dim)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(pad_size, s1, s2, dtype=original_tensor.dtype, device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, config, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attention_interface = WAN_ATTENTION_FUNCTIONS[config._attn_implementation]
        self.is_causal = False
        self.config = config

    def forward(self, query_states, key_states, value_states, **kwargs):
        query_states = rearrange(query_states, "b s (n d) -> b n s d", d=self.head_dim)
        key_states = rearrange(key_states, "b s (n d) -> b n s d", d=self.head_dim)
        value_states = rearrange(value_states, "b s (n d) -> b n s d", d=self.head_dim)

        kwargs["attention_mask"] = kwargs.get("attention_mask", None)

        deterministic = kwargs.get("deterministic", None)
        if deterministic is None:
            deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        kwargs["deterministic"] = deterministic

        attn_output = self.attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            **kwargs,
        )

        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        attn_output = rearrange(attn_output, "b s n d -> b s (n d)")
        return attn_output


class SelfAttention(nn.Module):
    def __init__(self, config, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(config, self.num_heads, self.head_dim)
        self.sp_async = False

    def forward(self, x, freqs, cos, sin, last_loss):
        if not self.sp_async:
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)

            if get_parallel_state().ulysses_enabled:
                q, k, v = gather_seq_scatter_heads_qkv(q, k, v, seq_dim=1, head_dim=2)

        else:
            q, k, v = async_ulysses_qkv_projection(
                hidden_states=x,
                seq_dimension=1,
                head_dimension=2,
                q_weight=self.q.weight,
                k_weight=self.k.weight,
                v_weight=self.v.weight,
                norm_type="rmsnorm",
                norm_q_weight=self.norm_q.weight,
                norm_k_weight=self.norm_k.weight,
                normalized_shape=self.dim,
                eps=1e-6,
                unpadded_dim_size=x.shape[1] * get_ulysses_sequence_parallel_world_size(),
            )

        q = rope_apply(q, freqs=freqs, cos=cos, sin=sin, head_dim=self.head_dim)
        k = rope_apply(k, freqs=freqs, cos=cos, sin=sin, head_dim=self.head_dim)

        x = self.attn(q, k, v, last_loss=last_loss, isSelfAttn=True)

        if not self.sp_async:
            if get_parallel_state().ulysses_enabled:
                x = gather_heads_scatter_seq(x, seq_dim=1, head_dim=2)

            x = self.o(x)
        else:
            x = async_ulysses_output_projection(
                hidden_states=x,
                seq_dimension=1,
                head_dimension=2,
                proj_weight=self.o.weight,
                proj_bias=self.o.bias,
                unpadded_dim_size=x.shape[1],
            )
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        config,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        has_image_input: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(config, self.num_heads, self.head_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = self.attn(q, k_img, v_img, head_dim=self.head_dim)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        config,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(config, dim, num_heads, eps)
        self.cross_attn = CrossAttention(config, dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, cos, sin, last_loss):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, cos, sin, last_loss))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        B, D = t_mod.shape
        t_mod = t_mod.view(B, 1, D)  # [B, 1, D]
        modulation = self.modulation.expand(B, -1, -1).to(t_mod)
        combined = modulation + t_mod
        shift, scale = (c.squeeze(1) for c in combined.chunk(2, dim=1))
        normalized = self.norm(x)
        modulated = normalized * (1 + scale[:, None, :]) + shift[:, None, :]
        return self.head(modulated)


class WanModel(PreTrainedModel):
    config_class = WanConfig
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["DiTBlock"]

    def __init__(self, config: WanConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.dim = config.dim
        self.freq_dim = config.freq_dim

        # TODO: add image input
        if config.has_image_input == "true":
            self.has_image_input = True
        else:
            self.has_image_input = False
        self.patch_size = config.patch_size
        self.micro_batch_size = 1
        self.patch_embedding = nn.Conv3d(
            config.in_dim,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, config.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.dim, config.dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.dim, config.dim * 6),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    config,
                    self.has_image_input,
                    config.dim,
                    config.num_heads,
                    config.ffn_dim,
                    config.eps,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.head = Head(config.dim, config.out_dim, config.patch_size, config.eps)
        head_dim = config.dim // config.num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if self.has_image_input:
            self.img_emb = MLP(1280, config.dim)  # clip_feature_dim = 1280

        self.gradient_checkpointing = False

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        last_loss=None,
        **kwargs,
    ):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))

        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            if self.micro_batch_size > 1:
                x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
                clip_embdding = self.img_emb(clip_feature)
                context = context.squeeze(1)
                clip_embdding = clip_embdding.squeeze(1)
                context = torch.cat([clip_embdding, context], dim=1)
            else:
                x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
                clip_embdding = self.img_emb(clip_feature)
                context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        cos = freqs.real.squeeze().contiguous()
        sin = freqs.imag.squeeze().contiguous()

        if get_parallel_state().ulysses_enabled:
            x = slice_input_tensor_scale_grad(x, dim=1)

        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(
                    block.__call__,
                    x,
                    context,
                    t_mod,
                    freqs,
                    cos,
                    sin,
                    last_loss=last_loss,
                )
            else:
                x = block(x, context, t_mod, freqs, cos, sin, last_loss=last_loss)

        x = self.head(x, t)

        if get_parallel_state().ulysses_enabled:
            x = gather_outputs(x, gather_dim=1)
        x = self.unpatchify(x, (f, h, w))
        return x


if is_liger_kernel_available():
    RMSNorm = LigerRMSNorm
    logger.info_rank0("Apply liger kernel to Wan.")

try:
    from veomni.ops.dit.rope_wan.rotary import apply_rotary_emb

    rope_apply = apply_rotary_emb
    logger.info_rank0("Apply fused interleaved rope to Wan.")
except ImportError:
    pass

WAN_ATTENTION_FUNCTIONS: Dict[str, Dict[str, Callable]] = {}
WAN_ATTENTION_FUNCTIONS.update(ALL_ATTENTION_FUNCTIONS)
WAN_ATTENTION_FUNCTIONS.update(
    {
        "eager": eager_attention_forward,
        "flash_attention_3": wrapped_flash_attention_3,
        "sageattention": wrapped_sageattention,
    }
)
