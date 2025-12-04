# Copyright 2024-2025 The Black-forest-labs Authors. All rights reserved.
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
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_utils import PreTrainedModel

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    slice_input_tensor,
)
from veomni.models.transformers.flux.config_flux import FluxConfig
from veomni.models.transformers.flux.utils_flux import (
    FluxDiTStateDictConverter,
    TileWorker,
    TimestepEmbeddings,
    init_weights_on_device,
)
from veomni.utils import logging
from veomni.utils.import_utils import is_liger_kernel_available


try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


if is_liger_kernel_available():
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

logger = logging.get_logger(__name__)


def gather_seq_scatter_heads_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_dim: int, head_dim: int):
    q = gather_seq_scatter_heads(q, seq_dim, head_dim)
    k = gather_seq_scatter_heads(k, seq_dim, head_dim)
    v = gather_seq_scatter_heads(v, seq_dim, head_dim)
    return q, k, v


def rearrange_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rerange_type: str):
    q = rearrange(q, rerange_type)
    k = rearrange(k, rerange_type)
    v = rearrange(v, rerange_type)
    return q, k, v


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, attn_mask=None):
    # bs, head_cont, seq, head_dim = q.shape

    if FLASH_ATTN_3_AVAILABLE or FLASH_ATTN_2_AVAILABLE:
        rerange_type_seq_head = "b n s d -> b s n d"
        rerange_type_head_seq = "b s n d -> b n s d"
        q, k, v = rearrange_qkv(q, k, v, rerange_type_seq_head)

    if FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_func(q, k, v, causal=causal)
        if isinstance(x, tuple):
            x = x[0]
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_func(q, k, v, causal=causal)
    else:
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return x

    x = rearrange(x, rerange_type_head_seq)
    return x


class AdaLayerNorm(torch.nn.Module):
    def __init__(self, dim, single=False, dual=False):
        super().__init__()
        self.single = single
        self.dual = dual
        self.linear = torch.nn.Linear(dim, dim * [[6, 2][single], 9][dual])
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(torch.nn.functional.silu(emb))
        if self.single:
            scale, shift = emb.unsqueeze(1).chunk(2, dim=2)
            x = self.norm(x) * (1 + scale) + shift
            return x
        elif self.dual:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = (
                emb.unsqueeze(1).chunk(9, dim=2)
            )
            norm_x = self.norm(x)
            x = norm_x * (1 + scale_msa) + shift_msa
            norm_x2 = norm_x * (1 + scale_msa2) + shift_msa2
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_x2, gate_msa2
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.unsqueeze(1).chunk(6, dim=2)
            x = self.norm(x) * (1 + scale_msa) + shift_msa
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones((dim,)))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states


def interact_with_ipadapter(hidden_states, q, ip_k, ip_v, scale=1.0):
    batch_size, num_tokens = hidden_states.shape[0:2]
    ip_hidden_states = torch.nn.functional.scaled_dot_product_attention(q, ip_k, ip_v)
    ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, num_tokens, -1)
    hidden_states = hidden_states + scale * ip_hidden_states
    return hidden_states


class RoPEEmbedding(torch.nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def rope(self, pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0, "The dimension must be even."

        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / (theta**scale)

        batch_size, seq_length = pos.shape
        out = torch.einsum("...n,d->...nd", pos, omega)
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)

        stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
        return out.float()

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat([self.rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


class FluxJointAttention(torch.nn.Module):
    def __init__(self, dim_a, dim_b, num_heads, head_dim, only_out_a=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.only_out_a = only_out_a

        self.a_to_qkv = torch.nn.Linear(dim_a, dim_a * 3)
        self.b_to_qkv = torch.nn.Linear(dim_b, dim_b * 3)

        self.norm_q_a = RMSNorm(head_dim, eps=1e-6)
        self.norm_k_a = RMSNorm(head_dim, eps=1e-6)
        self.norm_q_b = RMSNorm(head_dim, eps=1e-6)
        self.norm_k_b = RMSNorm(head_dim, eps=1e-6)

        self.a_to_out = torch.nn.Linear(dim_a, dim_a)
        if not only_out_a:
            self.b_to_out = torch.nn.Linear(dim_b, dim_b)

    def apply_rope(self, xq, xk, freqs_cis):
        # 打印输入大小，在一行

        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def forward(self, hidden_states_a, hidden_states_b, image_rotary_emb, attn_mask=None, ipadapter_kwargs_list=None):
        batch_size = hidden_states_a.shape[0]

        # Part A
        qkv_a = self.a_to_qkv(hidden_states_a)
        qkv_a = qkv_a.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_a, k_a, v_a = qkv_a.chunk(3, dim=1)
        q_a, k_a = self.norm_q_a(q_a), self.norm_k_a(k_a)

        # Part B
        qkv_b = self.b_to_qkv(hidden_states_b)
        qkv_b = qkv_b.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_b, k_b, v_b = qkv_b.chunk(3, dim=1)
        q_b, k_b = self.norm_q_b(q_b), self.norm_k_b(k_b)
        if get_parallel_state().ulysses_enabled:
            q_a, k_a, v_a = gather_seq_scatter_heads_qkv(q_a, k_a, v_a, seq_dim=2, head_dim=1)
            q_b, k_b, v_b = gather_seq_scatter_heads_qkv(q_b, k_b, v_b, seq_dim=2, head_dim=1)

        q = torch.concat([q_b, q_a], dim=2)
        k = torch.concat([k_b, k_a], dim=2)
        v = torch.concat([v_b, v_a], dim=2)
        q, k = self.apply_rope(q, k, image_rotary_emb)
        hidden_states = flash_attention(q, k, v, causal=True, attn_mask=attn_mask)

        if get_parallel_state().ulysses_enabled:
            hidden_states = gather_heads_scatter_seq(hidden_states, seq_dim=2, head_dim=1)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states_b, hidden_states_a = (
            hidden_states[:, : hidden_states_b.shape[1]],
            hidden_states[:, hidden_states_b.shape[1] :],
        )
        if ipadapter_kwargs_list is not None:
            hidden_states_a = interact_with_ipadapter(hidden_states_a, q_a, **ipadapter_kwargs_list)

        hidden_states_a = self.a_to_out(hidden_states_a)
        if self.only_out_a:
            return hidden_states_a
        else:
            hidden_states_b = self.b_to_out(hidden_states_b)
            return hidden_states_a, hidden_states_b


class FluxJointTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim)
        self.norm1_b = AdaLayerNorm(dim)

        self.attn = FluxJointAttention(dim, dim, num_attention_heads, dim // num_attention_heads)

        self.norm2_a = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_a = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(dim * 4, dim)
        )

        self.norm2_b = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_b = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(dim * 4, dim)
        )

    def forward(
        self, hidden_states_a, hidden_states_b, temb, image_rotary_emb, attn_mask=None, ipadapter_kwargs_list=None
    ):
        norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(
            hidden_states_a, emb=temb
        )
        norm_hidden_states_b, gate_msa_b, shift_mlp_b, scale_mlp_b, gate_mlp_b = self.norm1_b(
            hidden_states_b, emb=temb
        )

        # Attention
        attn_output_a, attn_output_b = self.attn(
            norm_hidden_states_a, norm_hidden_states_b, image_rotary_emb, attn_mask, ipadapter_kwargs_list
        )

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a(norm_hidden_states_a)

        # Part B
        hidden_states_b = hidden_states_b + gate_msa_b * attn_output_b
        norm_hidden_states_b = self.norm2_b(hidden_states_b) * (1 + scale_mlp_b) + shift_mlp_b
        hidden_states_b = hidden_states_b + gate_mlp_b * self.ff_b(norm_hidden_states_b)

        return hidden_states_a, hidden_states_b


class FluxSingleAttention(torch.nn.Module):
    def __init__(self, dim_a, dim_b, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.a_to_qkv = torch.nn.Linear(dim_a, dim_a * 3)

        self.norm_q_a = RMSNorm(head_dim, eps=1e-6)
        self.norm_k_a = RMSNorm(head_dim, eps=1e-6)

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def forward(self, hidden_states, image_rotary_emb):
        batch_size = hidden_states.shape[0]

        qkv_a = self.a_to_qkv(hidden_states)
        qkv_a = qkv_a.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_a, k_a, v = qkv_a.chunk(3, dim=1)
        q_a, k_a = self.norm_q_a(q_a), self.norm_k_a(k_a)

        q, k = self.apply_rope(q_a, k_a, image_rotary_emb)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        return hidden_states


class AdaLayerNormSingle(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(dim, 3 * dim, bias=True)
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class FluxSingleTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        self.dim = dim

        self.norm = AdaLayerNormSingle(dim)
        self.to_qkv_mlp = torch.nn.Linear(dim, dim * (3 + 4))
        self.norm_q_a = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k_a = RMSNorm(self.head_dim, eps=1e-6)

        self.proj_out = torch.nn.Linear(dim * 5, dim)

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def process_attention(self, hidden_states, image_rotary_emb, attn_mask=None, ipadapter_kwargs_list=None):
        batch_size = hidden_states.shape[0]

        qkv = hidden_states.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=1)
        q, k = self.norm_q_a(q), self.norm_k_a(k)

        if get_parallel_state().ulysses_enabled:
            q, k, v = gather_seq_scatter_heads_qkv(q, k, v, seq_dim=2, head_dim=1)

        q, k = self.apply_rope(q, k, image_rotary_emb)

        hidden_states = flash_attention(q, k, v, causal=True, attn_mask=attn_mask)

        if get_parallel_state().ulysses_enabled:
            hidden_states = gather_heads_scatter_seq(hidden_states, seq_dim=2, head_dim=1)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        if ipadapter_kwargs_list is not None:
            hidden_states = interact_with_ipadapter(hidden_states, q, **ipadapter_kwargs_list)
        return hidden_states

    def forward(
        self, hidden_states_a, hidden_states_b, temb, image_rotary_emb, attn_mask=None, ipadapter_kwargs_list=None
    ):
        residual = hidden_states_a
        norm_hidden_states, gate = self.norm(hidden_states_a, emb=temb)
        hidden_states_a = self.to_qkv_mlp(norm_hidden_states)
        attn_output, mlp_hidden_states = hidden_states_a[:, :, : self.dim * 3], hidden_states_a[:, :, self.dim * 3 :]

        attn_output = self.process_attention(attn_output, image_rotary_emb, attn_mask, ipadapter_kwargs_list)
        mlp_hidden_states = torch.nn.functional.gelu(mlp_hidden_states, approximate="tanh")

        hidden_states_a = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states_a = gate.unsqueeze(1) * self.proj_out(hidden_states_a)
        hidden_states_a = residual + hidden_states_a

        return hidden_states_a, hidden_states_b


class AdaLayerNormContinuous(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(dim, dim * 2, bias=True)
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x, conditioning):
        emb = self.linear(self.silu(conditioning))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None] + shift[:, None]
        return x


class FluxModel(PreTrainedModel):
    config_class = FluxConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["FluxJointTransformerBlock", "FluxSingleTransformerBlock"]
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True

    def __init__(self, config: FluxConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.pos_embedder = RoPEEmbedding(3072, 10000, [16, 56, 56])
        self.time_embedder = TimestepEmbeddings(256, 3072)
        self.guidance_embedder = None if config.disable_guidance_embedder else TimestepEmbeddings(256, 3072)
        self.pooled_text_embedder = torch.nn.Sequential(
            torch.nn.Linear(768, 3072), torch.nn.SiLU(), torch.nn.Linear(3072, 3072)
        )
        self.context_embedder = torch.nn.Linear(4096, 3072)
        self.x_embedder = torch.nn.Linear(config.input_dim, 3072)
        self.blocks = torch.nn.ModuleList([FluxJointTransformerBlock(3072, 24) for _ in range(config.num_blocks)])
        self.single_blocks = torch.nn.ModuleList([FluxSingleTransformerBlock(3072, 24) for _ in range(38)])
        self.final_norm_out = AdaLayerNormContinuous(3072)
        self.final_proj_out = torch.nn.Linear(3072, 64)
        self.input_dim = config.input_dim

        self.gradient_checkpointing = False

    def patchify(self, hidden_states):
        hidden_states = rearrange(hidden_states, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        return hidden_states

    def unpatchify(self, hidden_states, height, width):
        if height % 2 != 0 or width % 2 != 0:
            print_rank_0(f"Error: height({height}) or width({width}) not divisible by 2!")

        H = height // 2
        W = width // 2
        C = hidden_states.shape[-1] // (2 * 2)  # 计算推断的通道数

        output = rearrange(hidden_states, "B (H W) (C P Q) -> B C (H P) (W Q)", P=2, Q=2, H=H, W=W, C=C)
        return output

    def prepare_image_ids(self, latents):
        batch_size, _, height, width = latents.shape
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        latent_image_ids = latent_image_ids.to(device=latents.device, dtype=latents.dtype)

        return latent_image_ids

    def tiled_forward(
        self,
        hidden_states,
        timestep,
        prompt_emb,
        pooled_prompt_emb,
        guidance,
        text_ids,
        tile_size=128,
        tile_stride=64,
        **kwargs,
    ):
        hidden_states = TileWorker().tiled_forward(
            lambda x: self.forward(x, timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids, image_ids=None),
            hidden_states,
            tile_size,
            tile_stride,
            tile_device=hidden_states.device,
            tile_dtype=hidden_states.dtype,
        )
        return hidden_states

    def construct_mask(self, entity_masks, prompt_seq_len, image_seq_len):
        N = len(entity_masks)
        batch_size = entity_masks[0].shape[0]
        total_seq_len = N * prompt_seq_len + image_seq_len
        patched_masks = [self.patchify(entity_masks[i]) for i in range(N)]
        attention_mask = torch.ones((batch_size, total_seq_len, total_seq_len), dtype=torch.bool).to(
            device=entity_masks[0].device
        )

        image_start = N * prompt_seq_len
        image_end = N * prompt_seq_len + image_seq_len
        for i in range(N):
            prompt_start = i * prompt_seq_len
            prompt_end = (i + 1) * prompt_seq_len
            image_mask = torch.sum(patched_masks[i], dim=-1) > 0
            image_mask = image_mask.unsqueeze(1).repeat(1, prompt_seq_len, 1)
            attention_mask[:, prompt_start:prompt_end, image_start:image_end] = image_mask
            attention_mask[:, image_start:image_end, prompt_start:prompt_end] = image_mask.transpose(1, 2)
        for i in range(N):
            for j in range(N):
                if i != j:
                    prompt_start_i = i * prompt_seq_len
                    prompt_end_i = (i + 1) * prompt_seq_len
                    prompt_start_j = j * prompt_seq_len
                    prompt_end_j = (j + 1) * prompt_seq_len
                    attention_mask[:, prompt_start_i:prompt_end_i, prompt_start_j:prompt_end_j] = False

        attention_mask = attention_mask.float()
        attention_mask[attention_mask == 0] = float("-inf")
        attention_mask[attention_mask == 1] = 0
        return attention_mask

    def process_entity_masks(self, hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids):
        repeat_dim = hidden_states.shape[1]
        max_masks = 0
        attention_mask = None
        prompt_embs = [prompt_emb]
        if entity_masks is not None:
            # entity_masks
            max_masks = entity_masks.shape[1]
            entity_masks = entity_masks.repeat(1, 1, repeat_dim, 1, 1)
            entity_masks = [entity_masks[:, i, None].squeeze(1) for i in range(max_masks)]
            # global mask
            global_mask = torch.ones_like(entity_masks[0]).to(device=hidden_states.device, dtype=hidden_states.dtype)
            entity_masks = entity_masks + [global_mask]  # append global to last
            # attention mask
            attention_mask = self.construct_mask(entity_masks, prompt_emb.shape[1], hidden_states.shape[1])
            attention_mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
            attention_mask = attention_mask.unsqueeze(1)
            # embds: n_masks * b * seq * d
            local_embs = [entity_prompt_emb[:, i, None].squeeze(1) for i in range(max_masks)]
            prompt_embs = local_embs + prompt_embs  # append global to last
        prompt_embs = [self.context_embedder(prompt_emb) for prompt_emb in prompt_embs]
        prompt_emb = torch.cat(prompt_embs, dim=1)

        # positional embedding
        text_ids = torch.cat([text_ids] * (max_masks + 1), dim=1)
        image_rotary_emb = self.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
        return prompt_emb, image_rotary_emb, attention_mask

    def prepare_extra_input(self, latents=None, guidance=1.0):
        latent_image_ids = self.prepare_image_ids(latents)
        guidance = torch.Tensor([guidance] * latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
        return {"image_ids": latent_image_ids, "guidance": guidance}

    def forward(
        self,
        hidden_states,
        timestep,
        prompt_emb,
        pooled_prompt_emb,
        guidance,
        text_ids,
        image_ids=None,
        tiled=False,
        tile_size=128,
        tile_stride=64,
        entity_prompt_emb=None,
        entity_masks=None,
        **kwargs,
    ):
        if tiled:
            return self.tiled_forward(
                hidden_states,
                timestep,
                prompt_emb,
                pooled_prompt_emb,
                guidance,
                text_ids,
                tile_size=tile_size,
                tile_stride=tile_stride,
                **kwargs,
            )

        if image_ids is None:
            image_ids = self.prepare_image_ids(hidden_states)

        tmp_time_embedder = self.time_embedder(timestep, hidden_states.dtype)
        pooled_prompt_emb = pooled_prompt_emb.to(hidden_states.dtype)
        tmp_pooled_text_embedder = self.pooled_text_embedder(pooled_prompt_emb)
        conditioning = tmp_time_embedder + tmp_pooled_text_embedder
        if self.guidance_embedder is not None:
            guidance = guidance * 1000
            conditioning = conditioning + self.guidance_embedder(guidance, hidden_states.dtype)

        height, width = hidden_states.shape[-2:]
        hidden_states = self.patchify(hidden_states)
        hidden_states = self.x_embedder(hidden_states)

        if entity_prompt_emb is not None and entity_masks is not None:
            prompt_emb, image_rotary_emb, attention_mask = self.process_entity_masks(
                hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids
            )
        else:
            prompt_emb = self.context_embedder(prompt_emb)
            image_rotary_emb = self.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
            attention_mask = None

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        if get_parallel_state().ulysses_enabled:
            hidden_states = slice_input_tensor(hidden_states, dim=1)
            prompt_emb = slice_input_tensor(prompt_emb, dim=1)
            # image_rotary_emb = slice_input_tensor(image_rotary_emb, dim=2)

        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                hidden_states, prompt_emb = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    prompt_emb,
                    conditioning,
                    image_rotary_emb,
                    attention_mask,
                )
            else:
                hidden_states, prompt_emb = block(
                    hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask
                )

        if get_parallel_state().ulysses_enabled:
            hidden_states = gather_outputs(hidden_states, gather_dim=1)
            prompt_emb = gather_outputs(prompt_emb, gather_dim=1)

        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)

        if get_parallel_state().ulysses_enabled:
            hidden_states = slice_input_tensor(hidden_states, dim=1)

        for block in self.single_blocks:
            if self.training and self.gradient_checkpointing:
                hidden_states, prompt_emb = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    prompt_emb,
                    conditioning,
                    image_rotary_emb,
                    attention_mask,
                )
            else:
                hidden_states, prompt_emb = block(
                    hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask
                )

        if get_parallel_state().ulysses_enabled:
            hidden_states = gather_outputs(hidden_states, gather_dim=1)

        hidden_states = hidden_states[:, prompt_emb.shape[1] :]
        hidden_states = self.final_norm_out(hidden_states, conditioning)
        hidden_states = self.final_proj_out(hidden_states)
        hidden_states = self.unpatchify(hidden_states, height, width)
        return hidden_states

    def quantize(self):
        def cast_to(weight, dtype=None, device=None, copy=False):
            if device is None or weight.device == device:
                if not copy:
                    if dtype is None or weight.dtype == dtype:
                        return weight
                return weight.to(dtype=dtype, copy=copy)

            r = torch.empty_like(weight, dtype=dtype, device=device)
            r.copy_(weight)
            return r

        def cast_weight(s, input=None, dtype=None, device=None):
            if input is not None:
                if dtype is None:
                    dtype = input.dtype
                if device is None:
                    device = input.device
            weight = cast_to(s.weight, dtype, device)
            return weight

        def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
            if input is not None:
                if dtype is None:
                    dtype = input.dtype
                if bias_dtype is None:
                    bias_dtype = dtype
                if device is None:
                    device = input.device
            bias = None
            weight = cast_to(s.weight, dtype, device)
            bias = cast_to(s.bias, bias_dtype, device)
            return weight, bias

        class quantized_layer:
            class Linear(torch.nn.Linear):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

                def forward(self, input, **kwargs):
                    weight, bias = cast_bias_weight(self, input)
                    return torch.nn.functional.linear(input, weight, bias)

            class RMSNorm(torch.nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module

                def forward(self, hidden_states, **kwargs):
                    weight = cast_weight(self.module, hidden_states)
                    input_dtype = hidden_states.dtype
                    variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + self.module.eps)
                    hidden_states = hidden_states.to(input_dtype) * weight
                    return hidden_states

        def replace_layer(model):
            for name, module in model.named_children():
                if isinstance(module, torch.nn.Linear):
                    with init_weights_on_device():
                        new_layer = quantized_layer.Linear(module.in_features, module.out_features)
                    new_layer.weight = module.weight
                    if module.bias is not None:
                        new_layer.bias = module.bias
                    # del module
                    setattr(model, name, new_layer)
                elif isinstance(module, RMSNorm):
                    if hasattr(module, "quantized"):
                        continue
                    module.quantized = True
                    new_layer = quantized_layer.RMSNorm(module)
                    setattr(model, name, new_layer)
                else:
                    replace_layer(module)

        replace_layer(self)

    @staticmethod
    def state_dict_converter():
        return FluxDiTStateDictConverter()


if is_liger_kernel_available():
    RMSNorm = LigerRMSNorm
    logger.info_rank0("Apply liger kernel to Flux.")
