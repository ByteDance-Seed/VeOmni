import numpy as np
import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from transformers.activations import ACT2FN
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipPreTrainedModel

from ..base import BaseEncoderModelMixin
from .configuration_bagel_vision_encoder import BagelVisionEncoderConfig


class RotaryEmbedding2D(torch.nn.Module):
    def __init__(self, dim, max_h, max_w, base=10000):
        super().__init__()
        freq = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        inv_freq = 1.0 / (base**freq)

        grid_h = torch.arange(0, max_h)
        grid_h = grid_h.to(inv_freq.dtype)
        grid_h = grid_h[:, None].repeat(1, max_w)

        grid_w = torch.arange(0, max_w)
        grid_w = grid_w.to(inv_freq.dtype)
        grid_w = grid_w[None, :].repeat(max_h, 1)

        cos_h, sin_h = self._forward_one_side(grid_h, inv_freq)
        cos_w, sin_w = self._forward_one_side(grid_w, inv_freq)

        self.register_buffer("cos_h", cos_h)
        self.register_buffer("sin_h", sin_h)
        self.register_buffer("cos_w", cos_w)
        self.register_buffer("sin_w", sin_w)

    def _forward_one_side(self, grid, inv_freq):
        freqs = grid[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1).flatten(0, 1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # unsqueeze due to the head dimension
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: BagelVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        if not config.rope:
            self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.convert_conv2d_to_linear()

    def convert_conv2d_to_linear(self, meta=False):
        if meta:
            linear_patch_embedding = nn.Linear(
                self.config.num_channels * self.patch_size**2, self.embed_dim, bias=True, device="meta"
            )
        else:
            linear_patch_embedding = nn.Linear(
                self.config.num_channels * self.patch_size**2, self.embed_dim, bias=True
            )
        W = self.patch_embedding.weight.permute(0, 2, 3, 1).reshape(
            self.embed_dim, self.config.num_channels * self.patch_size**2
        )
        linear_patch_embedding.weight.data = W
        linear_patch_embedding.bias.data = self.patch_embedding.bias.data
        del self.patch_embedding
        self.patch_embedding = linear_patch_embedding

    def forward(
        self, packed_pixel_values: torch.FloatTensor, packed_flattened_position_ids: torch.LongTensor
    ) -> torch.Tensor:
        patch_embeds = self.patch_embedding(packed_pixel_values)
        if not self.config.rope:
            embeddings = patch_embeds + self.position_embedding(packed_flattened_position_ids)
        else:
            embeddings = patch_embeds
        return embeddings


class SiglipFlashAttention2(SiglipAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        total_q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(total_q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(total_q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(total_q_len, self.num_heads, self.head_dim)

        if self.config.rope:
            qh, qw = query_states[:, :, : self.head_dim // 2], query_states[:, :, self.head_dim // 2 :]
            kh, kw = key_states[:, :, : self.head_dim // 2], key_states[:, :, self.head_dim // 2 :]
            qh, kh = apply_rotary_pos_emb(qh, kh, cos_h, sin_h)
            qw, kw = apply_rotary_pos_emb(qw, kw, cos_w, sin_w)
            query_states = torch.cat([qh, qw], dim=-1)
            key_states = torch.cat([kh, kw], dim=-1)

        attn_output = flash_attn_varlen_func(
            query_states.to(torch.bfloat16),
            key_states.to(torch.bfloat16),
            value_states.to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )

        attn_output = self.out_proj(attn_output.reshape(total_q_len, -1))
        return attn_output


class SiglipMLP(nn.Module):
    def __init__(self, config: BagelVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: BagelVisionEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipFlashAttention2(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cos_h=cos_h,
            sin_h=sin_h,
            cos_w=cos_w,
            sin_w=sin_w,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: BagelVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states, cu_seqlens, max_seqlen, cos_h=cos_h, sin_h=sin_h, cos_w=cos_w, sin_w=sin_w
            )

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: BagelVisionEncoderConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        if config.rope:
            max_size = config.image_size // config.patch_size
            dim_head = config.hidden_size // config.num_attention_heads
            self.rope = RotaryEmbedding2D(dim_head // 2, max_size, max_size)

        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            packed_pixel_values=packed_pixel_values, packed_flattened_position_ids=packed_flattened_position_ids
        )

        extra_inputs = {}
        if self.config.rope:
            extra_inputs.update(
                cos_h=self.rope.cos_h[packed_flattened_position_ids],
                sin_h=self.rope.sin_h[packed_flattened_position_ids],
                cos_w=self.rope.cos_w[packed_flattened_position_ids],
                sin_w=self.rope.sin_w[packed_flattened_position_ids],
            )

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **extra_inputs
        )
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side, hidden_size):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size

        # 生成 sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)

        # 注册为 buffer，不会参与训练，也不会出现在 state_dict
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed).float(),
            persistent=False,  # 不保存到 state_dict
        )

    def forward(self, position_ids):
        return self.pos_embed[position_ids]


class MLPconnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BagelVisionEncoder(BaseEncoderModelMixin, SiglipPreTrainedModel):
    config_class = BagelVisionEncoderConfig
    main_input_name = "packed_pixel_values"
    _no_split_modules = ["SiglipEncoderLayer"]

    def __init__(self, config: BagelVisionEncoderConfig):
        super().__init__(config)

        self.vision_model = SiglipVisionTransformer(config)
        self.vit_pos_embed = PositionEmbedding(config.vit_max_num_patch_per_side, config.output_size)
        self.connector = MLPconnector(config.hidden_size, config.output_size, config.connector_act)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        packed_vit_token_embed = self.vision_model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_flattened_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        vit_token_pos_emb = self.vit_pos_embed(packed_flattened_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
        return packed_vit_token_embed
