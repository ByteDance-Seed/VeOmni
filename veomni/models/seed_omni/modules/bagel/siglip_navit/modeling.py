"""BAGEL SigLIP NaViT and visual connector structural module.

The weight-owning architecture is present for checkpoint splitting. Runtime
vision forward parity is intentionally left for the Bagel graph/parity phase.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .configuration import BagelSiglipNavitConfig
from .modulemixin import BagelSiglipNavitModuleMixin


class RotaryEmbedding2D(torch.nn.Module):
    def __init__(self, dim: int, max_h: int, max_w: int, base: int = 10000):
        super().__init__()
        freq = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        inv_freq = 1.0 / (base**freq)

        grid_h = torch.arange(0, max_h).to(inv_freq.dtype)
        grid_h = grid_h[:, None].repeat(1, max_w)

        grid_w = torch.arange(0, max_w).to(inv_freq.dtype)
        grid_w = grid_w[None, :].repeat(max_h, 1)

        cos_h, sin_h = self._forward_one_side(grid_h, inv_freq)
        cos_w, sin_w = self._forward_one_side(grid_w, inv_freq)

        self.register_buffer("cos_h", cos_h)
        self.register_buffer("sin_h", sin_h)
        self.register_buffer("cos_w", cos_w)
        self.register_buffer("sin_w", sin_w)

    def _forward_one_side(self, grid: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = grid[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1).flatten(0, 1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side: int, hidden_size: int):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side**2, hidden_size),
            requires_grad=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids: torch.LongTensor) -> torch.Tensor:
        return self.pos_embed[position_ids]


class MLPconnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class BagelSiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig):
        super().__init__()
        patch_dim = config.num_channels * config.patch_size * config.patch_size
        self.patch_embedding = nn.Linear(patch_dim, config.hidden_size, bias=True)
        self.position_embedding = (
            None if config.rope else nn.Embedding((config.image_size // config.patch_size) ** 2, config.hidden_size)
        )

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        patch_embeds = self.patch_embedding(packed_pixel_values)
        if self.position_embedding is None:
            return patch_embeds
        return patch_embeds + self.position_embedding(packed_flattened_position_ids)


class BagelSiglipAttention(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: Optional[torch.Tensor] = None,
        sin_h: Optional[torch.Tensor] = None,
        cos_w: Optional[torch.Tensor] = None,
        sin_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total_q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(total_q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(total_q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(total_q_len, self.num_heads, self.head_dim)

        if self.config.rope:
            if cos_h is None or sin_h is None or cos_w is None or sin_w is None:
                raise ValueError("2D RoPE tensors are required when BagelSiglipNavitConfig.rope=True.")
            qh, qw = query_states[:, :, : self.head_dim // 2], query_states[:, :, self.head_dim // 2 :]
            kh, kw = key_states[:, :, : self.head_dim // 2], key_states[:, :, self.head_dim // 2 :]
            qh, kh = _apply_rotary_pos_emb(qh, kh, cos_h, sin_h)
            qw, kw = _apply_rotary_pos_emb(qw, kw, cos_w, sin_w)
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
        return self.out_proj(attn_output.reshape(total_q_len, -1))


class BagelSiglipMLP(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class BagelSiglipEncoderLayer(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig):
        super().__init__()
        self.self_attn = BagelSiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = BagelSiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: Optional[torch.Tensor] = None,
        sin_h: Optional[torch.Tensor] = None,
        cos_w: Optional[torch.Tensor] = None,
        sin_w: Optional[torch.Tensor] = None,
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
        return residual + hidden_states


class BagelSiglipEncoder(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig):
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([BagelSiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: Optional[torch.Tensor] = None,
        sin_h: Optional[torch.Tensor] = None,
        cos_w: Optional[torch.Tensor] = None,
        sin_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    cos_h,
                    sin_h,
                    cos_w,
                    sin_w,
                )
            else:
                hidden_states = encoder_layer(
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    cos_h=cos_h,
                    sin_h=sin_h,
                    cos_w=cos_w,
                    sin_w=sin_w,
                )
        return hidden_states


class BagelSiglipVisionTransformer(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig):
        super().__init__()
        self.config = config
        self.embeddings = BagelSiglipVisionEmbeddings(config)
        if config.rope:
            max_size = config.image_size // config.patch_size
            dim_head = config.hidden_size // config.num_attention_heads
            self.rope = RotaryEmbedding2D(dim_head // 2, max_size, max_size)
        self.encoder = BagelSiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_flattened_position_ids,
        )
        extra_inputs: dict[str, torch.Tensor] = {}
        if self.config.rope:
            extra_inputs.update(
                cos_h=self.rope.cos_h[packed_flattened_position_ids],
                sin_h=self.rope.sin_h[packed_flattened_position_ids],
                cos_w=self.rope.cos_w[packed_flattened_position_ids],
                sin_w=self.rope.sin_w[packed_flattened_position_ids],
            )
        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **extra_inputs,
        )
        return self.post_layernorm(last_hidden_state)


class BagelSiglipNavit(BagelSiglipNavitModuleMixin, PreTrainedModel):
    config_class = BagelSiglipNavitConfig
    base_model_prefix = "bagel_siglip_navit"
    main_input_name = "packed_pixel_values"
    _no_split_modules = ["BagelSiglipEncoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: BagelSiglipNavitConfig):
        super().__init__(config)
        self.vision_model = BagelSiglipVisionTransformer(config)
        self.connector = MLPconnector(config.hidden_size, config.output_size, config.connector_act)
        self.vit_pos_embed = PositionEmbedding(config.vit_max_num_patch_per_side, config.output_size)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, PositionEmbedding):
            module._init_weights()
            return
        super()._init_weights(module)

    def forward(  # type: ignore[override]
        self,
        packed_pixel_values: Optional[torch.Tensor] = None,
        packed_flattened_position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if packed_pixel_values is None or packed_flattened_position_ids is None:
            raise ValueError(
                "BagelSiglipNavit.forward requires packed_pixel_values and packed_flattened_position_ids."
            )
        if cu_seqlens is None or max_seqlen is None:
            raise ValueError("BagelSiglipNavit.forward requires cu_seqlens and max_seqlen.")
        packed_vit_token_embed = self.vision_model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_flattened_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        vit_token_pos_emb = self.vit_pos_embed(packed_flattened_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
        return {"image_embeds": packed_vit_token_embed}
