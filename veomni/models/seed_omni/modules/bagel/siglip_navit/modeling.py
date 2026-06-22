"""BAGEL SigLIP NaViT vision encoder.

The training node is ``bagel_siglip_navit.forward``. It receives local
patchified image tensors from ``modulemixin.py``, runs SigLIP/NaViT attention,
projects features to the Bagel MoT hidden size, adds visual position embeddings,
and returns ``{"image_embeds": ..., "token_lens": ...}``. The post-hook writes
those feature spans back to the image ``ConversationItem.value`` fields.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .configuration import BagelSiglipNavitConfig
from .modulemixin import BagelSiglipNavitModuleMixin, BagelSiglipNavitTraceMixin
from .processing import BagelSiglipNavitProcessor


class BagelSiglipNavit(BagelSiglipNavitModuleMixin, BagelSiglipNavitTraceMixin, PreTrainedModel):
    config_class = BagelSiglipNavitConfig
    image_processor_class = BagelSiglipNavitProcessor
    base_model_prefix = "bagel_siglip_navit"
    main_input_name = "patchified_pixel_values"
    _no_split_modules = ["BagelSiglipEncoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: BagelSiglipNavitConfig) -> None:
        super().__init__(config)
        self.vision_model = BagelSiglipVisionTransformer(config)
        self.connector = MLPConnector(config.hidden_size, config.output_size, config.connector_act)
        self.vit_pos_embed = PositionEmbedding(config.vit_max_num_patch_per_side, config.output_size)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, PositionEmbedding):
            module.reset_parameters()
            return
        super()._init_weights(module)

    def forward(  # type: ignore[override]
        self,
        patchified_pixel_values: torch.Tensor | None = None,
        patchified_position_ids: torch.LongTensor | None = None,
        cu_seqlens: torch.IntTensor | None = None,
        max_seqlen: int | None = None,
        token_lens: torch.IntTensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs

        if patchified_pixel_values is None:
            dummy = self.dummy_inputs()
            outputs = self.forward(**dummy)
            outputs["is_dummy"] = True
            return outputs

        if patchified_position_ids is None or cu_seqlens is None or max_seqlen is None:
            raise ValueError("BagelSiglipNavit.forward requires position ids, cu_seqlens, and max_seqlen.")

        vit_device = self.vision_model.embeddings.patch_embedding.weight.device
        patchified_pixel_values = patchified_pixel_values.to(device=vit_device, dtype=self.dtype)
        patchified_position_ids = patchified_position_ids.to(device=vit_device, dtype=torch.long)
        cu_seqlens = cu_seqlens.to(device=vit_device, dtype=torch.int32)
        vit_hidden = self.vision_model(
            patchified_pixel_values=patchified_pixel_values,
            patchified_position_ids=patchified_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        connector_device = self.connector.fc1.weight.device
        vit_hidden = vit_hidden.to(device=connector_device, dtype=self.connector.fc1.weight.dtype)
        patchified_position_ids = patchified_position_ids.to(device=self.vit_pos_embed.pos_embed.device)
        image_embeds = self.connector(vit_hidden)
        image_embeds = image_embeds + self.vit_pos_embed(patchified_position_ids).to(
            device=image_embeds.device,
            dtype=image_embeds.dtype,
        )
        return {"image_embeds": image_embeds, "token_lens": token_lens}


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim: int, max_h: int, max_w: int, base: int = 10000) -> None:
        super().__init__()
        freq = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        inv_freq = 1.0 / (base**freq)

        grid_h = torch.arange(0, max_h).to(inv_freq.dtype)
        grid_h = grid_h[:, None].repeat(1, max_w)

        grid_w = torch.arange(0, max_w).to(inv_freq.dtype)
        grid_w = grid_w[None, :].repeat(max_h, 1)

        cos_h, sin_h = self._forward_one_side(grid_h, inv_freq)
        cos_w, sin_w = self._forward_one_side(grid_w, inv_freq)

        self.register_buffer("cos_h", cos_h, persistent=False)
        self.register_buffer("sin_h", sin_h, persistent=False)
        self.register_buffer("cos_w", cos_w, persistent=False)
        self.register_buffer("sin_w", sin_w, persistent=False)

    @staticmethod
    def _forward_one_side(grid: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    cos = cos.to(device=q.device, dtype=q.dtype).unsqueeze(1)
    sin = sin.to(device=q.device, dtype=q.dtype).unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side: int, hidden_size: int) -> None:
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side**2, hidden_size),
            requires_grad=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.pos_embed.data.copy_(_get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side))

    def forward(self, position_ids: torch.LongTensor) -> torch.Tensor:
        return self.pos_embed[position_ids.to(device=self.pos_embed.device)]


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.from_numpy(np.concatenate([emb_h, emb_w], axis=1)).float()


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class MLPConnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class BagelSiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig) -> None:
        super().__init__()
        patch_dim = config.num_channels * config.patch_size * config.patch_size
        self.patch_embedding = nn.Linear(patch_dim, config.hidden_size, bias=True)
        self.position_embedding = (
            None if config.rope else nn.Embedding((config.image_size // config.patch_size) ** 2, config.hidden_size)
        )

    def forward(
        self,
        patchified_pixel_values: torch.Tensor,
        patchified_position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        patchified_pixel_values = patchified_pixel_values.to(
            device=self.patch_embedding.weight.device,
            dtype=self.patch_embedding.weight.dtype,
        )
        patch_embeds = self.patch_embedding(patchified_pixel_values)
        if self.position_embedding is None:
            return patch_embeds
        position_ids = patchified_position_ids.to(device=self.position_embedding.weight.device, dtype=torch.long)
        position_embeds = self.position_embedding(position_ids)
        return patch_embeds + position_embeds.to(device=patch_embeds.device, dtype=patch_embeds.dtype)


class BagelSiglipAttention(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig) -> None:
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
        cos_h: torch.Tensor | None = None,
        sin_h: torch.Tensor | None = None,
        cos_w: torch.Tensor | None = None,
        sin_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total_q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)

        if self.config.rope:
            if cos_h is None or sin_h is None or cos_w is None or sin_w is None:
                raise ValueError("2D RoPE tensors are required when BagelSiglipNavitConfig.rope=True.")
            qh, qw = query_states[:, :, : self.head_dim // 2], query_states[:, :, self.head_dim // 2 :]
            kh, kw = key_states[:, :, : self.head_dim // 2], key_states[:, :, self.head_dim // 2 :]
            qh, kh = _apply_rotary_pos_emb(qh, kh, cos_h, sin_h)
            qw, kw = _apply_rotary_pos_emb(qw, kw, cos_w, sin_w)
            query_states = torch.cat([qh, qw], dim=-1)
            key_states = torch.cat([kh, kw], dim=-1)

        if not query_states.is_cuda:
            raise RuntimeError("BagelSiglipNavit attention requires CUDA flash-attn.")
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
        return self.out_proj(attn_output.reshape(total_q_len, -1).to(hidden_states.dtype))


class BagelSiglipMLP(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class BagelSiglipEncoderLayer(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig) -> None:
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
        cos_h: torch.Tensor | None = None,
        sin_h: torch.Tensor | None = None,
        cos_w: torch.Tensor | None = None,
        sin_w: torch.Tensor | None = None,
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
    def __init__(self, config: BagelSiglipNavitConfig) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([BagelSiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor | None = None,
        sin_h: torch.Tensor | None = None,
        cos_w: torch.Tensor | None = None,
        sin_w: torch.Tensor | None = None,
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
    def __init__(self, config: BagelSiglipNavitConfig) -> None:
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
        patchified_pixel_values: torch.Tensor,
        patchified_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            patchified_pixel_values=patchified_pixel_values,
            patchified_position_ids=patchified_position_ids,
        )
        extra_inputs: dict[str, torch.Tensor] = {}
        if self.config.rope:
            rope_position_ids = patchified_position_ids.to(device=self.rope.cos_h.device, dtype=torch.long)
            extra_inputs.update(
                cos_h=self.rope.cos_h[rope_position_ids],
                sin_h=self.rope.sin_h[rope_position_ids],
                cos_w=self.rope.cos_w[rope_position_ids],
                sin_w=self.rope.sin_w[rope_position_ids],
            )
        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **extra_inputs,
        )
        return self.post_layernorm(last_hidden_state)


__all__ = [
    "BagelSiglipNavit",
    "BagelSiglipNavitConfig",
    "MLPConnector",
    "PositionEmbedding",
]
