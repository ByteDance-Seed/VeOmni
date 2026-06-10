"""BAGEL SigLIP NaViT and visual connector structural module.

The weight-owning architecture is present for checkpoint splitting. Runtime
vision forward parity is intentionally left for the Bagel graph/parity phase.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .configuration import BagelSiglipNavitConfig
from .modulemixin import BagelSiglipNavitModuleMixin


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
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float(), persistent=False)

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
        self.position_embedding = nn.Embedding((config.image_size // config.patch_size) ** 2, config.hidden_size)

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        return self.patch_embedding(packed_pixel_values) + self.position_embedding(packed_flattened_position_ids)


class BagelSiglipAttention(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        # TODO(bagel-v2): port official SigLIP FlashAttention/RoPE attention here.
        raise NotImplementedError("BagelSiglipAttention forward is not implemented yet.")


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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        del hidden_states
        # TODO(bagel-v2): port official SigLIP encoder layer attention + MLP residuals here.
        raise NotImplementedError("BagelSiglipEncoderLayer forward is not implemented yet.")


class BagelSiglipEncoder(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig):
        super().__init__()
        self.layers = nn.ModuleList([BagelSiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        del hidden_states
        # TODO(bagel-v2): port official packed SigLIP encoder loop here.
        raise NotImplementedError("BagelSiglipEncoder forward is not implemented yet.")


class BagelSiglipVisionTransformer(nn.Module):
    def __init__(self, config: BagelSiglipNavitConfig):
        super().__init__()
        self.embeddings = BagelSiglipVisionEmbeddings(config)
        self.encoder = BagelSiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        del packed_pixel_values, packed_flattened_position_ids
        # TODO(bagel-v2): port official NaViT packed forward including cu_seqlens/max_seqlen.
        raise NotImplementedError("BagelSiglipVisionTransformer forward is not implemented yet.")


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

    def forward(  # type: ignore[override]
        self,
        packed_pixel_values: Optional[torch.Tensor] = None,
        packed_flattened_position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del packed_pixel_values, packed_flattened_position_ids, kwargs
        # TODO(bagel-v2): port official SigLIP NaViT attention/packing forward here.
        raise NotImplementedError("BagelSiglipNavit forward is not implemented yet.")
