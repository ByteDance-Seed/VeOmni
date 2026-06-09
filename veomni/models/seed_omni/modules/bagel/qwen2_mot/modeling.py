"""BAGEL Qwen2 MoT backbone.

This first SeedOmni module owns the checkpoint parameters for BAGEL's MoT
decoder blocks while leaving token embeddings and LM head to
``bagel_text_encoder``. Runtime forward parity is intentionally not implemented
yet; this module exists first as the loadable architecture target for the
module-aware splitter.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm

from .configuration import BagelQwen2MoTConfig
from .modulemixin import BagelQwen2MoTModuleMixin


class BagelQwen2MoTAttention(nn.Module):
    def __init__(self, config: BagelQwen2MoTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)


class BagelQwen2MoTDecoderLayer(nn.Module):
    def __init__(self, config: BagelQwen2MoTConfig):
        super().__init__()
        self.self_attn = BagelQwen2MoTAttention(config)
        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class BagelQwen2MoTBackbone(nn.Module):
    def __init__(self, config: BagelQwen2MoTConfig):
        super().__init__()
        self.layers = nn.ModuleList([BagelQwen2MoTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class BagelQwen2MoT(BagelQwen2MoTModuleMixin, PreTrainedModel):
    config_class = BagelQwen2MoTConfig
    base_model_prefix = "bagel_qwen2_mot"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["BagelQwen2MoTDecoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: BagelQwen2MoTConfig):
        super().__init__(config)
        self.model = BagelQwen2MoTBackbone(config)
        self.post_init()

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del inputs_embeds, kwargs
        # TODO(bagel-v2): port official Qwen2 MoT packed forward/inference here.
        raise NotImplementedError("BagelQwen2MoT forward is not implemented yet.")
