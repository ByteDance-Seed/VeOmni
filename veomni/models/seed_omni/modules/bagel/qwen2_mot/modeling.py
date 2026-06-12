"""BAGEL Qwen2 MoT backbone."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from transformers import PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from transformers.utils import ModelOutput

from .configuration import BagelQwen2MoTConfig
from .modulemixin import BagelQwen2MoTModuleMixin


class NaiveCache:
    """Official BAGEL packed KV cache."""

    def __init__(self, num_layers: int):
        self.key_cache = dict.fromkeys(range(num_layers))
        self.value_cache = dict.fromkeys(range(num_layers))

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)

    @property
    def seq_lens(self) -> int:
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        return 0


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor | None = None
    past_key_values: Optional[NaiveCache] = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _check_packed_inference_mode(
    mode: str,
    *,
    packed_vae_token_indexes: Optional[torch.Tensor],
    packed_text_indexes: Optional[torch.Tensor],
) -> bool:
    if mode == "und":
        return False
    if mode == "gen":
        if packed_text_indexes is None or packed_vae_token_indexes is None:
            raise ValueError("mode='gen' requires packed_text_indexes and packed_vae_token_indexes.")
        return True
    raise ValueError(f"Unsupported BAGEL Qwen2 MoT inference mode: {mode!r}")


class BagelQwen2RotaryEmbedding(nn.Module):
    """Official-compatible Qwen2 RoPE for BAGEL parity."""

    def __init__(self, config: BagelQwen2MoTConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_type = "default"
        self.attention_scaling = 1.0
        inv_freq, _ = self.compute_default_rope_parameters(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: BagelQwen2MoTConfig,
        device: Optional[torch.device] = None,
        seq_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)
        base = getattr(config, "rope_theta", None)
        if base is None:
            base = config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BagelQwen2MoTAttention(nn.Module):
    def __init__(self, config: BagelQwen2MoTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
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

    def _forward_packed_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: list[int],
        attention_mask: Any,
        packed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
    ) -> torch.Tensor:
        packed_query_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_heads * self.head_dim))
        packed_key_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )
        packed_value_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )

        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(packed_sequence_gen)
        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(packed_sequence_gen)
        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
        packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(packed_sequence_gen)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)
        packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
        packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(
            packed_query_states[packed_gen_token_indexes]
        )
        packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
        packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes])

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = _apply_rotary_pos_emb(
            packed_query_states_,
            packed_key_states_,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        if isinstance(attention_mask, list):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(
                1, 1, self.num_heads // self.num_key_value_heads, 1
            )
            packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(
                1,
                1,
                self.num_heads // self.num_key_value_heads,
                1,
            )
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states,
                unpacked_key_states,
                unpacked_value_states,
                attention_mask,
                strict=True,
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                unpacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(unpacked_attn_output, dim=1)
        else:
            raise NotImplementedError("BAGEL Qwen2 MoT training currently requires nested attention masks.")

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        packed_attn_output_[packed_und_token_indexes] = self.o_proj(packed_attn_output[packed_und_token_indexes])
        packed_attn_output_[packed_gen_token_indexes] = self.o_proj_moe_gen(
            packed_attn_output[packed_gen_token_indexes]
        )

        return packed_attn_output_

    def _forward_packed_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        is_gen = _check_packed_inference_mode(
            mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        if not is_gen:
            packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        else:
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_query_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_heads * self.head_dim)
            )
            packed_key_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )
            packed_value_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
            packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(packed_vae_query_sequence)
            packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
            packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(packed_vae_query_sequence)
            packed_value_states[packed_text_indexes] = self.v_proj(packed_text_query_sequence)
            packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(packed_vae_query_sequence)

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim).to(torch.float32)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim).to(torch.float32)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
            packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(
                packed_query_states[packed_vae_token_indexes]
            )
            packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
            packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(
                packed_key_states[packed_vae_token_indexes]
            )

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = _apply_rotary_pos_emb(
            packed_query_states,
            packed_key_states,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            if key_values_lens is None or packed_key_value_indexes is None:
                raise ValueError("key_values_lens and packed_key_value_indexes are required when cache is non-empty.")
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = int(query_lens.sum().item() + key_values_lens.sum().item())
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))
        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=int(query_lens.max().item()),
            max_seqlen_k=int(key_values_lens.max().item()),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        if not is_gen:
            packed_attn_output = self.o_proj(packed_attn_output)
        else:
            packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
            packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(
                packed_attn_output[packed_vae_token_indexes]
            )

        if update_past_key_values:
            if past_key_values is None:
                raise ValueError("past_key_values is required when update_past_key_values=True.")
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        if self.training:
            return self._forward_packed_train(*args, **kwargs), None
        return self._forward_packed_inference(*args, **kwargs)


class BagelQwen2MoTDecoderLayer(nn.Module):
    def __init__(self, config: BagelQwen2MoTConfig, layer_idx: int):
        super().__init__()
        self.self_attn = BagelQwen2MoTAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _forward_packed_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: list[int],
        attention_mask: Any,
        packed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
    ) -> torch.Tensor:
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.input_layernorm(packed_sequence[packed_und_token_indexes])
        packed_sequence_[packed_gen_token_indexes] = self.input_layernorm_moe_gen(
            packed_sequence[packed_gen_token_indexes]
        )

        packed_sequence_, _ = self.self_attn(
            packed_sequence=packed_sequence_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        packed_sequence = residual + packed_sequence_

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.mlp(
            self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        )
        packed_sequence_[packed_gen_token_indexes] = self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
        )
        return residual + packed_sequence_

    def _forward_packed_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        is_gen = _check_packed_inference_mode(
            mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        residual = packed_query_sequence
        if not is_gen:
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        else:
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.input_layernorm(
                packed_query_sequence[packed_text_indexes]
            )
            packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        residual = packed_query_sequence
        if not is_gen:
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        else:
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(packed_vae_query_sequence).to(
                torch.bfloat16
            )
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        if self.training:
            return self._forward_packed_train(*args, **kwargs), None
        return self._forward_packed_inference(*args, **kwargs)


class BagelQwen2MoTBackbone(nn.Module):
    def __init__(self, config: BagelQwen2MoTConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [BagelQwen2MoTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BagelQwen2RotaryEmbedding(config=config)
        self.use_moe = "Mo" in config.layer_module

    def _forward_packed_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: list[int],
        attention_mask: Any,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.Tensor] = None,
        packed_gen_token_indexes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        packed_position_embeddings = (cos.squeeze(0), sin.squeeze(0))

        if self.use_moe:
            if packed_und_token_indexes is None:
                raise ValueError("packed_und_token_indexes is required for BAGEL MoT training.")
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
        else:
            packed_und_token_indexes = torch.arange(packed_sequence.shape[0], device=packed_sequence.device)
            packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])

        for decoder_layer in self.layers:
            packed_sequence, _ = decoder_layer(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_und_token_indexes] = self.norm(packed_sequence[packed_und_token_indexes])
            packed_sequence_[packed_gen_token_indexes] = self.norm_moe_gen(packed_sequence[packed_gen_token_indexes])
            return packed_sequence_
        return self.norm(packed_sequence)

    def _forward_packed_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
    ) -> BaseNavitOutputWithPast:
        is_gen = _check_packed_inference_mode(
            mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        if past_key_values is None:
            past_key_values = NaiveCache(len(self.layers))

        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        packed_query_position_embeddings = (cos.squeeze(0), sin.squeeze(0))

        for decoder_layer in self.layers:
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
            )

        if not is_gen:
            packed_query_sequence = self.norm(packed_query_sequence)
        else:
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
            packed_query_sequence = packed_query_sequence_
        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )

    def forward(self, *args: Any, **kwargs: Any) -> BaseNavitOutputWithPast:
        if self.training:
            return BaseNavitOutputWithPast(packed_query_sequence=self._forward_packed_train(*args, **kwargs))
        return self._forward_packed_inference(*args, **kwargs)


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
        packed_query_sequence: Optional[torch.Tensor] = None,
        packed_sequence: Optional[torch.Tensor] = None,
        query_lens: Optional[torch.Tensor] = None,
        packed_query_position_ids: Optional[torch.Tensor] = None,
        packed_query_indexes: Optional[torch.Tensor] = None,
        sample_lens: Optional[list[int]] = None,
        attention_mask: Any = None,
        packed_position_ids: Optional[torch.Tensor] = None,
        packed_und_token_indexes: Optional[torch.Tensor] = None,
        packed_gen_token_indexes: Optional[torch.Tensor] = None,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if sample_lens is not None or attention_mask is not None or packed_position_ids is not None:
            if packed_sequence is None:
                packed_sequence = packed_query_sequence if packed_query_sequence is not None else inputs_embeds
            if packed_sequence is None:
                raise ValueError("BagelQwen2MoT training forward requires packed_sequence or inputs_embeds.")
            if sample_lens is None or attention_mask is None or packed_position_ids is None:
                raise ValueError("sample_lens, attention_mask, and packed_position_ids are required for training.")
            output = self.model(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=packed_position_ids,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )
            hidden_states = output.packed_query_sequence
            return {"hidden_states": hidden_states}

        if packed_query_sequence is None:
            packed_query_sequence = inputs_embeds
        if packed_query_sequence is None:
            raise ValueError("BagelQwen2MoT.forward requires inputs_embeds or packed_query_sequence.")
        if query_lens is None or packed_query_position_ids is None or packed_query_indexes is None:
            raise ValueError("query_lens, packed_query_position_ids, and packed_query_indexes are required.")

        output = self._forward_packed_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        return {
            "hidden_states": output.packed_query_sequence,
            "past_key_values": output.past_key_values,
        }

    def _forward_packed_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> BaseNavitOutputWithPast:
        del kwargs
        is_gen = _check_packed_inference_mode(
            mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        call_kwargs: Dict[str, Any] = {
            "packed_query_sequence": packed_query_sequence,
            "query_lens": query_lens,
            "packed_query_position_ids": packed_query_position_ids,
            "packed_query_indexes": packed_query_indexes,
            "past_key_values": past_key_values,
            "key_values_lens": key_values_lens,
            "packed_key_value_indexes": packed_key_value_indexes,
            "update_past_key_values": update_past_key_values,
            "is_causal": is_causal,
            "mode": mode,
        }
        if is_gen:
            call_kwargs["packed_vae_token_indexes"] = packed_vae_token_indexes
            call_kwargs["packed_text_indexes"] = packed_text_indexes
        return self.model._forward_packed_inference(**call_kwargs)


__all__ = ["BaseNavitOutputWithPast", "BagelQwen2MoT", "NaiveCache"]
