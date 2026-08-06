"""BAGEL Qwen2 MoT backbone."""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask

from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


if IS_NPU_AVAILABLE:
    import torch_npu
from transformers import PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP as TransformersQwen2MLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as TransformersQwen2RMSNorm
from transformers.utils import ModelOutput

from ......ops.dispatch import OpSlot
from ......ops.kernels.attention import fused_attention_forward
from .checkpoint_conversion import (
    BagelQwen2MoTCheckpointTensorConverter,
    combine_qkv_state_dict_pre_hook,
    split_qkv_state_dict_post_hook,
)
from .configuration import BagelQwen2MoTConfig
from .masking import build_mot_block_mask
from .modulemixin import BagelQwen2MoTMetricMeterMixin, BagelQwen2MoTModuleMixin


veomni_rms_norm = OpSlot("rms_norm", "standard")
veomni_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "full")
veomni_swiglu_mlp = OpSlot("swiglu_mlp", "standard")


@contextmanager
def _temporary_attention_implementation(
    config: BagelQwen2MoTConfig,
    implementation: Optional[str],
) -> Iterator[None]:
    if implementation is None:
        yield
        return

    previous = config._attn_implementation
    config._attn_implementation = implementation
    try:
        yield
    finally:
        config._attn_implementation = previous


class BagelQwen2MoT(BagelQwen2MoTModuleMixin, BagelQwen2MoTMetricMeterMixin, PreTrainedModel):
    config_class = BagelQwen2MoTConfig
    base_model_prefix = "bagel_qwen2_mot"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["BagelQwen2MoTDecoderLayer"]
    supports_gradient_checkpointing = True
    _supports_flex_attn = True
    _export_hf_checkpoint_with_weight_conversions = True

    def __init__(self, config: BagelQwen2MoTConfig):
        super().__init__(config)
        self.model = BagelQwen2MoTBackbone(config)
        self.post_init()

    # The runtime stores one merged QKV parameter per MoT branch, while existing
    # HF checkpoints keep separate Q/K/V keys. VeOmni's streaming loader bypasses
    # load_state_dict hooks, so it needs this dedicated checkpoint converter.
    @staticmethod
    def _create_checkpoint_tensor_converter(
        model: PreTrainedModel,
    ) -> BagelQwen2MoTCheckpointTensorConverter:
        del model
        return BagelQwen2MoTCheckpointTensorConverter()

    def forward(  # type: ignore[override]
        self,
        packed_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_token_type_ids: torch.Tensor,
        packed_attention_metadata: torch.Tensor,
    ) -> Dict[str, Any]:
        packed_und_token_indexes = torch.nonzero(packed_token_type_ids == 0, as_tuple=False).flatten()
        packed_gen_token_indexes = torch.nonzero(packed_token_type_ids == 1, as_tuple=False).flatten()
        output = self.model(
            packed_sequence=packed_sequence,
            packed_position_ids=packed_position_ids,
            packed_attention_metadata=packed_attention_metadata,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return {"hidden_states": output.packed_query_sequence}

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional["NaiveCache"] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        attention_implementation: Optional[str] = None,
        packed_attention_metadata: Optional[torch.Tensor] = None,
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
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
            "packed_attention_metadata": packed_attention_metadata,
        }

        is_gen = _check_packed_inference_mode(mode)
        if is_gen:
            call_kwargs["packed_vae_token_indexes"] = packed_vae_token_indexes
            call_kwargs["packed_text_indexes"] = packed_text_indexes

        with _temporary_attention_implementation(self.config, attention_implementation):
            output = self.model._forward_packed_inference(**call_kwargs)

        return {
            "hidden_states": output.packed_query_sequence,
            "past_key_values": output.past_key_values,
        }


class NaiveCache:
    """Official BAGEL packed KV cache."""

    def __init__(self, num_layers: int):
        self.key_cache = dict.fromkeys(range(num_layers))
        self.value_cache = dict.fromkeys(range(num_layers))


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
    if q.numel() != 0 and k.numel() != 0 and veomni_apply_rotary_pos_emb.use_non_eager_impl:
        if unsqueeze_dim != 1:
            raise NotImplementedError("BAGEL packed fused RoPE requires unsqueeze_dim=1.")
        if q.ndim != 3 or k.ndim != 3 or cos.ndim != 2 or sin.ndim != 2:
            raise NotImplementedError(
                "BAGEL fused RoPE requires packed q/k tensors shaped [tokens, heads, head_dim] "
                "and cos/sin tensors shaped [tokens, head_dim]."
            )
        if not (q.shape[0] == k.shape[0] == cos.shape[0] == sin.shape[0]):
            raise ValueError("BAGEL packed q/k/cos/sin tensors must share the token dimension.")
        if not (q.shape[-1] == k.shape[-1] == cos.shape[-1] == sin.shape[-1]):
            raise NotImplementedError("Liger full RoPE does not support partial rotary dimensions.")

        # The shared full-RoPE kernels consume [batch, heads, sequence, head_dim].
        # BAGEL stores packed Q/K as [tokens, heads, head_dim], so adapt through
        # a synthetic batch dimension and restore the packed layout afterwards.
        q_embed, k_embed = veomni_apply_rotary_pos_emb(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            cos.unsqueeze(0),
            sin.unsqueeze(0),
            unsqueeze_dim=1,
        )
        return (
            q_embed.squeeze(0).transpose(0, 1).contiguous(),
            k_embed.squeeze(0).transpose(0, 1).contiguous(),
        )

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _fold_zero_anchors(target: torch.Tensor, *anchors: torch.Tensor) -> torch.Tensor:
    # When a packed batch has no generation tokens, the MoT "gen" expert weights still
    # run on an empty slice and produce zero-sized outputs. Folding ``sum() * 0.0`` of
    # those outputs into ``target`` keeps the gen-expert parameters in the autograd graph
    # (with zero gradient) so FSDP/DP gradient reduction sees the same parameter set on
    # every rank regardless of which modalities a micro-batch contains.
    anchor = target.new_zeros(())
    has_anchor = False
    for value in anchors:
        if torch.is_tensor(value):
            anchor = anchor + value.sum() * 0.0
            has_anchor = True
    if not has_anchor:
        return target
    return target + anchor


def _check_packed_inference_mode(
    mode: str,
) -> bool:
    if mode == "und":
        return False
    if mode == "gen":
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
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        module = super()._apply(fn, recurse=recurse)
        self.inv_freq = self.inv_freq.float()
        self.original_inv_freq = self.original_inv_freq.float()
        return module

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
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
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


class Qwen2MLP(TransformersQwen2MLP):
    """Qwen2 SwiGLU MLP using the configured VeOmni ops backend."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Empty modality branches retain zero-gradient FSDP anchors. Liger's
        # Triton kernel cannot launch the resulting zero-row grid.
        if x.numel() == 0:
            return super().forward(x)

        if veomni_swiglu_mlp.use_non_eager_impl:
            if self.config.hidden_act not in {"silu", "swish"}:
                raise NotImplementedError(
                    f"Liger SwiGLU requires hidden_act='silu' or 'swish', got {self.config.hidden_act!r}."
                )
            return veomni_swiglu_mlp(self, x)

        return super().forward(x)


class Qwen2RMSNorm(TransformersQwen2RMSNorm):
    """Qwen2 RMSNorm using the configured VeOmni ops backend."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Empty modality branches retain zero-gradient FSDP anchors. Liger's
        # Triton kernel cannot launch the resulting zero-row grid.
        if hidden_states.numel() == 0:
            return super().forward(hidden_states)

        if veomni_rms_norm.use_non_eager_impl:
            return veomni_rms_norm(hidden_states, self.weight, self.variance_epsilon)

        return super().forward(hidden_states)


class BagelQwen2MoTAttention(nn.Module):
    def __init__(self, config: BagelQwen2MoTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.query_size = self.num_heads * self.head_dim
        self.key_value_size = self.num_key_value_heads * self.head_dim
        self.qkv_split_sizes = (self.query_size, self.key_value_size, self.key_value_size)
        qkv_size = sum(self.qkv_split_sizes)

        self.qkv_proj_und = nn.Linear(self.hidden_size, qkv_size, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.qkv_proj_gen = nn.Linear(self.hidden_size, qkv_size, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Keep the public checkpoint schema unchanged even though the runtime
        # parameters are merged: direct PyTorch loads combine legacy Q/K/V keys,
        # and ordinary state_dict exports split the combined parameters again.
        self.register_load_state_dict_pre_hook(combine_qkv_state_dict_pre_hook)
        self.register_state_dict_post_hook(split_qkv_state_dict_post_hook)

    def _forward_packed_train(
        self,
        packed_sequence: torch.Tensor,
        attention_mask: BlockMask,
        packed_position_cos: torch.Tensor,
        packed_position_sin: torch.Tensor,
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
    ) -> torch.Tensor:
        packed_qkv_states = packed_sequence.new_zeros((packed_sequence.shape[0], sum(self.qkv_split_sizes)))

        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]
        has_und_tokens = int(packed_und_token_indexes.numel()) > 0
        has_gen_tokens = int(packed_gen_token_indexes.numel()) > 0

        qkv_states_und = self.qkv_proj_und(packed_sequence_und)
        qkv_states_gen = self.qkv_proj_gen(packed_sequence_gen)
        packed_qkv_states[packed_und_token_indexes] = qkv_states_und
        packed_qkv_states[packed_gen_token_indexes] = qkv_states_gen
        if not has_und_tokens:
            packed_qkv_states = _fold_zero_anchors(packed_qkv_states, qkv_states_und)
        if not has_gen_tokens:
            packed_qkv_states = _fold_zero_anchors(packed_qkv_states, qkv_states_gen)

        packed_query_states, packed_key_states, packed_value_states = packed_qkv_states.split(
            self.qkv_split_sizes, dim=-1
        )
        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)
        query_states_norm_und = self.q_norm(packed_query_states[packed_und_token_indexes])
        query_states_norm_gen = self.q_norm_moe_gen(packed_query_states[packed_gen_token_indexes])
        key_states_norm_und = self.k_norm(packed_key_states[packed_und_token_indexes])
        key_states_norm_gen = self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes])
        packed_query_states_[packed_und_token_indexes] = query_states_norm_und
        packed_query_states_[packed_gen_token_indexes] = query_states_norm_gen
        packed_key_states_[packed_und_token_indexes] = key_states_norm_und
        packed_key_states_[packed_gen_token_indexes] = key_states_norm_gen
        if not has_und_tokens:
            packed_query_states_ = _fold_zero_anchors(packed_query_states_, query_states_norm_und)
            packed_key_states_ = _fold_zero_anchors(packed_key_states_, key_states_norm_und)
        if not has_gen_tokens:
            packed_query_states_ = _fold_zero_anchors(packed_query_states_, query_states_norm_gen)
            packed_key_states_ = _fold_zero_anchors(packed_key_states_, key_states_norm_gen)

        packed_query_states_, packed_key_states_ = _apply_rotary_pos_emb(
            packed_query_states_,
            packed_key_states_,
            packed_position_cos,
            packed_position_sin,
            unsqueeze_dim=1,
        )

        if self.config._attn_implementation != "veomni_flex_attention_with_sp":
            raise ValueError(
                "BAGEL Qwen2-MoT training requires "
                "attn_implementation='veomni_flex_attention_with_sp', got "
                f"{self.config._attn_implementation!r}."
            )
        packed_attn_output, _ = fused_attention_forward(
            self,
            packed_query_states_.transpose(0, 1).unsqueeze(0),
            packed_key_states_.transpose(0, 1).unsqueeze(0),
            packed_value_states.transpose(0, 1).unsqueeze(0),
            attention_mask,
            dropout=0.0,
            scaling=self.head_dim**-0.5,
        )
        packed_attn_output = packed_attn_output.squeeze(0)
        packed_attn_output = packed_attn_output.reshape(-1, self.num_heads * self.head_dim)
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        attn_output_und = self.o_proj(packed_attn_output[packed_und_token_indexes])
        attn_output_gen = self.o_proj_moe_gen(packed_attn_output[packed_gen_token_indexes])
        packed_attn_output_[packed_und_token_indexes] = attn_output_und
        packed_attn_output_[packed_gen_token_indexes] = attn_output_gen
        if not has_und_tokens:
            packed_attn_output_ = _fold_zero_anchors(packed_attn_output_, attn_output_und)
        if not has_gen_tokens:
            packed_attn_output_ = _fold_zero_anchors(packed_attn_output_, attn_output_gen)
        return packed_attn_output_

    def _project_inference_qkv(
        self,
        packed_query_sequence: torch.Tensor,
        *,
        is_gen: bool,
        packed_text_indexes: Optional[torch.Tensor],
        packed_vae_token_indexes: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project packed inference tokens through the matching MoT attention expert."""
        if not is_gen:
            packed_query_states, packed_key_states, packed_value_states = self.qkv_proj_und(
                packed_query_sequence
            ).split(self.qkv_split_sizes, dim=-1)
            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
            return packed_query_states, packed_key_states, packed_value_states

        packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
        packed_qkv_states = packed_query_sequence.new_zeros(
            (packed_query_sequence.shape[0], sum(self.qkv_split_sizes))
        )

        packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
        packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
        packed_qkv_states[packed_text_indexes] = self.qkv_proj_und(packed_text_query_sequence)
        packed_qkv_states[packed_vae_token_indexes] = self.qkv_proj_gen(packed_vae_query_sequence)

        packed_query_states, packed_key_states, packed_value_states = packed_qkv_states.split(
            self.qkv_split_sizes, dim=-1
        )
        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim).to(torch.float32)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim).to(torch.float32)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
        packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(
            packed_query_states[packed_vae_token_indexes]
        )
        packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
        packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_vae_token_indexes])
        return packed_query_states, packed_key_states, packed_value_states

    def _merge_inference_kv_cache(
        self,
        packed_key_states: torch.Tensor,
        packed_value_states: torch.Tensor,
        *,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache],
        key_values_lens: Optional[torch.Tensor],
        packed_key_value_indexes: Optional[torch.Tensor],
        total_key_value_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge the current query span with this layer's packed KV cache."""
        if past_key_values is None or past_key_values.key_cache[self.layer_idx] is None:
            return packed_key_states, packed_value_states
        if key_values_lens is None or packed_key_value_indexes is None:
            raise ValueError("key_values_lens and packed_key_value_indexes are required when cache is non-empty.")

        past_key_states = past_key_values.key_cache[self.layer_idx]
        past_value_states = past_key_values.value_cache[self.layer_idx]
        merged_key_states = past_key_states.new_zeros(
            (total_key_value_tokens, self.num_key_value_heads, self.head_dim)
        )
        merged_value_states = past_key_states.new_zeros(
            (total_key_value_tokens, self.num_key_value_heads, self.head_dim)
        )
        merged_key_states[packed_query_indexes] = packed_key_states
        merged_key_states[packed_key_value_indexes] = past_key_states
        merged_value_states[packed_query_indexes] = packed_value_states
        merged_value_states[packed_key_value_indexes] = past_value_states
        return merged_key_states, merged_value_states

    def _run_inference_attention(
        self,
        packed_query_states: torch.Tensor,
        merged_key_states: torch.Tensor,
        merged_value_states: torch.Tensor,
        *,
        attention_mask: Optional[BlockMask],
        is_causal: bool,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
    ) -> torch.Tensor:
        """Dispatch packed inference attention to FlexAttention, FlashAttention, or NPU."""
        if attention_mask is not None:
            if self.config._attn_implementation != "veomni_flex_attention_with_sp":
                raise ValueError(
                    "BAGEL Qwen2-MoT packed prefill requires "
                    "attn_implementation='veomni_flex_attention_with_sp', got "
                    f"{self.config._attn_implementation!r}."
                )
            packed_attn_output, _ = fused_attention_forward(
                self,
                packed_query_states.transpose(0, 1).unsqueeze(0),
                merged_key_states.transpose(0, 1).unsqueeze(0),
                merged_value_states.transpose(0, 1).unsqueeze(0),
                attention_mask,
                dropout=0.0,
                scaling=self.head_dim**-0.5,
                # Inference packs complete logical documents locally and does
                # not enter the training-only Ulysses redistribution.
                skip_ulysses=True,
            )
            return packed_attn_output.squeeze(0)

        if IS_CUDA_AVAILABLE:
            packed_attn_output, _ = fused_attention_forward(
                self,
                packed_query_states.transpose(0, 1).unsqueeze(0),
                merged_key_states.transpose(0, 1).unsqueeze(0),
                merged_value_states.transpose(0, 1).unsqueeze(0),
                attention_mask=None,
                dropout=0.0,
                is_causal=is_causal,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                # Inference owns its packed KV-cache layout and does not enter the
                # training-only module Ulysses redistribution.
                skip_ulysses=True,
            )
            return packed_attn_output.squeeze(0)

        head_num = packed_query_states.shape[1]
        if is_causal:
            atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).bool().to(packed_query_states.device)
            return torch_npu.npu_fusion_attention(
                packed_query_states,
                merged_key_states,
                merged_value_states,
                head_num,
                pse=None,
                padding_mask=None,
                atten_mask=atten_mask_npu,
                scale=1.0 / math.sqrt(packed_query_states.shape[-1]),
                keep_prob=1,
                input_layout="TND",
                actual_seq_qlen=tuple(cu_seq_lens_q[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seq_lens_k[1:].cpu().numpy().tolist()),
                sparse_mode=3,
            )[0]
        return torch_npu.npu_fusion_attention(
            packed_query_states,
            merged_key_states,
            merged_value_states,
            head_num,
            pse=None,
            atten_mask=None,
            scale=1.0 / math.sqrt(packed_query_states.shape[-1]),
            keep_prob=1,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seq_lens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seq_lens_k[1:].cpu().numpy().tolist()),
        )[0]

    def _project_inference_output(
        self,
        packed_attn_output: torch.Tensor,
        *,
        is_gen: bool,
        packed_text_indexes: Optional[torch.Tensor],
        packed_vae_token_indexes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Project packed attention outputs through the matching MoT output expert."""
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        if not is_gen:
            return self.o_proj(packed_attn_output)

        packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
        packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(
            packed_attn_output[packed_vae_token_indexes]
        )
        return packed_attn_output

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
        attention_mask: Optional[BlockMask] = None,
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
        *,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
        total_key_value_tokens: int,
    ) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        is_gen = _check_packed_inference_mode(mode)
        packed_query_states, packed_key_states, packed_value_states = self._project_inference_qkv(
            packed_query_sequence,
            is_gen=is_gen,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
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

        merged_key_states, merged_value_states = self._merge_inference_kv_cache(
            packed_key_states,
            packed_value_states,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            total_key_value_tokens=total_key_value_tokens,
        )
        packed_attn_output = self._run_inference_attention(
            packed_query_states,
            merged_key_states,
            merged_value_states,
            attention_mask=attention_mask,
            is_causal=is_causal,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
        )
        packed_attn_output = self._project_inference_output(
            packed_attn_output,
            is_gen=is_gen,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
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
        attention_mask: BlockMask,
        packed_position_cos: torch.Tensor,
        packed_position_sin: torch.Tensor,
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
    ) -> torch.Tensor:
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        has_und_tokens = int(packed_und_token_indexes.numel()) > 0
        has_gen_tokens = int(packed_gen_token_indexes.numel()) > 0
        normed_sequence_und = self.input_layernorm(packed_sequence[packed_und_token_indexes])
        normed_sequence_gen = self.input_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
        packed_sequence_[packed_und_token_indexes] = normed_sequence_und
        packed_sequence_[packed_gen_token_indexes] = normed_sequence_gen
        if not has_und_tokens:
            packed_sequence_ = _fold_zero_anchors(packed_sequence_, normed_sequence_und)
        if not has_gen_tokens:
            packed_sequence_ = _fold_zero_anchors(packed_sequence_, normed_sequence_gen)

        packed_sequence_, _ = self.self_attn(
            packed_sequence=packed_sequence_,
            attention_mask=attention_mask,
            packed_position_cos=packed_position_cos,
            packed_position_sin=packed_position_sin,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        packed_sequence = residual + packed_sequence_

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        post_attn_und = self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        post_attn_gen = self.post_attention_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
        mlp_und = self.mlp(post_attn_und)
        mlp_gen = self.mlp_moe_gen(post_attn_gen)
        packed_sequence_[packed_und_token_indexes] = mlp_und
        packed_sequence_[packed_gen_token_indexes] = mlp_gen
        if not has_und_tokens:
            packed_sequence_ = _fold_zero_anchors(packed_sequence_, post_attn_und, mlp_und)
        if not has_gen_tokens:
            packed_sequence_ = _fold_zero_anchors(packed_sequence_, post_attn_gen, mlp_gen)
        output = residual + packed_sequence_
        return output

    def _apply_inference_input_norm(
        self,
        packed_query_sequence: torch.Tensor,
        *,
        is_gen: bool,
        packed_text_indexes: Optional[torch.Tensor],
        packed_vae_token_indexes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Normalize inference tokens with the matching MoT expert."""
        if not is_gen:
            return self.input_layernorm(packed_query_sequence)

        normed_sequence = torch.zeros_like(packed_query_sequence)
        normed_sequence[packed_text_indexes] = self.input_layernorm(packed_query_sequence[packed_text_indexes])
        normed_sequence[packed_vae_token_indexes] = self.input_layernorm_moe_gen(
            packed_query_sequence[packed_vae_token_indexes]
        )
        return normed_sequence

    def _apply_inference_mlp(
        self,
        packed_query_sequence: torch.Tensor,
        *,
        is_gen: bool,
        packed_text_indexes: Optional[torch.Tensor],
        packed_vae_token_indexes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the inference MLP with text and VAE tokens routed independently."""
        if not is_gen:
            return self.mlp(self.post_attention_layernorm(packed_query_sequence))

        packed_text_query_sequence = self.post_attention_layernorm(packed_query_sequence[packed_text_indexes]).to(
            torch.bfloat16
        )
        packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(
            packed_query_sequence[packed_vae_token_indexes]
        ).to(torch.bfloat16)
        mlp_output = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
        mlp_output[packed_text_indexes] = self.mlp(packed_text_query_sequence)
        mlp_output[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
        return mlp_output

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
        attention_mask: Optional[BlockMask] = None,
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
        *,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
        total_key_value_tokens: int,
    ) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        is_gen = _check_packed_inference_mode(mode)
        residual = packed_query_sequence
        packed_query_sequence = self._apply_inference_input_norm(
            packed_query_sequence,
            is_gen=is_gen,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
        )
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
            attention_mask=attention_mask,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            total_key_value_tokens=total_key_value_tokens,
        )
        packed_query_sequence = residual + packed_query_sequence

        residual = packed_query_sequence
        packed_query_sequence = self._apply_inference_mlp(
            packed_query_sequence,
            is_gen=is_gen,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        if self.training:
            return self._forward_packed_train(*args, **kwargs), None
        return self._forward_packed_inference(*args, **kwargs)


class BagelQwen2MoTBackbone(nn.Module):
    def __init__(self, config: BagelQwen2MoTConfig):
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList(
            [BagelQwen2MoTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BagelQwen2RotaryEmbedding(config=config)
        self.use_moe = "Mo" in config.layer_module

    def _build_inference_attention_mask(
        self,
        packed_query_sequence: torch.Tensor,
        packed_attention_metadata: Optional[torch.Tensor],
        *,
        cache_has_values: bool,
    ) -> Optional[BlockMask]:
        """Validate packed-prefill invariants and build its FlexAttention mask."""
        if packed_attention_metadata is None:
            return None
        if cache_has_values:
            raise ValueError("BAGEL FlexAttention prefill requires an empty KV cache.")

        expected_metadata_shape = (3, int(packed_query_sequence.shape[0]))
        if tuple(packed_attention_metadata.shape) != expected_metadata_shape:
            raise ValueError(
                "BAGEL FlexAttention prefill metadata must match the packed query sequence: "
                f"expected {expected_metadata_shape}, got {tuple(packed_attention_metadata.shape)}."
            )
        return build_mot_block_mask(packed_attention_metadata.to(device=packed_query_sequence.device))

    def _apply_inference_final_norm(
        self,
        packed_query_sequence: torch.Tensor,
        *,
        is_gen: bool,
        packed_text_indexes: Optional[torch.Tensor],
        packed_vae_token_indexes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the final backbone norm with the matching MoT expert."""
        if not is_gen:
            return self.norm(packed_query_sequence)

        normed_sequence = torch.zeros_like(packed_query_sequence)
        normed_sequence[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
        normed_sequence[packed_vae_token_indexes] = self.norm_moe_gen(packed_query_sequence[packed_vae_token_indexes])
        return normed_sequence

    def _forward_packed_train(
        self,
        packed_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_attention_metadata: torch.Tensor,
        packed_und_token_indexes: Optional[torch.Tensor] = None,
        packed_gen_token_indexes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_mask = build_mot_block_mask(packed_attention_metadata)
        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        packed_position_cos = cos.squeeze(0)
        packed_position_sin = sin.squeeze(0)

        if self.use_moe:
            if packed_und_token_indexes is None:
                raise ValueError("packed_und_token_indexes is required for BAGEL MoT training.")
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
        else:
            packed_und_token_indexes = torch.arange(packed_sequence.shape[0], device=packed_sequence.device)
            packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])

        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                packed_sequence, _ = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    packed_sequence,
                    attention_mask,
                    packed_position_cos,
                    packed_position_sin,
                    packed_und_token_indexes,
                    packed_gen_token_indexes,
                )
            else:
                packed_sequence, _ = decoder_layer(
                    packed_sequence=packed_sequence,
                    attention_mask=attention_mask,
                    packed_position_cos=packed_position_cos,
                    packed_position_sin=packed_position_sin,
                    packed_und_token_indexes=packed_und_token_indexes,
                    packed_gen_token_indexes=packed_gen_token_indexes,
                )

        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            normed_sequence_und = self.norm(packed_sequence[packed_und_token_indexes])
            normed_sequence_gen = self.norm_moe_gen(packed_sequence[packed_gen_token_indexes])
            packed_sequence_[packed_und_token_indexes] = normed_sequence_und
            packed_sequence_[packed_gen_token_indexes] = normed_sequence_gen
            if int(packed_und_token_indexes.numel()) == 0:
                packed_sequence_ = _fold_zero_anchors(packed_sequence_, normed_sequence_und)
            if int(packed_gen_token_indexes.numel()) == 0:
                packed_sequence_ = _fold_zero_anchors(packed_sequence_, normed_sequence_gen)
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
        packed_attention_metadata: Optional[torch.Tensor] = None,
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
    ) -> BaseNavitOutputWithPast:
        is_gen = _check_packed_inference_mode(mode)
        query_device = packed_query_sequence.device
        packed_query_indexes = packed_query_indexes.to(device=query_device)
        packed_query_position_ids = packed_query_position_ids.to(device=query_device)
        if packed_key_value_indexes is not None:
            packed_key_value_indexes = packed_key_value_indexes.to(device=query_device)
        if packed_vae_token_indexes is not None:
            packed_vae_token_indexes = packed_vae_token_indexes.to(device=query_device)
        if packed_text_indexes is not None:
            packed_text_indexes = packed_text_indexes.to(device=query_device)
        if past_key_values is None:
            past_key_values = NaiveCache(len(self.layers))

        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        packed_query_position_embeddings = (cos.squeeze(0), sin.squeeze(0))

        cache_has_values = past_key_values.key_cache[0] is not None
        if cache_has_values and key_values_lens is None:
            raise ValueError("key_values_lens is required when cache is non-empty.")
        attention_mask = self._build_inference_attention_mask(
            packed_query_sequence,
            packed_attention_metadata,
            cache_has_values=cache_has_values,
        )

        effective_key_values_lens = key_values_lens + query_lens if cache_has_values else query_lens
        cu_seq_lens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0)).to(torch.int32)
        cu_seq_lens_k = torch.nn.functional.pad(torch.cumsum(effective_key_values_lens, dim=0), (1, 0)).to(torch.int32)
        max_length_q = int(query_lens.max().item())
        max_length_k = int(effective_key_values_lens.max().item())
        total_key_value_tokens = int(effective_key_values_lens.sum().item())

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
                attention_mask=attention_mask,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                total_key_value_tokens=total_key_value_tokens,
            )

        packed_query_sequence = self._apply_inference_final_norm(
            packed_query_sequence,
            is_gen=is_gen,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
        )
        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )

    def forward(self, *args: Any, **kwargs: Any) -> BaseNavitOutputWithPast:
        if self.training:
            return BaseNavitOutputWithPast(packed_query_sequence=self._forward_packed_train(*args, **kwargs))
        return self._forward_packed_inference(*args, **kwargs)


__all__ = ["BaseNavitOutputWithPast", "BagelQwen2MoT", "NaiveCache"]
