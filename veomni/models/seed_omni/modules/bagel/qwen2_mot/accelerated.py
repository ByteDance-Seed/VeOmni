"""VeOmni-accelerated BAGEL Qwen2-MoT — fused OpSlot ops plus training-graph hooks.

Native weights, packed ``forward``, and FSM ``generate`` live on
:class:`BagelQwen2MoT` in ``modeling.py``. This module supplies the fused RMSNorm / SwiGLU / RoPE replacements,
``fused_attention_forward``, and the training pre/forward/post hooks.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import (
    gather_outputs,
    slice_input_tensor,
    sp_pad,
)
from ......ops.dispatch import OpSlot
from ......ops.kernels.attention import fused_attention_forward
from ......utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE
from ....mixins.base_mixin import BaseMixin
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, iter_desired_items
from ..sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from .configuration import BagelQwen2MoTConfig
from .masking import build_mot_block_mask, pad_mot_attention_metadata
from .modeling import (
    BagelQwen2MoT,
    BagelQwen2MoTAttention,
)
from .modeling import (
    _apply_rotary_pos_emb as _apply_rotary_pos_emb_eager,
)
from .processing import PackedConversation, preprocess_mot_inputs


if IS_NPU_AVAILABLE:
    import torch_npu


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks — depends on :class:`BagelQwen2MoT` modeling APIs."""

    config: BagelQwen2MoTConfig
    device: torch.device
    dtype: torch.dtype

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._packed_training: PackedConversation | None = None
        self._sp_full_sequence_length: int | None = None
        self._validated_ulysses_size: int | None = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._packed_training = preprocess_mot_inputs(
            conversation_list,
            device=self.device,
            dtype=self.dtype,
            hidden_size=int(self.config.hidden_size),
        )
        if self._packed_training is None:
            raise ValueError(
                "BAGEL Qwen2-MoT forward requires a non-empty conversation_list with real tokens; "
                "got no packable tokens across the whole batch."
            )

        # Meter the full replicated batch before SP slices its packed sequence.
        self.metric_meter_set_seqlens(
            "forward",
            [int(sum(splits)) for splits in self._packed_training.sample_splits],
        )

        packed_sequence = self._fold_dummy_anchors(self._packed_training.packed_sequence, conversation_list)
        sequence_length = int(packed_sequence.shape[0])
        attention_metadata = self._packed_training.packed_attention_metadata
        expected_metadata_shape = (3, sequence_length)
        if tuple(attention_metadata.shape) != expected_metadata_shape:
            raise ValueError(
                "BAGEL Qwen2-MoT attention metadata must match the full packed sample: "
                f"expected {expected_metadata_shape}, got {tuple(attention_metadata.shape)}."
            )

        ps = get_parallel_state()
        if ps.sp_size == 1:
            self._sp_full_sequence_length = None
            return {
                "packed_sequence": packed_sequence,
                "packed_position_ids": self._packed_training.packed_position_ids,
                "packed_token_type_ids": self._packed_training.packed_token_type_ids,
                "packed_attention_metadata": attention_metadata,
            }

        if ps.cp_size != 1:
            raise ValueError(
                f"BAGEL Qwen2-MoT training supports Ulysses sequence parallelism only; got cp_size={ps.cp_size}."
            )
        if self._validated_ulysses_size != ps.ulysses_size:
            num_heads = int(self.config.num_attention_heads)
            num_key_value_heads = int(self.config.num_key_value_heads)
            if num_heads % ps.ulysses_size != 0:
                raise ValueError(
                    "BAGEL Qwen2-MoT attention heads must be divisible by "
                    f"ulysses_size: num_heads={num_heads}, ulysses_size={ps.ulysses_size}."
                )
            if num_key_value_heads % ps.ulysses_size != 0:
                raise ValueError(
                    "BAGEL Qwen2-MoT KV heads must be divisible by "
                    f"ulysses_size for native GQA: num_key_value_heads={num_key_value_heads}, "
                    f"ulysses_size={ps.ulysses_size}."
                )
            self._validated_ulysses_size = ps.ulysses_size

        # Tensor inputs are padded then sliced for Ulysses. Attention metadata
        # stays full because the Flex adapter reconstructs the full sequence
        # before building/using its BlockMask.
        self._sp_full_sequence_length = sequence_length
        packed_sequence = sp_pad(packed_sequence, dim=0, pad_value=0)
        packed_position_ids = sp_pad(self._packed_training.packed_position_ids, dim=0, pad_value=0)
        packed_token_type_ids = sp_pad(self._packed_training.packed_token_type_ids, dim=0, pad_value=-1)
        attention_metadata = pad_mot_attention_metadata(
            attention_metadata,
            padded_length=int(packed_sequence.shape[0]),
        )
        packed_sequence = slice_input_tensor(packed_sequence, dim=0, padding=False, group=ps.sp_group)
        packed_position_ids = slice_input_tensor(
            packed_position_ids,
            dim=0,
            padding=False,
            group=ps.sp_group,
        )
        packed_token_type_ids = slice_input_tensor(
            packed_token_type_ids,
            dim=0,
            padding=False,
            group=ps.sp_group,
        )

        return {
            "packed_sequence": packed_sequence,
            "packed_position_ids": packed_position_ids,
            "packed_token_type_ids": packed_token_type_ids,
            # FlexAttention's generic Ulysses adapter reconstructs the complete
            # padded sequence, so every rank keeps the same full metadata.
            "packed_attention_metadata": attention_metadata,
        }

    @post_forward("forward")
    def forward_post(self, hidden_states: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        ps = get_parallel_state()
        if ps.sp_size > 1:
            if self._sp_full_sequence_length is None:
                raise RuntimeError("BAGEL Qwen2-MoT SP sequence length was not initialized.")
            hidden_states = gather_outputs(hidden_states, gather_dim=0, group=ps.sp_group)
            hidden_states = hidden_states.narrow(0, 0, self._sp_full_sequence_length)

        conversation = self._conversation_carrier
        packed = self._packed_training
        self._conversation_carrier = None
        self._packed_training = None
        self._sp_full_sequence_length = None

        for span in packed.spans:
            span_hidden = hidden_states[span.start : span.start + span.length].to(device=self.device)
            offset = 0
            for item, length in zip(span.items, span.lengths, strict=True):
                item.value = span_hidden[offset : offset + length]
                offset += length
        return {"conversation_list": conversation}

    def _fold_dummy_anchors(
        self,
        packed_sequence: torch.Tensor,
        conversation_list: list[list[ConversationItem]] | None,
    ) -> torch.Tensor:
        # Dummy encoder outputs must still touch MoT on ranks where another rank
        # has the real branch, otherwise FSDP sees different gradient buckets.
        anchor = packed_sequence.new_zeros(())
        has_anchor = False
        include_siglip_dummy, include_flow_dummy = self._has_valid_upstream_embeddings(conversation_list)

        for item in iter_desired_items(conversation_list or [], roles=["dummy"]):
            if not torch.is_tensor(item.value):
                continue

            source = item.source
            if source not in ["bagel_flow_connector", BAGEL_SIGLIP_CONTEXT]:
                continue
            if source == "bagel_flow_connector" and not include_flow_dummy:
                continue
            if source == BAGEL_SIGLIP_CONTEXT and not include_siglip_dummy:
                continue

            has_anchor = True
            anchor = (
                anchor
                + item.value.to(
                    device=packed_sequence.device,
                    dtype=packed_sequence.dtype,
                ).sum()
                * 0.0
            )

        if not has_anchor:
            return packed_sequence
        return packed_sequence + anchor

    def _has_valid_upstream_embeddings(
        self,
        conversation_list: list[list[ConversationItem]] | None,
    ) -> tuple[bool, bool]:
        has_siglip = int(
            self._has_valid_upstream_embedding(
                conversation_list,
                label="SigLIP",
                types=["image"],
                sources=[BAGEL_SIGLIP_CONTEXT],
            )
        )
        has_flow = int(
            self._has_valid_upstream_embedding(
                conversation_list,
                label="flow",
                types=["image"],
                sources=[BAGEL_VAE_CONTEXT],
                meta_keys=["flow_velocity_target"],
            )
        )

        if not dist.is_available() or not dist.is_initialized():
            return bool(has_siglip), bool(has_flow)
        flags = torch.tensor([has_siglip, has_flow], device=self.device, dtype=torch.int32)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        return bool(flags[0].item()), bool(flags[1].item())

    def _has_valid_upstream_embedding(
        self,
        conversation_list: list[list[ConversationItem]] | None,
        *,
        label: str,
        types: list[str],
        sources: list[str] | None = None,
        meta_keys: list[str] | None = None,
    ) -> bool:
        for item in iter_desired_items(
            conversation_list or [],
            types=types,
            roles=["user", "assistant"],
            sources=sources,
            meta_keys=meta_keys,
        ):
            value = item.value
            if not torch.is_tensor(value):
                continue
            if value.dim() == 3 and value.shape[0] == 1:
                value = value.squeeze(0)
            if value.dim() != 2:
                raise ValueError(
                    f"BAGEL Qwen2-MoT {label} alignment expects rank-2 embeddings, got {tuple(value.shape)}."
                )
            if int(value.shape[-1]) != int(self.config.hidden_size):
                raise ValueError(
                    f"BAGEL Qwen2-MoT {label} alignment hidden-size mismatch: "
                    f"got {value.shape[-1]}, expected {self.config.hidden_size}."
                )
            return True
        return False


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for BAGEL's Qwen2-MoT backbone (transformer layers only)."""

    config: BagelQwen2MoTConfig

    def estimate_flops(self, seqlens: list[int]) -> float:
        cfg = self.config
        hidden = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        # Attention uses head_dim = hidden // num_heads (see BagelQwen2MoTAttention).
        head_dim = hidden // num_heads

        # SwiGLU MLP (gate/up/down) + attention projections (q, o over num_heads;
        # k, v over num_kv_heads). Biases are O(hidden) → negligible, ignored.
        mlp_n = hidden * cfg.intermediate_size * 3
        attn_linear_n = hidden * head_dim * (num_heads * 2 + num_kv_heads * 2)
        dense_n = (mlp_n + attn_linear_n) * num_layers

        tokens = sum(seqlens)
        seqlen_sq = sum(s * s for s in seqlens)
        dense_flops = 6 * dense_n * tokens
        attn_flops = 12 * seqlen_sq * head_dim * num_heads * num_layers
        return (dense_flops + attn_flops) / 1e12


class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin):
    """Carrier hooks and graph entrypoints for BAGEL's packed MoT backbone.

    ``generate()`` / ``denoise_branch()`` / ``collect_velocity()`` and the
    shared MoT generation-FSM state already live on the native
    :class:`~.modeling.BagelQwen2MoT` (via its own :class:`~.modeling.InferenceMixin`),
    so no ``InferenceMixin`` is needed here.
    """


veomni_rms_norm = OpSlot("rms_norm", "standard")
veomni_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "full")
veomni_swiglu_mlp = OpSlot("swiglu_mlp", "standard")


class Qwen2RMSNormAccelerated(Qwen2RMSNorm):
    """Qwen2 RMSNorm with VeOmni OpSlot fused dispatch."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Empty modality branches retain zero-gradient FSDP anchors. Liger's
        # Triton kernel cannot launch the resulting zero-row grid.
        if hidden_states.numel() == 0:
            return super().forward(hidden_states)

        if veomni_rms_norm.use_non_eager_impl:
            return veomni_rms_norm(hidden_states, self.weight, self.variance_epsilon)

        return super().forward(hidden_states)


class Qwen2MLPAccelerated(Qwen2MLP):
    """Qwen2 SwiGLU MLP with VeOmni OpSlot fused dispatch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Empty modality branches retain zero-gradient FSDP anchors. Liger's
        # Triton kernel cannot launch the resulting zero-row grid.
        if x.numel() == 0:
            return super().forward(x)

        if veomni_swiglu_mlp.use_non_eager_impl:
            if self.config.hidden_act not in {"silu", "swish"}:
                raise ValueError(
                    f"Liger SwiGLU requires hidden_act='silu' or 'swish', got {self.config.hidden_act!r}. "
                    "Set model.ops_implementation.swiglu_mlp_implementation='eager' "
                    "to use the Transformers reference implementation."
                )
            return veomni_swiglu_mlp(self, x)

        return super().forward(x)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed RoPE with optional fused OpSlot dispatch."""
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

    return _apply_rotary_pos_emb_eager(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


_PACKED_FUSED_ATTN_IMPLEMENTATIONS = (
    "veomni_flex_attention_with_sp",
    "veomni_magi_attention_with_sp",
)


class BagelQwen2MoTAttentionAccelerated(BagelQwen2MoTAttention):
    """MoT attention with fused RoPE and ``fused_attention_forward``."""

    @staticmethod
    def build_attention_mask(packed_attention_metadata: torch.Tensor) -> BlockMask:
        return build_mot_block_mask(packed_attention_metadata)

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    def _attend(
        self,
        packed_query_states: torch.Tensor,
        packed_key_states: torch.Tensor,
        packed_value_states: torch.Tensor,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        if self.config._attn_implementation not in _PACKED_FUSED_ATTN_IMPLEMENTATIONS:
            raise ValueError(
                "BAGEL Qwen2-MoT training requires packed fused attention "
                f"({', '.join(_PACKED_FUSED_ATTN_IMPLEMENTATIONS)}), got "
                f"{self.config._attn_implementation!r}."
            )
        packed_attn_output, _ = fused_attention_forward(
            self,
            packed_query_states.transpose(0, 1).unsqueeze(0),
            packed_key_states.transpose(0, 1).unsqueeze(0),
            packed_value_states.transpose(0, 1).unsqueeze(0),
            attention_mask,
            dropout=0.0,
            scaling=self.head_dim**-0.5,
        )
        packed_attn_output = packed_attn_output.squeeze(0)
        return packed_attn_output.reshape(-1, self.num_heads * self.head_dim)

    def _attend_inference(
        self,
        packed_query_states: torch.Tensor,
        merged_key_states: torch.Tensor,
        merged_value_states: torch.Tensor,
        *,
        attention_mask: BlockMask | None,
        is_causal: bool,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
    ) -> torch.Tensor:
        if attention_mask is not None:
            if self.config._attn_implementation not in _PACKED_FUSED_ATTN_IMPLEMENTATIONS:
                raise ValueError(
                    "BAGEL Qwen2-MoT packed prefill requires packed fused attention "
                    f"({', '.join(_PACKED_FUSED_ATTN_IMPLEMENTATIONS)}), got "
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


class BagelQwen2MoTAccelerated(VeOmniMixin, BagelQwen2MoT):
    _supports_flex_attn = True
    attention_cls = BagelQwen2MoTAttentionAccelerated
    mlp_cls = Qwen2MLPAccelerated
    rms_norm_cls = Qwen2RMSNormAccelerated


__all__ = ["BagelQwen2MoTAccelerated"]
