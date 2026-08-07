"""SeedOmni V2 carrier hooks for BAGEL's Qwen2-MoT backbone — training-graph hooks only.

``generate()``, ``denoise_branch()``, ``collect_velocity()`` and the shared
MoT generation-FSM state live on the native :class:`BagelQwen2MoT` in
``modeling.py`` — this file only carries the training pre/forward/post hooks.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import (
    gather_outputs,
    slice_input_tensor,
    sp_pad,
)
from ....mixins.base_mixin import BaseMixin
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, iter_desired_items
from ..sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from .configuration import BagelQwen2MoTConfig
from .modeling import BagelQwen2MoT
from .processing import PackedConversation, build_mot_attention_mask, preprocess_mot_inputs


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
        attention_mask = build_mot_attention_mask(
            self._packed_training.sample_splits,
            self._packed_training.sample_attn_modes,
            device=self.device,
        )
        sequence_length = int(packed_sequence.shape[0])
        expected_mask_shape = (1, 1, sequence_length, sequence_length)
        if tuple(attention_mask.shape) != expected_mask_shape:
            raise ValueError(
                "BAGEL Qwen2-MoT attention mask must match the full packed sample: "
                f"expected {expected_mask_shape}, got {tuple(attention_mask.shape)}."
            )

        ps = get_parallel_state()
        if ps.sp_size == 1:
            self._sp_full_sequence_length = None
            return {
                "packed_sequence": packed_sequence,
                "packed_position_ids": self._packed_training.packed_position_ids,
                "packed_token_type_ids": self._packed_training.packed_token_type_ids,
                "attention_mask": attention_mask,
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

        self._sp_full_sequence_length = sequence_length
        packed_sequence = sp_pad(packed_sequence, dim=0, pad_value=0)
        packed_position_ids = sp_pad(self._packed_training.packed_position_ids, dim=0, pad_value=0)
        packed_token_type_ids = sp_pad(self._packed_training.packed_token_type_ids, dim=0, pad_value=-1)
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
            "attention_mask": attention_mask,
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


class BagelQwen2MoTAccelerated(VeOmniMixin, BagelQwen2MoT):
    pass


__all__ = ["BagelQwen2MoTAccelerated"]
