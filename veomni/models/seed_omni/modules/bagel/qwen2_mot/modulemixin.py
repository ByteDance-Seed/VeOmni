"""SeedOmni V2 carrier hooks for BAGEL's Qwen2-MoT backbone."""

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
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.modulemixin import ModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, get_tail_output_item, iter_desired_items
from ..sources import (
    BAGEL_FLOW_HIDDEN,
    BAGEL_FLOW_QUERY,
    BAGEL_FLOW_VELOCITY,
    BAGEL_SIGLIP_CONTEXT,
    BAGEL_VAE_CONTEXT,
)
from .configuration import BagelQwen2MoTConfig
from .generation_state import MotCacheContext, MotGenerationState
from .processing import PackedConversation, PackedSpan, build_mot_attention_mask, preprocess_mot_inputs


class BagelQwen2MoTModuleMixin(ModuleMixin):
    """Carrier hooks and graph entrypoints for BAGEL's packed MoT backbone."""

    def init_omni_state(self) -> None:
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._packed_training: PackedConversation | None = None
        self._sp_full_sequence_length: int | None = None
        self._validated_ulysses_size: int | None = None
        self._generation_state = MotGenerationState()

    def reset_local_inference_state(self) -> None:
        self._generation_state.reset()

    # ── Graph Entrypoints ──────────────────────────────────

    def generate(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if conversation_list is None:
            raise ValueError("BAGEL Qwen2-MoT generate requires conversation_list.")

        generation_kwargs = generation_kwargs or {}
        infer_mode = self._generation_state.update_infer_mode(generation_kwargs)
        if self._generation_state.main.cache is None or infer_mode == "gen":
            hidden_states = self._prefill_prompt(conversation_list, generation_kwargs)
        else:
            hidden_states = self._decode_next_token(conversation_list)

        if infer_mode != "gen":
            if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
                hidden_states = hidden_states.squeeze(0)
            if hidden_states.dim() != 2:
                raise ValueError(f"BAGEL Qwen2-MoT expected packed hidden states, got {tuple(hidden_states.shape)}.")
            conversation_list.append(
                ConversationItem(
                    type="output",
                    value=hidden_states[-1:].contiguous(),
                    role="assistant",
                )
            )
        return {"conversation_list": conversation_list}

    def denoise_branch(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if conversation_list is None:
            raise ValueError("BAGEL Qwen2-MoT denoise_branch requires conversation_list.")

        self._generation_state.validate_cfg_request(generation_kwargs or {})
        self._generation_state.main.require_ready()
        tail = get_tail_output_item(conversation_list, sources=[BAGEL_FLOW_QUERY])
        if tail is None or not torch.is_tensor(tail.value):
            raise ValueError("BAGEL Qwen2-MoT denoise branch requires source='bagel_flow_query'.")

        query = tail.value
        if query.dim() == 3 and query.shape[0] == 1:
            query = query.squeeze(0)
        if query.dim() != 2:
            raise ValueError(f"BAGEL Qwen2-MoT denoise branch expects rank-2 query tensor, got {tuple(query.shape)}.")
        if int(query.shape[-1]) != int(self.config.hidden_size):
            raise ValueError(
                "BAGEL Qwen2-MoT denoise branch hidden-size mismatch: "
                f"got {query.shape[-1]}, expected {self.config.hidden_size}."
            )
        if int(query.shape[0]) < 3:
            raise ValueError("BAGEL Qwen2-MoT denoise query must include start/end marker embeddings.")

        inputs = self._generation_state.preprocess_parallel_denoise_inputs(
            query,
            generation_kwargs or {},
            timestep=tail.meta.get("timestep"),
            empty_cache_factory=self._new_empty_cache,
            device=self.device,
            dtype=self.dtype,
        )
        outputs = self.forward_inference(
            **inputs,
            update_past_key_values=False,
            is_causal=False,
            mode="gen",
        )

        tail.source = BAGEL_FLOW_HIDDEN
        tail.value = outputs["hidden_states"].to(device=self.device, dtype=self.dtype)
        return {"conversation_list": conversation_list}

    def collect_velocity(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if conversation_list is None:
            raise ValueError("BAGEL Qwen2-MoT collect_velocity requires conversation_list.")

        self._generation_state.validate_cfg_request(generation_kwargs or {})
        tail = get_tail_output_item(conversation_list, sources=[BAGEL_FLOW_VELOCITY])
        if tail is None or not torch.is_tensor(tail.value):
            raise ValueError("BAGEL Qwen2-MoT velocity collection requires source='bagel_flow_velocity'.")

        velocity = tail.value
        if velocity.dim() == 3 and velocity.shape[0] == 1:
            velocity = velocity.squeeze(0)
        if velocity.dim() != 2:
            raise ValueError(
                f"BAGEL Qwen2-MoT velocity collection expects rank-2 velocity, got {tuple(velocity.shape)}."
            )

        tail.value = self._generation_state.collect_velocity(
            velocity,
            generation_kwargs or {},
            device=self.device,
            dtype=self.dtype,
        )
        return {"conversation_list": conversation_list}

    # ── Training hooks ──────────────────────────────────

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

    # ── Dummy helper ──────────────────────────────────

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

    # ── Internal helpers ──────────────────────────────────

    def _prefill_prompt(
        self,
        conversation_list: list[ConversationItem],
        generation_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        packed = preprocess_mot_inputs(
            [conversation_list],
            device=self.device,
            dtype=self.dtype,
            hidden_size=int(self.config.hidden_size),
        )
        if packed is None:
            raise ValueError("BAGEL Qwen2-MoT generate requires at least one embedded text/image item.")

        state = self._generation_state
        main_context = state.main
        main_context.reset()
        main_context.ensure_empty(empty_cache_factory=self._new_empty_cache, device=self.device)
        outputs = None

        for span in packed.spans:
            if span.item.type == "text":
                # Text CFG branches start from the main prompt cache before
                # the current text span, so snapshot before pre-filling it.
                self._generation_state.cfg_text.snapshot(
                    cache=main_context.cache,
                    key_values_lens=main_context.key_values_lens,
                    packed_key_value_indexes=main_context.packed_key_value_indexes,
                    next_position_id=packed.packed_position_ids[span.start],
                    empty_cache_factory=self._new_empty_cache,
                    device=self.device,
                )
                self._prefill_text_cfg_contexts(span, packed, generation_kwargs=generation_kwargs)

            outputs = self._prefill_main_prompt_span(span, packed, main_context)

            if span.item.type == "image":
                # After image context is in the main cache, text CFG should
                # keep that visual context while dropping later text condition.
                state.cfg_text.snapshot(
                    cache=main_context.cache,
                    key_values_lens=main_context.key_values_lens,
                    packed_key_value_indexes=main_context.packed_key_value_indexes,
                    next_position_id=main_context.next_position_ids,
                    empty_cache_factory=self._new_empty_cache,
                    device=self.device,
                )

        if outputs is None:
            raise RuntimeError("BAGEL Qwen2-MoT prefill produced no outputs.")
        return outputs["hidden_states"]

    def _decode_next_token(self, conversation_list: list[ConversationItem]) -> torch.Tensor:
        main_context = self._generation_state.main
        main_context.require_ready()
        tail = conversation_list[-1]
        if tail.type != "output":
            raise ValueError(f"BAGEL Qwen2-MoT decode expects tail output item, got {tail.type!r}.")

        packed_query_sequence = tail.value
        if not torch.is_tensor(packed_query_sequence):
            raise ValueError("BAGEL Qwen2-MoT decode expects tail output.value to be an embedding tensor.")
        if packed_query_sequence.dim() == 3 and packed_query_sequence.shape[0] == 1:
            packed_query_sequence = packed_query_sequence.squeeze(0)
        if packed_query_sequence.dim() != 2:
            raise ValueError(
                f"BAGEL Qwen2-MoT expected tail output embedding rank 2, got {tuple(packed_query_sequence.shape)}."
            )
        packed_query_sequence = packed_query_sequence[-1:].contiguous().to(device=self.device, dtype=self.dtype)

        query_lens, packed_query_indexes, packed_position_ids = main_context.packed_query_args(
            1,
            device=self.device,
        )
        outputs = self.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=main_context.cache,
            key_values_lens=main_context.key_values_lens,
            packed_key_value_indexes=main_context.packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        main_context.append_packed_query(
            cache=outputs["past_key_values"],
            query_lens=query_lens,
            device=self.device,
        )

        return outputs["hidden_states"]

    # ── Cache and context helpers ──────────────────────────────────

    def _prefill_main_prompt_span(
        self,
        span: PackedSpan,
        packed: PackedConversation,
        main_context: MotCacheContext,
    ) -> Any:
        span_end = span.start + span.length
        span_position_ids = packed.packed_position_ids[span.start : span_end]
        query_lens, packed_query_indexes, packed_position_ids = main_context.packed_query_args(
            span.length,
            device=self.device,
            position_ids=span_position_ids,
        )
        call_kwargs = {
            "packed_query_sequence": packed.packed_sequence[span.start : span_end],
            "query_lens": query_lens,
            "packed_query_position_ids": packed_position_ids,
            "packed_query_indexes": packed_query_indexes,
            "past_key_values": main_context.cache,
            "key_values_lens": main_context.key_values_lens,
            "packed_key_value_indexes": main_context.packed_key_value_indexes,
            "update_past_key_values": True,
            "is_causal": span.item.type == "text",
            "mode": "und",
        }
        if span.item.type == "output":
            if span.length < 3:
                raise ValueError("BAGEL Qwen2-MoT output query must include start/end marker embeddings.")
            # Runtime flow query output remains marker-wrapped: marker tokens
            # stay on the text path, while interior latent tokens use the gen expert.
            call_kwargs["is_causal"] = False
            call_kwargs["mode"] = "gen"
            call_kwargs["packed_text_indexes"] = torch.tensor([0, span.length - 1], device=self.device)
            call_kwargs["packed_vae_token_indexes"] = torch.arange(
                1,
                span.length - 1,
                device=self.device,
                dtype=torch.long,
            )
        elif span.item.type == "image" and span.item.source == BAGEL_VAE_CONTEXT:
            # Prompt/edit VAE context is now source-routed as an image carrier.
            # Surrounding vision marker rows stay on the text path while the
            # image span itself uses the generation expert.
            call_kwargs["is_causal"] = False
            call_kwargs["mode"] = "gen"
            if span.is_image_triplet:
                call_kwargs["packed_text_indexes"] = torch.tensor(
                    [0, span.length - 1],
                    device=self.device,
                    dtype=torch.long,
                )
                start = span.primary_start
                end = start + span.primary_length
            else:
                start = 0
                end = span.length
            call_kwargs["packed_vae_token_indexes"] = torch.arange(
                start,
                end,
                device=self.device,
                dtype=torch.long,
            )

        outputs = self.forward_inference(**call_kwargs)
        main_context.append_packed_query(
            cache=outputs["past_key_values"],
            query_lens=query_lens,
            device=self.device,
            next_position_ids=packed_position_ids.max().reshape(1) + 1,
        )
        return outputs

    def _prefill_text_cfg_contexts(
        self,
        span: PackedSpan,
        packed: PackedConversation,
        *,
        generation_kwargs: dict[str, Any],
    ) -> None:
        # Text-only image CFG is only needed when image guidance is active.
        if not self._generation_state.cfg_img_requested(generation_kwargs):
            return

        # Image CFG keeps text conditioning while excluding image conditioning,
        # so it needs an independent text prefill branch.
        cfg_img_context = self._generation_state.cfg_img
        cfg_img_context.ensure_empty(empty_cache_factory=self._new_empty_cache, device=self.device)
        query_lens, packed_query_indexes, packed_position_ids = cfg_img_context.packed_query_args(
            span.length,
            device=self.device,
        )
        span_end = span.start + span.length
        outputs = self.forward_inference(
            packed_query_sequence=packed.packed_sequence[span.start : span_end],
            query_lens=query_lens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=cfg_img_context.cache,
            key_values_lens=cfg_img_context.key_values_lens,
            packed_key_value_indexes=cfg_img_context.packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        cfg_img_context.append_packed_query(
            cache=outputs["past_key_values"],
            query_lens=query_lens,
            device=self.device,
        )

    def _new_empty_cache(self) -> Any:
        try:
            from .modeling import NaiveCache
        except ImportError as exc:
            raise RuntimeError("Unable to import BAGEL NaiveCache for text CFG context initialization.") from exc
        return NaiveCache(len(self.model.layers))


class BagelQwen2MoTMetricMeterMixin(MetricMeterMixin):
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


__all__ = ["BagelQwen2MoTModuleMixin", "BagelQwen2MoTMetricMeterMixin"]
