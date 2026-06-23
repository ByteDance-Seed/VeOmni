"""SeedOmni V2 carrier hooks for BAGEL's Qwen2-MoT backbone."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from ....conversation import ConversationItem, is_dummy
from ....generation_graph import FSM_SIGNAL_KEY
from ....module import ModuleMixin, post_forward, pre_forward
from ..carrier_updates import materialize_carrier_updates, replace_fields
from ..sources import BAGEL_FLOW_HIDDEN, BAGEL_FLOW_QUERY, BAGEL_FLOW_VELOCITY
from .generation_state import MotGenerationState
from .processing import (
    PackedConversation,
    active_output_item,
    append_query_indexes_to_cache,
    cfg_branch_sequence,
    merge_cfg_velocity,
    next_position_ids_from_packed,
    pack_training_conversation,
    scatter_hidden_states,
    shift_packed_key_value_indexes,
    single_inference_conversation,
    tail_hidden_from_packed,
    tail_query_embedding,
    validate_cfg_request,
)


SIGNAL_NEED_DENOISE_BRANCH = "need_denoise_branch"


class BagelQwen2MoTModuleMixin(ModuleMixin):
    """Training carrier adapter for BAGEL's packed MoT backbone."""

    def init_omni_state(self) -> None:
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._packed_training: PackedConversation | None = None
        self._generation_state = MotGenerationState()

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        packed = pack_training_conversation(
            conversation_list,
            device=self.device,
            dtype=self.dtype,
            hidden_size=int(self.config.hidden_size),
        )
        self._packed_training = packed
        if packed is None:
            return self.dummy_inputs()
        packed_sequence = self._fold_dummy_anchors(packed.packed_sequence, conversation_list)
        return {
            "packed_sequence": packed_sequence,
            "sample_lens": packed.sample_lens,
            "attention_mask": packed.nested_attention_masks,
            "packed_position_ids": packed.packed_position_ids,
            "packed_und_token_indexes": packed.packed_und_token_indexes,
            "packed_gen_token_indexes": packed.packed_gen_token_indexes,
        }

    @post_forward("forward")
    def forward_post(self, hidden_states: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        conversation = self._conversation_carrier
        packed = self._packed_training
        self._conversation_carrier = None
        self._packed_training = None
        if packed is None:
            if conversation is not None:
                for sample in conversation:
                    sample.append(
                        ConversationItem(
                            type="output",
                            value=hidden_states[:1].squeeze(0).to(device=self.device),
                            role="dummy",
                            meta={"source": "bagel_qwen2_mot"},
                        )
                    )
            return {"conversation_list": conversation}
        scatter_hidden_states(packed.spans, hidden_states, device=self.device)
        self._append_missing_flow_hidden_anchors(conversation)
        return {"conversation_list": conversation}

    def dummy_inputs(self) -> dict[str, Any]:
        return {
            "packed_sequence": torch.zeros(1, int(self.config.hidden_size), device=self.device, dtype=self.dtype),
            "sample_lens": [1],
            "attention_mask": [torch.zeros(1, 1, device=self.device, dtype=torch.float32)],
            "packed_position_ids": torch.zeros(1, device=self.device, dtype=torch.long),
            "packed_und_token_indexes": torch.zeros(1, device=self.device, dtype=torch.long),
            "packed_gen_token_indexes": torch.empty(0, device=self.device, dtype=torch.long),
        }

    def _fold_dummy_anchors(
        self,
        packed_sequence: torch.Tensor,
        conversation_list: list[list[ConversationItem]] | None,
    ) -> torch.Tensor:
        # Dummy encoder outputs must still touch MoT on ranks where another rank
        # has the real branch, otherwise FSDP sees different gradient buckets.
        anchor = packed_sequence.new_zeros(())
        has_anchor = False
        include_siglip_dummy = self._global_has_real_siglip_image(conversation_list)
        for sample in conversation_list or []:
            for item in sample:
                if not is_dummy(item) or not torch.is_tensor(item.value):
                    continue
                meta = item.meta if isinstance(item.meta, dict) else {}
                source = meta.get("source")
                if source != "bagel_flow_connector" and not (source == "bagel_siglip_navit" and include_siglip_dummy):
                    continue
                anchor = anchor + item.value.to(device=packed_sequence.device).sum() * 0.0
                has_anchor = True
        if not has_anchor:
            return packed_sequence
        return packed_sequence + anchor

    def _global_has_real_siglip_image(self, conversation_list: list[list[ConversationItem]] | None) -> bool:
        local = 0
        for sample in conversation_list or []:
            for item in sample:
                if is_dummy(item) or item.type != "image" or not torch.is_tensor(item.value):
                    continue
                value = item.value
                if value.dim() == 3 and value.shape[0] == 1:
                    value = value.squeeze(0)
                if value.dim() == 2 and int(value.shape[-1]) == int(self.config.hidden_size):
                    local = 1
                    break
            if local:
                break
        if not dist.is_available() or not dist.is_initialized():
            return bool(local)
        flag = torch.tensor(local, device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _append_missing_flow_hidden_anchors(self, conversation_list: list[list[ConversationItem]] | None) -> None:
        # Text-only samples can share a batch with image-generation samples.
        # Give flow decode a zero MoT output anchor so the MSE path still flows
        # through MoT before reaching llm2vae on those ranks.
        for sample in conversation_list or []:
            has_flow_hidden = False
            anchor: torch.Tensor | None = None
            for item in sample:
                if is_dummy(item) or not torch.is_tensor(item.value):
                    continue
                value = item.value
                if value.dim() == 3 and value.shape[0] == 1:
                    value = value.squeeze(0)
                if value.dim() != 2 or int(value.shape[-1]) != int(self.config.hidden_size):
                    continue
                if anchor is None:
                    anchor = value[:1].to(device=self.device, dtype=self.dtype) * 0.0
                if item.type == "output":
                    has_flow_hidden = True
            if has_flow_hidden or anchor is None:
                continue
            sample.append(
                ConversationItem(
                    type="output",
                    value=anchor,
                    role="assistant",
                    meta={"source": "bagel_qwen2_mot_dummy"},
                )
            )

    def reset_local_inference_state(self) -> None:
        self._generation_state.reset()

    def generate(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        validate_cfg_request(generation_kwargs or {})
        conversation = single_inference_conversation(conversation_list)
        phase = self._resolve_inference_phase(conversation, generation_kwargs or {})
        if phase == "prefill":
            return self._prefill_inference(
                conversation_list=conversation_list,
                conversation=conversation,
                generation_kwargs=generation_kwargs or {},
            )
        if phase == "text_decode":
            return self._decode_text_step(conversation_list=conversation_list, conversation=conversation)
        if phase == "denoise_branch":
            return self._run_denoise_branch(
                conversation_list=conversation_list,
                conversation=conversation,
                generation_kwargs=generation_kwargs or {},
            )
        if phase == "velocity_collect":
            return self._collect_or_merge_velocity(
                conversation_list=conversation_list,
                conversation=conversation,
                generation_kwargs=generation_kwargs or {},
            )
        raise RuntimeError(f"Unsupported BAGEL Qwen2-MoT inference phase: {phase!r}")

    def run_denoise_branch(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        validate_cfg_request(generation_kwargs or {})
        conversation = single_inference_conversation(conversation_list)
        return self._run_denoise_branch(
            conversation_list=conversation_list,
            conversation=conversation,
            generation_kwargs=generation_kwargs or {},
        )

    def collect_velocity(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        validate_cfg_request(generation_kwargs or {})
        conversation = single_inference_conversation(conversation_list)
        return self._collect_or_merge_velocity(
            conversation_list=conversation_list,
            conversation=conversation,
            generation_kwargs=generation_kwargs or {},
        )

    def _resolve_inference_phase(
        self,
        conversation: list[ConversationItem],
        generation_kwargs: dict[str, Any],
    ) -> str:
        state = self._generation_state
        state.infer_mode = str(generation_kwargs.get("infer_mode", state.infer_mode or "und"))
        if state.main.cache is None:
            return "prefill"
        tail = active_output_item(conversation)
        if tail is not None and torch.is_tensor(tail.value):
            value = tail.value
            if value.dim() == 3 and value.shape[0] == 1:
                value = value.squeeze(0)
            if tail.source == BAGEL_FLOW_QUERY:
                if value.dim() != 2:
                    raise ValueError(
                        f"BAGEL Qwen2-MoT denoise query expects rank-2 output tensors, got {tuple(value.shape)}."
                    )
                if int(value.shape[-1]) != int(self.config.hidden_size):
                    raise ValueError(
                        "BAGEL Qwen2-MoT denoise query hidden-size mismatch: "
                        f"got {value.shape[-1]}, expected {self.config.hidden_size}."
                    )
                return "denoise_branch"
            if tail.source == BAGEL_FLOW_VELOCITY:
                if value.dim() != 2:
                    raise ValueError(
                        f"BAGEL Qwen2-MoT velocity collection expects rank-2 output tensors, got {tuple(value.shape)}."
                    )
                return "velocity_collect"
            if state.infer_mode == "gen" and value.dim() != 2:
                raise ValueError(f"BAGEL Qwen2-MoT infer_gen expects rank-2 output tensors, got {tuple(value.shape)}.")
        if state.infer_mode != "gen":
            return "text_decode"
        raise ValueError("BAGEL Qwen2-MoT infer_gen requires a current output item.")

    def _prefill_inference(
        self,
        *,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]],
        conversation: list[ConversationItem],
        generation_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        packed = pack_training_conversation(
            [conversation],
            device=self.device,
            dtype=self.dtype,
            hidden_size=int(self.config.hidden_size),
        )
        if packed is None:
            raise ValueError("BAGEL Qwen2-MoT generate requires at least one embedded text/image item.")

        state = self._generation_state
        past_key_values = state.main.cache
        key_values_lens: torch.Tensor | None = None
        packed_key_value_indexes: torch.Tensor | None = None
        outputs = None
        cache_len = 0
        for span in packed.spans:
            span_end = span.start + span.length
            query_lens = torch.tensor([span.length], device=self.device, dtype=torch.int32)
            packed_query_indexes = torch.arange(span.start, span_end, device=self.device, dtype=torch.long)
            if span.item.type == "text":
                self._snapshot_cfg_text_context(
                    past_key_values=past_key_values,
                    key_values_lens=key_values_lens,
                    packed_key_value_indexes=packed_key_value_indexes,
                    next_position_id=packed.packed_position_ids[span.start],
                )
                self._update_cfg_img_text_context(span=span, packed=packed)
            call_kwargs = {
                "packed_query_sequence": packed.packed_sequence[span.start : span_end],
                "query_lens": query_lens,
                "packed_query_position_ids": packed.packed_position_ids[span.start : span_end],
                "packed_query_indexes": packed_query_indexes,
                "past_key_values": past_key_values,
                "key_values_lens": key_values_lens,
                "packed_key_value_indexes": packed_key_value_indexes,
                "update_past_key_values": True,
                "is_causal": span.item.type == "text",
                "mode": "und",
            }
            if span.item.type == "output":
                if span.length < 3:
                    raise ValueError("BAGEL Qwen2-MoT VAE context output must include start/end marker embeddings.")
                call_kwargs["is_causal"] = False
                call_kwargs["mode"] = "gen"
                call_kwargs["packed_text_indexes"] = torch.tensor([0, span.length - 1], device=self.device)
                call_kwargs["packed_vae_token_indexes"] = torch.arange(
                    1,
                    span.length - 1,
                    device=self.device,
                    dtype=torch.long,
                )
            outputs = self(
                **call_kwargs,
            )
            past_key_values = outputs["past_key_values"]
            cache_len += span.length
            key_values_lens = torch.tensor([cache_len], device=self.device, dtype=torch.int32)
            packed_key_value_indexes = torch.arange(cache_len, device=self.device, dtype=torch.long)
            if span.item.type == "image":
                self._snapshot_cfg_text_context(
                    past_key_values=past_key_values,
                    key_values_lens=key_values_lens,
                    packed_key_value_indexes=packed_key_value_indexes,
                    next_position_id=packed.packed_position_ids[span.start] + 1,
                )

        if outputs is None:
            raise RuntimeError("BAGEL Qwen2-MoT prefill produced no outputs.")
        state.main.set(
            cache=outputs["past_key_values"],
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            next_position_ids=next_position_ids_from_packed(packed),
        )

        if str(generation_kwargs.get("infer_mode", state.infer_mode or "und")) != "gen":
            conversation.append(
                ConversationItem(
                    type="output",
                    value=tail_hidden_from_packed(outputs["hidden_states"]),
                    role="assistant",
                )
            )
        return {"conversation_list": conversation_list}

    def _decode_text_step(
        self,
        *,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]],
        conversation: list[ConversationItem],
    ) -> dict[str, Any]:
        main_context = self._generation_state.main
        main_context.require_ready()
        if main_context.key_values_lens is None or main_context.packed_key_value_indexes is None:
            raise RuntimeError("BAGEL Qwen2-MoT inference cache is incomplete; reset and prefill before decoding.")
        tail = conversation[-1]
        if tail.type != "output":
            raise ValueError(f"BAGEL Qwen2-MoT decode expects tail output item, got {tail.type!r}.")

        packed_query_sequence = tail_query_embedding(tail).to(device=self.device, dtype=self.dtype)
        query_lens = torch.ones_like(main_context.key_values_lens)
        packed_query_indexes = torch.cumsum(main_context.key_values_lens, dim=0) + torch.arange(
            len(main_context.key_values_lens),
            device=self.device,
            dtype=main_context.key_values_lens.dtype,
        )
        packed_key_value_indexes = shift_packed_key_value_indexes(
            main_context.packed_key_value_indexes,
            main_context.key_values_lens,
        )
        outputs = self(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=main_context.next_position_ids,
            packed_query_indexes=packed_query_indexes.to(dtype=torch.long),
            past_key_values=main_context.cache,
            key_values_lens=main_context.key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        main_context.cache = outputs["past_key_values"]
        main_context.packed_key_value_indexes = append_query_indexes_to_cache(
            packed_key_value_indexes,
            main_context.key_values_lens,
        )
        main_context.key_values_lens = main_context.key_values_lens + query_lens
        main_context.next_position_ids = main_context.next_position_ids + query_lens

        conversation.append(
            ConversationItem(
                type="output",
                value=tail_hidden_from_packed(outputs["hidden_states"]),
                role="assistant",
            )
        )
        return {"conversation_list": conversation_list}

    def _run_denoise_branch(
        self,
        *,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]],
        conversation: list[ConversationItem],
        generation_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        validate_cfg_request(generation_kwargs)
        self._generation_state.main.require_ready()
        tail = active_output_item(conversation, sources={BAGEL_FLOW_QUERY})
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

        query = query.to(device=self.device, dtype=self.dtype)
        query_len = int(query.shape[0])
        branch_context = self._generation_state.branch_context()
        cache_len = branch_context.cache_len()
        query_lens = torch.tensor([query_len], device=self.device, dtype=torch.int32)
        packed_query_indexes = torch.arange(cache_len, cache_len + query_len, device=self.device, dtype=torch.long)
        packed_position_ids = branch_context.position_ids(query_len, device=self.device)
        packed_text_indexes = torch.tensor([0, query_len - 1], device=self.device, dtype=torch.long)
        packed_vae_token_indexes = torch.arange(1, query_len - 1, device=self.device, dtype=torch.long)
        if branch_context.cache is None:
            raise RuntimeError(f"BAGEL {branch_context.name} branch cache is not initialized.")
        outputs = self(
            packed_query_sequence=query,
            query_lens=query_lens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=branch_context.cache,
            key_values_lens=branch_context.key_values_lens,
            packed_key_value_indexes=branch_context.packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            mode="gen",
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        materialize_carrier_updates(
            None,
            [
                replace_fields(
                    tail,
                    source=BAGEL_FLOW_HIDDEN,
                    value=outputs["hidden_states"].to(device=self.device, dtype=self.dtype),
                )
            ],
        )
        return {"conversation_list": conversation_list}

    def _collect_or_merge_velocity(
        self,
        *,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]],
        conversation: list[ConversationItem],
        generation_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        validate_cfg_request(generation_kwargs)
        tail = active_output_item(conversation, sources={BAGEL_FLOW_VELOCITY})
        if tail is None or not torch.is_tensor(tail.value):
            raise ValueError("BAGEL Qwen2-MoT velocity collection requires source='bagel_flow_velocity'.")
        velocity = tail.value
        if velocity.dim() == 3 and velocity.shape[0] == 1:
            velocity = velocity.squeeze(0)
        if velocity.dim() != 2:
            raise ValueError(
                f"BAGEL Qwen2-MoT velocity collection expects rank-2 velocity, got {tuple(velocity.shape)}."
            )
        state = self._generation_state
        branches = cfg_branch_sequence(generation_kwargs, tail.meta.get("timestep"))
        collect_state = state.collect_velocity(
            velocity,
            branches,
            device=self.device,
            dtype=self.dtype,
        )
        if collect_state == "ready":
            return {"conversation_list": conversation_list}
        if collect_state == "need_branch":
            tail.value = None
            return {"conversation_list": conversation_list, FSM_SIGNAL_KEY: SIGNAL_NEED_DENOISE_BRANCH}

        main_velocity, cfg_text_velocity, cfg_img_velocity = state.velocities_for_merge()
        tail.value = merge_cfg_velocity(
            main_velocity=main_velocity,
            cfg_text_velocity=cfg_text_velocity.to(device=main_velocity.device, dtype=main_velocity.dtype),
            cfg_img_velocity=(
                cfg_img_velocity.to(device=main_velocity.device, dtype=main_velocity.dtype)
                if cfg_img_velocity is not None
                else None
            ),
            generation_kwargs=generation_kwargs,
        ).to(device=self.device)
        state.finish_velocity_round()
        return {"conversation_list": conversation_list}

    def _snapshot_cfg_text_context(
        self,
        *,
        past_key_values: Any,
        key_values_lens: torch.Tensor | None,
        packed_key_value_indexes: torch.Tensor | None,
        next_position_id: torch.Tensor,
    ) -> None:
        self._generation_state.cfg_text.snapshot(
            cache=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            next_position_id=next_position_id,
            empty_cache_factory=self._new_empty_cache,
            device=self.device,
        )

    def _ensure_cfg_img_context(self) -> None:
        self._generation_state.cfg_img.ensure_empty(empty_cache_factory=self._new_empty_cache, device=self.device)

    def _update_cfg_img_text_context(self, *, span: Any, packed: PackedConversation) -> None:
        self._ensure_cfg_img_context()
        cfg_img_context = self._generation_state.cfg_img
        cfg_img_context.require_ready()
        if cfg_img_context.key_values_lens is None or cfg_img_context.packed_key_value_indexes is None:
            raise RuntimeError("BAGEL image CFG text context failed to initialize.")
        span_end = span.start + span.length
        cache_len = cfg_img_context.cache_len()
        query_lens = torch.tensor([span.length], device=self.device, dtype=torch.int32)
        packed_query_indexes = torch.arange(cache_len, cache_len + span.length, device=self.device, dtype=torch.long)
        packed_position_ids = cfg_img_context.next_position_ids.reshape(1) + torch.arange(
            span.length,
            device=self.device,
            dtype=torch.long,
        )
        outputs = self(
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
        cfg_img_context.cache = outputs["past_key_values"]
        cfg_img_context.key_values_lens = cfg_img_context.key_values_lens + query_lens
        cfg_img_context.packed_key_value_indexes = torch.arange(
            int(cfg_img_context.key_values_lens.sum().item()),
            device=self.device,
            dtype=torch.long,
        )
        cfg_img_context.next_position_ids = cfg_img_context.next_position_ids + query_lens

    def _new_empty_cache(self) -> Any:
        try:
            from .modeling import NaiveCache
        except ImportError as exc:
            raise RuntimeError("Unable to import BAGEL NaiveCache for text CFG context initialization.") from exc
        return NaiveCache(len(self.model.layers))


__all__ = ["BagelQwen2MoTModuleMixin"]
