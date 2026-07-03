from typing import Any, Dict, List, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_outputs,
    slice_input_tensor,
    sp_gather_seqs,
    sp_pad,
    sp_take_own_seq,
)
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids, valid_seqlens_from_cu_seqlens
from veomni.utils.tensor_utils import naflatten, unflatten

from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.modulemixin import ModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy
from .configuration import JanusLlamaConfig


class JanusLlamaModuleMixin(ModuleMixin):
    def init_omni_state(self) -> None:
        # Training state
        self._conversation_carrier: Optional[list[list[ConversationItem]]] = None
        self._pack_inputs_embeds_shape: Optional[torch.Tensor] = None
        # Combined module-SP-group packed length captured after the per-rank
        # aggregation but before the SP seq-pad/slice in ``forward_pre``;
        # ``forward_post`` gathers the hidden states back and trims to it.
        self._sp_seqlen: Optional[int] = None
        # Per-rank packed lengths + this rank's index within the module SP group,
        # used by ``forward_post`` to narrow the gathered full sequence back to
        # this rank's own segment.
        self._sp_rep_lengths: Optional[List[int]] = None
        self._sp_group_index: int = 0

        # Inference state
        self._cfg_active: bool = False
        self._past_key_values: Any = None
        self._uncond_past_key_values: Any = None

    # Training hooks
    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        inputs_embeds, attention_mask, position_ids, inputs_embeds_shape = self._pack_conversations_for_forward(
            conversation_list
        )

        if self.training and get_parallel_state().fsdp_enabled:
            inputs_embeds = _fold_fsdp_dummy_anchors(inputs_embeds, conversation_list)

        self._conversation_carrier = conversation_list
        self._pack_inputs_embeds_shape = inputs_embeds_shape

        # Metering: this rank's OWN per-sample lengths, from the
        # pre-gather / pre-slice packed ``position_ids``. ``OmniEnvironMeter``
        # sums tokens+FLOPs over the DP group only, so each rank must report just
        # its own data — NOT the module_sp ranks the module SP later aggregates
        # (that would over-count by ~module_sp). This keeps tokens/FLOPs identical
        # to the non-SP run for both SP-disabled and per-module SP.
        (_metric_cu, _), _ = prepare_fa_kwargs_from_position_ids(position_ids)
        self.metric_meter_set_seqlens("forward", [int(s) for s in valid_seqlens_from_cu_seqlens(_metric_cu).tolist()])

        self._sp_seqlen = None
        self._sp_rep_lengths = None
        self._sp_group_index = 0
        ps = get_parallel_state()
        if ps.sp_enabled:
            # 1) Per-rank aggregation: this module's SP group of module_sp ranks
            #    each holds a DISTINCT packed sequence. Concatenate them into the
            #    combined module-group sequence (autograd-aware).
            inputs_embeds, rep_lengths, group_index = sp_gather_seqs(inputs_embeds, dim=1)
            position_ids, _, _ = sp_gather_seqs(position_ids, dim=1)
            attention_mask, _, _ = sp_gather_seqs(attention_mask, dim=1)
            self._sp_rep_lengths = rep_lengths
            self._sp_group_index = group_index
            # 2) SP seq slice over the MODULE's own SP group: pad the full
            #    sequence to a multiple of module sp_size, build FA cu_seqlens
            #    over the FULL padded sequence (the attention all-to-all
            #    reconstructs it before the varlen kernel), then feed this rank
            #    only its sequence chunk. ``attention_mask`` stays full (all-ones)
            #    per the SP contract; ``forward_post`` gathers the hidden states.
            self._sp_seqlen = inputs_embeds.size(1)
            inputs_embeds = sp_pad(inputs_embeds, dim=1, pad_value=0)
            attention_mask = sp_pad(attention_mask, dim=1, pad_value=1)
            position_ids = sp_pad(position_ids, dim=1, pad_value=0)
            (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
                position_ids
            )
            inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, padding=False, group=ps.sp_group)
            position_ids = slice_input_tensor(position_ids, dim=1, padding=False, group=ps.sp_group)
        else:
            (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
                position_ids
            )
        return dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            **kwargs,
        )

    @post_forward("forward")
    def forward_post(self, **outputs: Any) -> Dict[str, Any]:
        hidden_states = outputs.get("hidden_states")

        ps = get_parallel_state()
        if ps.sp_enabled:
            # Gather the per-rank sequence chunks back into the combined
            # module-group packed sequence over the MODULE's SP group and drop the
            # SP padding tail (autograd-aware), then narrow back to this rank's own
            # segment so the carrier returns to the per-rank layout.
            hidden_states = gather_outputs(
                hidden_states,
                gather_dim=1,
                padding_dim=1,
                unpad_dim_size=self._sp_seqlen,
                group=ps.sp_group,
            )
            hidden_states = sp_take_own_seq(
                hidden_states, dim=1, seg_lengths=self._sp_rep_lengths, sp_rank=self._sp_group_index
            )
            self._sp_seqlen = None
            self._sp_rep_lengths = None

        conversation = self._conversation_carrier
        pack_shape = self._pack_inputs_embeds_shape
        self._conversation_carrier = None
        self._pack_inputs_embeds_shape = None

        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
        self._scatter_hidden_states(conversation, unflatten(hidden_states, pack_shape))
        return {"conversation_list": conversation}

    def _pack_conversations_for_forward(
        self,
        conversations: list[list[ConversationItem]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs_embeds_list = []
        attention_mask = []
        position_ids = []
        for sample in conversations:
            sample_lengths = 0
            for item in sample:
                role = item.role
                if role != "dummy":
                    embeds = item.value
                    embeds_length = embeds.size(0)
                    chunk_attention_mask = item.meta.pop("attention_mask", None)
                    if chunk_attention_mask is None:
                        chunk_attention_mask = torch.ones(embeds_length, dtype=torch.long, device=self.device)
                    inputs_embeds_list.append(embeds.to(self.device))
                    attention_mask.append(chunk_attention_mask.to(self.device))
                    sample_lengths += embeds_length
            sample_position_ids = torch.arange(sample_lengths, dtype=torch.long, device=self.device)
            position_ids.append(sample_position_ids)
        inputs_embeds, inputs_embeds_shape = naflatten(inputs_embeds_list)
        position_ids = torch.cat(position_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        return inputs_embeds, attention_mask, position_ids, inputs_embeds_shape

    def _scatter_hidden_states(
        self,
        conversation_list: list[list[ConversationItem]],
        hidden_states_list: list[torch.Tensor],
    ) -> None:
        hidden_states_list_iter = iter(hidden_states_list)
        for sample in conversation_list:
            for part in sample:
                if is_dummy(part):
                    continue
                part.value = next(hidden_states_list_iter)
        if next(hidden_states_list_iter, None) is not None:
            raise RuntimeError(
                "JanusLlama._scatter_hidden_states: segment count exceeds non-dummy conversation items."
            )

    # Inference hooks
    def reset_local_inference_state(self) -> None:
        self._cfg_active = False
        self._uncond_past_key_values = None

    def reset_global_inference_state(self) -> None:
        self.reset_local_inference_state()
        self._past_key_values = None

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # GenerationGraph invokes this endpoint through ``self.__call__`` so
        # FSDP/DDP hooks have already fired; its dispatch trampoline restores
        # the real ``forward`` while this endpoint runs.
        del kwargs
        if self._past_key_values is None:
            inputs_embeds, attention_mask, position_ids, _ = self._pack_conversations_for_forward([conversation_list])
            (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
                position_ids
            )

            outputs = self.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=self._past_key_values,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                use_cache=True,
            )
            self._past_key_values = outputs["past_key_values"]

            hidden_states = outputs["hidden_states"]
            conversation_list.append(
                ConversationItem(
                    type="output",
                    value=self._tail_hidden_from_forward(hidden_states),
                    role="assistant",
                )
            )
            return {"conversation_list": conversation_list}
        tail_part = conversation_list[-1]
        assert tail_part.type == "output"

        cfg_uncond_inputs_embeds = tail_part.meta.pop("cfg_uncond_inputs_embeds", None)
        if cfg_uncond_inputs_embeds is not None and not self._cfg_active:
            uncond = cfg_uncond_inputs_embeds.to(self.device)
            if uncond.dim() == 2:
                uncond = uncond.unsqueeze(0)
            uncond_out = self.forward(
                inputs_embeds=uncond,
                attention_mask=None,
                past_key_values=None,
                use_cache=True,
            )
            self._uncond_past_key_values = uncond_out["past_key_values"]
            self._cfg_active = True
        elif tail_part.meta.get("collapse_cfg", False):
            self._uncond_past_key_values = None
            self._cfg_active = False

        inputs_embeds: torch.Tensor = tail_part.value[-1:].to(self.device)
        inputs_embeds = inputs_embeds.unsqueeze(0)

        if self._cfg_active:
            cond_out = self.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                past_key_values=self._past_key_values,
                use_cache=True,
            )
            uncond_out = self.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                past_key_values=self._uncond_past_key_values,
                use_cache=True,
            )
            self._past_key_values = cond_out["past_key_values"]
            self._uncond_past_key_values = uncond_out["past_key_values"]
            hidden_states = torch.cat([cond_out["hidden_states"], uncond_out["hidden_states"]], dim=0)
        else:
            outputs = self.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                past_key_values=self._past_key_values,
                use_cache=True,
            )
            self._past_key_values = outputs["past_key_values"]
            hidden_states = outputs["hidden_states"]

        conversation_list.append(
            ConversationItem(
                type="output",
                value=self._tail_hidden_from_forward(hidden_states),
                role="assistant",
            )
        )
        return {"conversation_list": conversation_list}

    @staticmethod
    def _tail_hidden_from_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
            return hidden_states[-1:].contiguous()
        return hidden_states[:, -1:, :].contiguous()


def _fold_fsdp_dummy_anchors(
    inputs_embeds: torch.Tensor,
    conversations: list[list[ConversationItem]],
) -> torch.Tensor:
    for sample in conversations:
        for part in sample:
            if not is_dummy(part):
                continue
            if not isinstance(part.value, torch.Tensor):
                continue
            fake = part.value.mean().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype) * 0.0
            inputs_embeds = inputs_embeds + fake
    return inputs_embeds


class JanusLlamaMetricMeterMixin(MetricMeterMixin):
    """Per-module training meter for the Janus LLaMA backbone (transformer layers only)."""

    config: JanusLlamaConfig

    def estimate_flops(self, seqlens: List[int]) -> float:
        # Transformer layers only: this backbone owns no wte / lm_head (those
        # live in the text_encoder module), so we do NOT add a vocab projection.
        # fwd+bwd ⇒ 6x for the linear params, 12x for the quadratic attention.
        cfg = self.config.text_config
        hidden = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", hidden // num_heads)

        # SwiGLU MLP (gate/up/down) + attention projections (q, k, v, o).
        mlp_n = hidden * cfg.intermediate_size * 3
        attn_linear_n = hidden * (num_heads * head_dim * 2 + num_kv_heads * head_dim * 2)
        dense_n = (mlp_n + attn_linear_n) * num_layers

        tokens = sum(seqlens)
        seqlen_sq = sum(s * s for s in seqlens)
        dense_flops = 6 * dense_n * tokens
        attn_flops = 12 * seqlen_sq * head_dim * num_heads * num_layers
        return (dense_flops + attn_flops) / 1e12


__all__ = ["JanusLlamaModuleMixin", "JanusLlamaMetricMeterMixin"]
