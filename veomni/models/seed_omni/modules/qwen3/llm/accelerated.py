"""VeOmni-accelerated Qwen3Llm — training-graph hooks only.

``generate()`` and inference-state management live on the native
:class:`Qwen3Llm` in ``modeling.py`` — this file overrides them only if
FSDP/DDP/SP needs a different inference path (it doesn't, today).
"""

from typing import Any, Dict, List, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids, valid_seqlens_from_cu_seqlens
from veomni.utils.tensor_utils import unflatten

from ....mixins.base_mixin import BaseMixin
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy
from ...base.llm_packing import pack_llm_conversations_for_forward, scatter_llm_hidden_states
from .configuration import Qwen3LlmConfig
from .modeling import Qwen3Llm


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks — depends on :class:`Qwen3Llm` modeling APIs."""

    config: Qwen3LlmConfig
    device: torch.device
    training: bool

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Optional[list[list[ConversationItem]]] = None
        self._pack_inputs_embeds_shape: Optional[torch.Tensor] = None
        # Active sample's pre-pad packed length. Under SP the output-gather hook
        # (``forward_sp_post``) narrows the all-gathered (padded) output back to it.
        self._sp_own_len: Optional[int] = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        inputs_embeds, attention_mask, position_ids, inputs_embeds_shape = pack_llm_conversations_for_forward(
            conversation_list, self.device
        )

        if self.training and get_parallel_state().fsdp_enabled:
            inputs_embeds = _fold_fsdp_dummy_anchors(inputs_embeds, conversation_list)

        self._conversation_carrier = conversation_list
        self._pack_inputs_embeds_shape = inputs_embeds_shape

        # Metering: this rank's OWN per-sample lengths, from the packed
        # ``position_ids``. ``OmniEnvironMeter`` sums tokens+FLOPs over the DP group
        # only, so each rank reports just its own data — NOT the module_sp peers a
        # per-module SP forward redistributes (that would over-count by ~module_sp).
        # This keeps tokens/FLOPs identical to the non-SP run for both SP-disabled
        # and per-module SP.
        (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
            position_ids
        )
        self.metric_meter_set_seqlens(
            "forward", [int(s) for s in valid_seqlens_from_cu_seqlens(cu_seq_lens_q).tolist()]
        )

        if get_parallel_state().sp_size > 1:
            # SP input-slice: slice the (replicated) packed sample to this rank's
            # Ulysses shard (+ per-shard ``cu_seqlens``). The dataloader replicates
            # each shard across the SP group, so ``inputs_embeds`` / ``attention_mask``
            # / ``position_ids`` are the SAME on every rank. Pad them to a multiple of
            # ``sp_size``, rebuild FA varlen ``cu_seqlens`` over the padded sample (the
            # attention all-to-all reconstructs the full sequence before the kernel, so
            # mask/lengths stay full), then hand this rank only its ``1/sp_size`` chunk
            # (the full-sample ``cu_seqlens`` above are for metering / the non-SP path
            # only). ``forward_post`` all-gathers the shards back.
            group = get_parallel_state().sp_group
            self._sp_own_len = inputs_embeds.size(1)
            embeds = sp_pad(inputs_embeds, dim=1, pad_value=0)
            mask = sp_pad(attention_mask, dim=1, pad_value=1)
            pids = sp_pad(position_ids, dim=1, pad_value=0)
            (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(pids)
            embeds = slice_input_tensor(embeds, dim=1, padding=False, group=group)
            pids = slice_input_tensor(pids, dim=1, padding=False, group=group)
            return dict(
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=pids,
                cu_seq_lens_q=cu_q,
                cu_seq_lens_k=cu_k,
                max_length_q=max_q,
                max_length_k=max_k,
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

        if get_parallel_state().sp_size > 1:
            # SP output-gather: all-gather Ulysses shards back to the full sequence
            # (autograd-aware; backward sums grads across the SP group), then strip
            # the SP pad tail so the packed length matches its carrier below.
            hidden_states = gather_outputs(hidden_states, gather_dim=1, group=get_parallel_state().sp_group)
            hidden_states = hidden_states.narrow(1, 0, self._sp_own_len)

        conversation = self._conversation_carrier
        pack_shape = self._pack_inputs_embeds_shape
        self._conversation_carrier = None
        self._pack_inputs_embeds_shape = None

        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
        scatter_llm_hidden_states(conversation, unflatten(hidden_states, pack_shape))
        return {"conversation_list": conversation}


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for the Qwen3 backbone (transformer layers only)."""

    config: Qwen3LlmConfig

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


class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin):
    """``generate()`` and inference-state reset already live on the native
    :class:`~.modeling.Qwen3Llm` (via ``SimpleArGenerationMixin``), so no
    ``InferenceMixin`` is needed here.
    """


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


class Qwen3LlmAccelerated(VeOmniMixin, Qwen3Llm):
    pass


__all__ = ["Qwen3LlmAccelerated"]
