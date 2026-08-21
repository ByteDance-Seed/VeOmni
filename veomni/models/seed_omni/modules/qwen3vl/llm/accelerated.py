"""VeOmni-accelerated Qwen3VLLlm — training-graph hooks only.

``generate()`` and inference-state management live on the native
:class:`Qwen3VLLlm` in ``modeling.py`` — this file overrides them only if
FSDP/DDP/SP needs a different inference path (it doesn't, today).
"""

from typing import Any, Dict, List, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad
from veomni.utils.tensor_utils import unflatten

from ....mixins.base_mixin import BaseMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy
from ...base.llm_packing import scatter_llm_hidden_states
from .configuration import Qwen3VLLlmConfig
from .modeling import Qwen3VLLlm, pack_qwen3vl_conversations_for_forward


class TrainingMixin(TrainingModuleMixin):
    """Graph hooks for the Qwen3-VL AR backbone (training path).

    Packs every non-dummy ``conversation_list`` item's embedding segment into one
    bs=1 varlen sequence, rebuilds 3-row M-RoPE position ids (text runs +
    per-image grid positions), marks image positions for DeepStack, and threads
    the per-layer DeepStack features into ``Qwen3VLTextModel``.
    """

    config: Qwen3VLLlmConfig
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
        packed = pack_qwen3vl_conversations_for_forward(conversation_list, self.device, self._spatial_merge_size)
        inputs_embeds = packed["inputs_embeds"]

        if self.training and get_parallel_state().fsdp_enabled:
            inputs_embeds = _fold_fsdp_dummy_anchors(inputs_embeds, conversation_list)

        self._conversation_carrier = conversation_list
        self._pack_inputs_embeds_shape = packed["inputs_embeds_shape"]

        # Normalize the visual side to always-real tensors: the SP input-slice branch
        # below slices ``visual_pos_masks`` and indexes the per-layer DeepStack embeds,
        # so they must be real tensors (not Python ``None`` / lists) on every rank. So
        # ``visual_pos_masks`` becomes a real (possibly all-False) bool tensor and the
        # per-layer DeepStack embeds become one stacked ``(num_layers, N, D)`` tensor
        # (the generated modeling indexes it per layer exactly like the list). The old
        # cross-rank reconciliation (all_gather_object cu_seqlens, MAX-reduce
        # has_visual) is gone — each sample is self-describing.
        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._normalize_visual_inputs(
            inputs_embeds, packed["visual_pos_masks"], packed["deepstack_visual_embeds"]
        )

        if get_parallel_state().sp_size > 1:
            # SP input-slice: slice the (replicated) packed sample to this rank's
            # Ulysses shard, including the M-RoPE positions and DeepStack rows. Every
            # SP rank holds the active sample's exact ``forward_pre`` output (the
            # dataloader replicates each shard). Pad the sample to a multiple of
            # ``sp_size``, extend the full-sample varlen ``cu_seqlens`` with the pad
            # tail (the attention all-to-all reconstructs the full sequence before the
            # kernel), then hand this rank only its ``1/sp_size`` chunk of embeds
            # (dim 1), 3-row positions (dim 2) and ``visual_pos_masks`` (dim 1). The
            # DeepStack embeds are indexed by visual position, so slice out exactly
            # the rows whose visual tokens fall in this rank's chunk (``n_before``
            # skipped + ``k_local`` kept). ``forward_post`` all-gathers the output.
            ps = get_parallel_state()
            group, sp_size, sp_rank = ps.sp_group, ps.sp_size, ps.sp_rank

            self._sp_own_len = inputs_embeds.size(1)
            cu = packed["cu_seq_lens"]
            max_length = int((cu[1:] - cu[:-1]).max().item()) if cu.numel() > 1 else 0

            embeds = sp_pad(inputs_embeds, dim=1, pad_value=0)
            pids = sp_pad(packed["position_ids"], dim=2, pad_value=0)
            mask = sp_pad(visual_pos_masks, dim=1, pad_value=0)
            padded_len = embeds.size(1)
            pad_len = padded_len - self._sp_own_len
            if pad_len > 0:
                tail = torch.tensor([padded_len], dtype=cu.dtype, device=cu.device)
                cu = torch.cat([cu, tail], dim=0)
                max_length = max(max_length, pad_len)

            unit = padded_len // sp_size
            n_before = int(mask[:, : unit * sp_rank].sum().item())
            embeds = slice_input_tensor(embeds, dim=1, padding=False, group=group)
            pids = slice_input_tensor(pids, dim=2, padding=False, group=group)
            mask = slice_input_tensor(mask, dim=1, padding=False, group=group)
            if deepstack_visual_embeds is not None:
                k_local = int(mask.sum().item())
                deepstack_visual_embeds = deepstack_visual_embeds[:, n_before : n_before + k_local, :]

            return dict(
                inputs_embeds=embeds,
                position_ids=pids,
                visual_pos_masks=mask,
                deepstack_visual_embeds=deepstack_visual_embeds,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=max_length,
                max_length_k=max_length,
            )

        return dict(
            inputs_embeds=inputs_embeds,
            position_ids=packed["position_ids"],
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            cu_seq_lens_q=packed["cu_seq_lens"],
            cu_seq_lens_k=packed["cu_seq_lens"],
            max_length_q=packed["max_length"],
            max_length_k=packed["max_length"],
            **kwargs,
        )

    def _normalize_visual_inputs(
        self,
        inputs_embeds: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Turn the packed visual side into always-real tensors (see ``forward_pre``).

        Real visuals: stack the per-layer DeepStack embeds into ``(num_layers, N, D)``.
        All-text sample: return an all-False mask + an EMPTY ``(num_layers, 0, D)``
        deepstack, and fold the dummy DeepStack anchor into ``inputs_embeds`` so the
        vision mergers that produced it still get (zero) gradient. This replaces the
        modeling's ``visual_pos_masks=None`` anchor path, which we can no longer use
        because the SP input-slice needs a real mask on every rank.
        """
        if visual_pos_masks is not None:
            return inputs_embeds, visual_pos_masks, torch.stack(deepstack_visual_embeds, dim=0)

        mask = torch.zeros(1, inputs_embeds.size(1), dtype=torch.bool, device=self.device)
        if not deepstack_visual_embeds:
            # No DeepStack layers to anchor (e.g. text-only run without vision).
            return inputs_embeds, mask, None
        num_layers = len(deepstack_visual_embeds)
        hidden_dim = deepstack_visual_embeds[0].size(-1)
        anchor = sum(layer.float().mean() for layer in deepstack_visual_embeds) * 0.0
        inputs_embeds = inputs_embeds + anchor.to(inputs_embeds.dtype)
        empty_deepstack = inputs_embeds.new_zeros(num_layers, 0, hidden_dim)
        return inputs_embeds, mask, empty_deepstack

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


class VeOmniMixin(BaseMixin, TrainingMixin):
    """``_spatial_merge_size`` is inherited from the native :class:`Qwen3VLLlm`.

    ``generate()`` and inference-state reset already live on that native class
    (via its own :class:`~.modeling.InferenceMixin`), so no ``InferenceMixin``
    is needed here.
    """


def _fold_fsdp_dummy_anchors(
    inputs_embeds: torch.Tensor,
    conversations: list[list[ConversationItem]],
) -> torch.Tensor:
    for sample in conversations:
        for part in sample:
            if not is_dummy(part):
                continue
            if isinstance(part.value, torch.Tensor):
                inputs_embeds = (
                    inputs_embeds + part.value.mean().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype) * 0.0
                )
    return inputs_embeds


class Qwen3VLLlmAccelerated(VeOmniMixin, Qwen3VLLlm):
    pass


__all__ = ["Qwen3VLLlmAccelerated"]
