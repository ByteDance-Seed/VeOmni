from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_outputs,
    slice_input_tensor,
    sp_gather_seqs,
    sp_pad,
    sp_take_own_seq,
)
from veomni.utils.tensor_utils import naflatten, unflatten

from ....mixins.modulemixin import ModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy


class Qwen3VLLlmModuleMixin(ModuleMixin):
    """Graph hooks for the Qwen3-VL AR backbone.

    Packs every non-dummy ``conversation_list`` item's embedding segment into one
    bs=1 varlen sequence, rebuilds 3-row M-RoPE position ids (text runs +
    per-image grid positions), marks image positions for DeepStack, and threads
    the per-layer DeepStack features into ``Qwen3VLTextModel``.
    """

    def init_omni_state(self) -> None:
        self._conversation_carrier: Optional[list[list[ConversationItem]]] = None
        self._pack_inputs_embeds_shape: Optional[torch.Tensor] = None
        self._past_key_values: Any = None
        self._next_position: int = 0
        # Combined module-SP-group packed length captured after the per-rank
        # aggregation but before the SP seq-pad/slice in ``forward_pre``;
        # ``forward_post`` gathers the hidden states and trims to it.
        self._sp_seqlen: Optional[int] = None
        # Per-rank packed lengths + this rank's index within the module SP group,
        # used by ``forward_post`` to narrow the gathered full sequence back to
        # this rank's own segment (``sp_take_own_seq``).
        self._sp_rep_lengths: Optional[List[int]] = None
        self._sp_group_index: int = 0

    @property
    def _spatial_merge_size(self) -> int:
        return self.config.spatial_merge_size

    # ── Training hooks ──────────────────────────────────────────────────────
    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        packed = self._pack_conversations_for_forward(conversation_list)
        inputs_embeds = packed["inputs_embeds"]

        if self.training and get_parallel_state().fsdp_enabled:
            inputs_embeds = _fold_fsdp_dummy_anchors(inputs_embeds, conversation_list)

        self._conversation_carrier = conversation_list
        self._pack_inputs_embeds_shape = packed["inputs_embeds_shape"]

        position_ids = packed["position_ids"]
        visual_pos_masks = packed["visual_pos_masks"]
        deepstack_visual_embeds = packed["deepstack_visual_embeds"]
        cu_seq_lens = packed["cu_seq_lens"]
        max_length = packed["max_length"]

        self._sp_seqlen = None
        self._sp_rep_lengths = None
        self._sp_group_index = 0
        if get_parallel_state().sp_enabled:
            (
                inputs_embeds,
                position_ids,
                visual_pos_masks,
                deepstack_visual_embeds,
                cu_seq_lens,
                max_length,
            ) = self._sp_gather_and_slice_forward_inputs(
                inputs_embeds, position_ids, visual_pos_masks, deepstack_visual_embeds, cu_seq_lens, max_length
            )

        return dict(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=max_length,
            max_length_k=max_length,
            **kwargs,
        )

    def _sp_gather_and_slice_forward_inputs(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
        cu_seq_lens: torch.Tensor,
        max_length: int,
    ):
        """Aggregate the module SP group's distinct packed sequences, then SP-slice.

        Each SP rank packs a DISTINCT bs=1 sequence (the orchestrator runs
        SP-disabled). We first concatenate the group's sequences into one combined
        packed sequence — ``inputs_embeds``, 3-row M-RoPE ``position_ids``,
        ``visual_pos_masks`` and the per-layer DeepStack visual embeds in lockstep
        — rebuild the varlen ``cu_seqlens`` from the combined per-sample lengths,
        then pad the combined sequence to a multiple of sp_size and feed this rank
        only its chunk (the attention all-to-all reconstructs the full sequence).
        ``forward_post`` gathers the hidden states and narrows back to this rank's
        own segment.

        A "text-only" rank carries ``visual_pos_masks=None`` + a dummy DeepStack
        anchor; if ANY rank in the group has real visuals (MAX-reduced) the
        combined sequence uses the real DeepStack-injection path — text-only ranks
        then contribute an all-False mask + empty visual rows (their dummy anchor
        fixes num_layers + hidden dim) — otherwise the all-dummy anchor path is
        kept.
        """
        ps = get_parallel_state()
        sp_size, sp_rank = ps.sp_size, ps.sp_rank
        group = ps.sp_group

        local_len = inputs_embeds.size(1)

        # Combined per-sample lengths (host metadata) → combined varlen cu_seqlens.
        local_sample_lengths = [int(x) for x in (cu_seq_lens[1:] - cu_seq_lens[:-1]).tolist()]
        gathered_sample_lengths: List[Optional[List[int]]] = [None] * sp_size
        dist.all_gather_object(gathered_sample_lengths, local_sample_lengths, group=group)
        combined_sample_lengths = [s for lst in gathered_sample_lengths for s in lst]
        cu_seq_lens = torch.tensor([0, *_cumsum(combined_sample_lengths)], dtype=torch.int32, device=self.device)
        max_length = max(combined_sample_lengths) if combined_sample_lengths else 0

        # Reconcile whether the COMBINED sequence has any real visual positions.
        has_visual = torch.tensor([0 if visual_pos_masks is None else 1], device=self.device)
        dist.all_reduce(has_visual, op=dist.ReduceOp.MAX, group=group)
        combined_has_visual = bool(has_visual.item())

        # Aggregate the distinct per-rank sequences (autograd-aware for embeds).
        inputs_embeds, rep_lengths, group_index = sp_gather_seqs(inputs_embeds, dim=1)
        position_ids, _, _ = sp_gather_seqs(position_ids, dim=2)
        self._sp_rep_lengths = rep_lengths
        self._sp_group_index = group_index

        if combined_has_visual:
            if visual_pos_masks is None:
                # Text-only rank: contribute an all-False mask + empty (0, D) rows;
                # its dummy DeepStack anchor fixes num_layers + hidden dim.
                visual_pos_masks = torch.zeros(1, local_len, dtype=torch.bool, device=self.device)
                deepstack_visual_embeds = [layer[:0] for layer in deepstack_visual_embeds]
            visual_pos_masks, _, _ = sp_gather_seqs(visual_pos_masks, dim=1)
            deepstack_visual_embeds = [sp_gather_seqs(layer, dim=0)[0] for layer in deepstack_visual_embeds]
        else:
            # All-dummy combined batch: keep the None + per-rank dummy-anchor path.
            visual_pos_masks = None

        # ── SP seq slice over the combined sequence ──
        self._sp_seqlen = inputs_embeds.size(1)
        inputs_embeds = sp_pad(inputs_embeds, dim=1, pad_value=0)
        position_ids = sp_pad(position_ids, dim=2, pad_value=0)
        padded_len = inputs_embeds.size(1)
        pad_len = padded_len - self._sp_seqlen
        if pad_len > 0:
            tail = torch.tensor([padded_len], dtype=cu_seq_lens.dtype, device=cu_seq_lens.device)
            cu_seq_lens = torch.cat([cu_seq_lens, tail], dim=0)
            max_length = max(max_length, pad_len)

        if visual_pos_masks is not None:
            visual_pos_masks = sp_pad(visual_pos_masks, dim=1, pad_value=0)
            unit = padded_len // sp_size
            n_before = int(visual_pos_masks[:, : unit * sp_rank].sum().item())
            visual_pos_masks = slice_input_tensor(visual_pos_masks, dim=1, padding=False, group=group)
            k_local = int(visual_pos_masks.sum().item())
            deepstack_visual_embeds = [layer[n_before : n_before + k_local] for layer in deepstack_visual_embeds]

        inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, padding=False, group=group)
        position_ids = slice_input_tensor(position_ids, dim=2, padding=False, group=group)
        return inputs_embeds, position_ids, visual_pos_masks, deepstack_visual_embeds, cu_seq_lens, max_length

    @post_forward("forward")
    def forward_post(self, **outputs: Any) -> Dict[str, Any]:
        hidden_states = outputs.get("hidden_states")

        ps = get_parallel_state()
        if ps.sp_enabled:
            # Gather the per-rank sequence chunks back into the full combined
            # packed sequence over the MODULE's SP group and drop the SP padding
            # tail (autograd-aware), then narrow back to this rank's own segment
            # so the carrier returns to the per-rank layout.
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
            self._sp_group_index = 0

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
    ) -> Dict[str, Any]:
        inputs_embeds_list: list[torch.Tensor] = []
        position_ids_list: list[torch.Tensor] = []
        visual_pos_masks_list: list[torch.Tensor] = []
        sample_lengths: list[int] = []
        deepstack_chunks: list[list[torch.Tensor]] = []  # per real-image: list[layer] of (N_i, D)

        for sample in conversations:
            sample_len = 0
            current_pos = 0
            for item in sample:
                if is_dummy(item):
                    continue
                embeds = item.value.to(self.device)
                length = embeds.size(0)
                inputs_embeds_list.append(embeds)
                is_visual = item.type in ("image", "video")
                if is_visual:
                    grid_thw = item.meta["grid_thw"]
                    seg_pos = self._vision_position_ids(current_pos, grid_thw, self._spatial_merge_size).to(
                        self.device
                    )
                    current_pos += int(max(int(grid_thw[1]), int(grid_thw[2])) // self._spatial_merge_size)
                    visual_pos_masks_list.append(torch.ones(length, dtype=torch.bool, device=self.device))
                    deepstack_chunks.append([d.to(self.device) for d in item.meta["deepstack"]])
                else:
                    seg_pos = (
                        torch.arange(length, dtype=torch.long, device=self.device).view(1, -1).expand(3, -1)
                        + current_pos
                    )
                    current_pos += length
                    visual_pos_masks_list.append(torch.zeros(length, dtype=torch.bool, device=self.device))
                position_ids_list.append(seg_pos)
                sample_len += length
            sample_lengths.append(sample_len)

        inputs_embeds, inputs_embeds_shape = naflatten(inputs_embeds_list)
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        position_ids = torch.cat(position_ids_list, dim=1).unsqueeze(1)  # (3, 1, total)
        visual_pos_masks = torch.cat(visual_pos_masks_list, dim=0).unsqueeze(0)  # (1, total)

        if deepstack_chunks:
            num_layers = len(deepstack_chunks[0])
            deepstack_visual_embeds = [
                torch.cat([chunk[layer] for chunk in deepstack_chunks], dim=0) for layer in range(num_layers)
            ]
        else:
            # All-dummy (text-only) micro-batch: keep the DeepStack mergers on the
            # FSDP grad graph via the visual_pos_masks=None add-0.0 path.
            deepstack_visual_embeds = self._collect_dummy_deepstack(conversations)
            visual_pos_masks = None

        cu_seq_lens = torch.tensor([0, *_cumsum(sample_lengths)], dtype=torch.int32, device=self.device)
        max_length = max(sample_lengths) if sample_lengths else 0

        return {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": deepstack_visual_embeds,
            "cu_seq_lens": cu_seq_lens,
            "max_length": max_length,
            "inputs_embeds_shape": inputs_embeds_shape,
        }

    def _collect_dummy_deepstack(
        self,
        conversations: list[list[ConversationItem]],
    ) -> Optional[List[torch.Tensor]]:
        for sample in conversations:
            for item in sample:
                if is_dummy(item) and item.type == "image" and "deepstack" in item.meta:
                    return [d.to(self.device) for d in item.meta["deepstack"]]
        return None

    @staticmethod
    def _vision_position_ids(start: int, grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
        """3-row (t/h/w) M-RoPE positions for one image — mirrors Qwen3VLModel.get_vision_position_ids."""
        t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
        gt, gh, gw = t, h // merge, w // merge
        temporal = torch.arange(gt).repeat_interleave(gh * gw) + start
        height = torch.arange(gh).repeat_interleave(gw).repeat(gt) + start
        width = torch.arange(gw).repeat(gh * gt) + start
        return torch.stack([temporal, height, width], dim=0).long()

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
            raise RuntimeError("Qwen3VLLlm._scatter_hidden_states: segment count exceeds non-dummy items.")

    # ── Inference hooks ─────────────────────────────────────────────────────
    def reset_local_inference_state(self) -> None:
        return

    def reset_global_inference_state(self) -> None:
        self.reset_local_inference_state()
        self._past_key_values = None
        self._next_position = 0

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        # GenerationGraph invokes this endpoint through ``self.__call__`` so
        # FSDP/DDP hooks have already fired; its dispatch trampoline restores
        # the real ``forward`` while this endpoint runs.
        if self._past_key_values is None:
            packed = self._pack_conversations_for_forward([conversation_list])
            position_ids = packed["position_ids"]
            outputs = self.forward(
                inputs_embeds=packed["inputs_embeds"],
                attention_mask=None,
                position_ids=position_ids,
                visual_pos_masks=packed["visual_pos_masks"],
                deepstack_visual_embeds=packed["deepstack_visual_embeds"],
                past_key_values=self._past_key_values,
                cu_seq_lens_q=packed["cu_seq_lens"],
                cu_seq_lens_k=packed["cu_seq_lens"],
                max_length_q=packed["max_length"],
                max_length_k=packed["max_length"],
                use_cache=True,
            )
            self._past_key_values = outputs["past_key_values"]
            self._next_position = int(position_ids.max()) + 1
            conversation_list.append(
                ConversationItem(
                    type="output",
                    value=self._tail_hidden_from_forward(outputs["hidden_states"]),
                    role="assistant",
                )
            )
            return {"conversation_list": conversation_list}

        tail_part = conversation_list[-1]
        assert tail_part.type == "output"
        inputs_embeds = tail_part.value[-1:].to(self.device).unsqueeze(0)
        position_ids = torch.full((3, 1, 1), self._next_position, dtype=torch.long, device=self.device)
        outputs = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=self._past_key_values,
            use_cache=True,
        )
        self._past_key_values = outputs["past_key_values"]
        self._next_position += 1
        conversation_list.append(
            ConversationItem(
                type="output",
                value=self._tail_hidden_from_forward(outputs["hidden_states"]),
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


def _cumsum(values: list[int]) -> list[int]:
    out: list[int] = []
    total = 0
    for v in values:
        total += v
        out.append(total)
    return out


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


__all__ = ["Qwen3VLLlmModuleMixin"]
