from typing import Any, Dict, List, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils.tensor_utils import naflatten, unflatten

from ....conversation import ConversationItem, is_dummy
from ....module import ModuleMixin, post_forward, pre_forward


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
        if get_parallel_state().sp_enabled:
            raise NotImplementedError("SP is not supported yet")

        packed = self._pack_conversations_for_forward(conversation_list)
        inputs_embeds = packed["inputs_embeds"]

        if self.training and get_parallel_state().fsdp_enabled:
            inputs_embeds = _fold_fsdp_dummy_anchors(inputs_embeds, conversation_list)

        self._conversation_carrier = conversation_list
        self._pack_inputs_embeds_shape = packed["inputs_embeds_shape"]

        return dict(
            inputs_embeds=inputs_embeds,
            position_ids=packed["position_ids"],
            visual_pos_masks=packed["visual_pos_masks"],
            deepstack_visual_embeds=packed["deepstack_visual_embeds"],
            cu_seq_lens_q=packed["cu_seq_lens"],
            cu_seq_lens_k=packed["cu_seq_lens"],
            max_length_q=packed["max_length"],
            max_length_k=packed["max_length"],
            **kwargs,
        )

    @post_forward("forward")
    def forward_post(self, **outputs: Any) -> Dict[str, Any]:
        if get_parallel_state().sp_enabled:
            raise NotImplementedError("SP is not supported yet")

        hidden_states = outputs.get("hidden_states")
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
        # NOTE: every forward below is invoked via ``self(...)`` (``__call__``),
        # not ``self.forward(...)``. Under FSDP2 inference the root pre-forward
        # hook must fire (lazy_init + unshard of root-owned params); calling
        # ``.forward`` directly skips it and the sharded DTensor params crash
        # with "FSDPCommContext has no all_gather_copy_in_stream" / mixed
        # Tensor and DTensor.
        if self._past_key_values is None:
            packed = self._pack_conversations_for_forward([conversation_list])
            position_ids = packed["position_ids"]
            outputs = self(
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
        outputs = self(
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
