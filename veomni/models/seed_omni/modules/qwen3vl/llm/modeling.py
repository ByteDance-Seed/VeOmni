"""Qwen3-VL AR backbone (no wte / lm_head, with DeepStack injection)."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils.device import IS_NPU_AVAILABLE
from veomni.utils.tensor_utils import naflatten

from ....utils.conversation import ConversationItem
from ...module_modeling_base import PretrainedOmniModule
from .configuration import Qwen3VLLlmConfig


if IS_NPU_AVAILABLE:
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_npu import Qwen3VLTextModel
else:
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import Qwen3VLTextModel


def _cumsum(values: list[int]) -> list[int]:
    out: list[int] = []
    total = 0
    for v in values:
        total += v
        out.append(total)
    return out


def qwen3vl_vision_position_advance(grid_thw: torch.Tensor, merge: int) -> int:
    """How far M-RoPE positions move past one visual item.

    A video is a run of ``t`` single-frame grids (see
    :func:`qwen3vl_vision_position_ids`), and each frame advances positions by
    ``max(h, w) // merge``, so the item advances ``t`` times that.
    """
    t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    return t * (max(h, w) // merge)


def qwen3vl_vision_position_ids(start: int, grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
    """3-row (t/h/w) M-RoPE positions for one visual item, mirroring HF ``get_rope_index``.

    Qwen3-VL separates video frames with timestamps, so HF splits a video's
    ``(T, H, W)`` grid into ``T`` single-frame grids *before* assigning
    positions: each frame restarts height/width and advances the running
    position by ``max(h, w) // merge``. Emitting the whole grid at once would
    share one temporal axis and one h/w range across all frames, which does not
    match the checkpoint's training distribution. An image is ``T == 1``, where
    both schemes coincide.
    """
    t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    gh, gw = h // merge, w // merge
    per_frame = gh * gw
    step = max(h, w) // merge
    # One single-frame grid per timestamp: its temporal coordinate is the frame's
    # own start position, and h/w restart there.
    frame_starts = [start + frame * step for frame in range(t)]
    temporal = torch.cat([torch.full((per_frame,), s) for s in frame_starts])
    height = torch.cat([torch.arange(gh).repeat_interleave(gw) + s for s in frame_starts])
    width = torch.cat([torch.arange(gw).repeat(gh) + s for s in frame_starts])
    return torch.stack([temporal, height, width], dim=0).long()


def collect_qwen3vl_dummy_deepstack(
    conversations: list[list[ConversationItem]],
    device: torch.device,
) -> Optional[List[torch.Tensor]]:
    for sample in conversations:
        for item in sample:
            if item.is_dummy and item.type == "image" and "deepstack" in item.meta:
                return [d.to(device) for d in item.meta["deepstack"]]
    return None


def pack_qwen3vl_conversations_for_forward(
    conversations: list[list[ConversationItem]],
    device: torch.device,
    spatial_merge_size: int,
) -> Dict[str, Any]:
    """Pack Qwen3-VL conversation embeds for backbone forward (training + inference)."""
    inputs_embeds_list: list[torch.Tensor] = []
    position_ids_list: list[torch.Tensor] = []
    visual_pos_masks_list: list[torch.Tensor] = []
    sample_lengths: list[int] = []
    deepstack_chunks: list[list[torch.Tensor]] = []

    for sample in conversations:
        sample_len = 0
        current_pos = 0
        for item in sample:
            if item.is_dummy:
                continue
            embeds = item.value.to(device)
            length = embeds.size(0)
            inputs_embeds_list.append(embeds)
            is_visual = item.type in ("image", "video")
            if is_visual:
                grid_thw = item.meta["grid_thw"]
                seg_pos = qwen3vl_vision_position_ids(current_pos, grid_thw, spatial_merge_size).to(device)
                current_pos += qwen3vl_vision_position_advance(grid_thw, spatial_merge_size)
                visual_pos_masks_list.append(torch.ones(length, dtype=torch.bool, device=device))
                deepstack_chunks.append([d.to(device) for d in item.meta["deepstack"]])
            else:
                seg_pos = torch.arange(length, dtype=torch.long, device=device).view(1, -1).expand(3, -1) + current_pos
                current_pos += length
                visual_pos_masks_list.append(torch.zeros(length, dtype=torch.bool, device=device))
            position_ids_list.append(seg_pos)
            sample_len += length
        sample_lengths.append(sample_len)

    inputs_embeds, inputs_embeds_shape = naflatten(inputs_embeds_list)
    if inputs_embeds.dim() == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)
    position_ids = torch.cat(position_ids_list, dim=1).unsqueeze(1)
    visual_pos_masks = torch.cat(visual_pos_masks_list, dim=0).unsqueeze(0)

    if deepstack_chunks:
        num_layers = len(deepstack_chunks[0])
        deepstack_visual_embeds = [
            torch.cat([chunk[layer] for chunk in deepstack_chunks], dim=0) for layer in range(num_layers)
        ]
    else:
        deepstack_visual_embeds = collect_qwen3vl_dummy_deepstack(conversations, device)
        visual_pos_masks = None
    if not deepstack_visual_embeds:
        # ``deepstack_visual_indexes: []`` (``disable_deepstack``) leaves every
        # item's ``deepstack`` list empty, so there is nothing to inject and the
        # backbone must not receive a mask that promises layer features.
        deepstack_visual_embeds = None
        visual_pos_masks = None

    cu_seq_lens = torch.tensor([0, *_cumsum(sample_lengths)], dtype=torch.int32, device=device)
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


class InferenceMixin:
    """FSM ``generate`` (with M-RoPE + DeepStack) — HF ``GenerationMixin`` analog.

    Listed *before* :class:`~...module_modeling_base.PretrainedOmniModule` in
    :class:`Qwen3VLLlm`'s bases: ``PretrainedOmniModule`` ships no-op
    ``reset_local_inference_state`` / ``reset_global_inference_state`` defaults
    (kept as a safety net for modules that don't need real inference state),
    and MRO resolves left-to-right — put second, those no-ops would shadow the
    real implementations below.
    """

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
        if get_parallel_state().sp_size > 1:
            raise ValueError(
                "Qwen3VLLlm.generate feeds the full prompt to the backbone, which conflicts with "
                "Ulysses sequence parallelism (attention expects this rank's shard); run inference "
                "with sp_size=1."
            )
        if self._past_key_values is None:
            packed = pack_qwen3vl_conversations_for_forward([conversation_list], self.device, self._spatial_merge_size)
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


class Qwen3VLLlm(InferenceMixin, PretrainedOmniModule):
    """Qwen3-VL text backbone (no wte, no lm_head).

    Token / image embeds are produced by the sibling ``qwen3vl_text_encoder`` /
    ``qwen3vl_vision`` modules and live on ``conversation_list`` items.
    :meth:`pre_forward` concatenates them per sample, rebuilds M-RoPE position
    ids, and threads the per-layer DeepStack features into the text model.
    """

    config_class = Qwen3VLLlmConfig
    base_model_prefix = "qwen3vl_llm"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]
    supports_gradient_checkpointing = True
    # This wrapper owns no attention — `self.language_model` does, and it supports
    # both SDPA and eager. Without this the base `_supports_sdpa = False` rejects
    # `sdpa` / `veomni_sdpa` at construction.
    _supports_sdpa = True

    def __init__(self, config: Qwen3VLLlmConfig):
        super().__init__(config)
        self.config = config
        self.language_model = Qwen3VLTextModel._from_config(self.config.text_config)
        self.language_model.set_input_embeddings(nn.Identity())
        self._past_key_values: Any = None
        self._next_position: int = 0
        self.post_init()

    @property
    def _spatial_merge_size(self) -> int:
        return self.config.spatial_merge_size

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        return {
            "hidden_states": outputs.last_hidden_state,
            "past_key_values": outputs.past_key_values,
        }
