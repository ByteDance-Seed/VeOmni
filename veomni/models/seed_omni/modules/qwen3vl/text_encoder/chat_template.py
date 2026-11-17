"""Qwen3-VL ChatML template (text + image + video) as readable Python.

Mirrors the upstream ``chat_template.json``:

* each turn is wrapped in ``<|im_start|>{role}\\n … <|im_end|>\\n``;
* image / video become ``<|vision_start|><|image_pad|><|vision_end|>`` /
  ``<|vision_start|><|video_pad|><|vision_end|>`` — in the SeedOmni segment model the
  ``<|*_pad|>`` run is *not* tokenized; the sibling ``image`` / ``video`` item
  already carries the merged vision tokens, so the template emits
  ``<|vision_start|>`` text · the media item · ``<|vision_end|>`` text.

Qwen3-VL has no audio modality (audio-in-video is an Omni feature — see
``docs/seed_omni/design/media.md``, design-only).

Reuses :class:`TextEncoderChatTemplate` for tokenize / merge / pack; only the
ChatML templating (:meth:`Qwen3VLChatTemplate.apply_chat_template`) and the
generation prompt are model-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ......data.seed_omni.utils.media_metadata import VIDEO_METADATA_KEY
from ....utils.conversation import _OMNI_TOKEN_LEN, ConversationItem
from ...text_encoder.chat_template import TextEncoderChatTemplate
from ..vision.processing import _OMNI_GRID


def _frame_timestamps(item: ConversationItem, num_patches: int, temporal_patch_size: int) -> list[float]:
    """When each temporal patch happens, in seconds on the source clip.

    Delegates to :meth:`VideoMetadata.frame_timestamps`, which folds frames into
    patches exactly as ``Qwen3VLProcessor._calculate_timestamps`` does — pad to a
    whole number of patches by repeating the last frame, then take each patch's
    mid-time. Dividing ``frames_indices`` by the *source* fps is the trap that
    helper exists to encapsulate.

    Without decode metadata there are no real times to report. The sequence then
    keeps the right *shape* (one stamp per patch, which is what the M-RoPE layout
    depends on) but the values are synthetic.
    """
    metadata = (item.meta or {}).get(VIDEO_METADATA_KEY)
    if metadata is not None:
        stamps = metadata.frame_timestamps(int(temporal_patch_size))
        if len(stamps) >= num_patches:
            return stamps[:num_patches]
    patch = max(1, int(temporal_patch_size))
    return [float(patch * i) for i in range(num_patches)]


@dataclass(frozen=True)
class Qwen3VLChatMarkers:
    """Wire-format ChatML markers (fixed literals — not tokenizer-dependent)."""

    im_start_token: str = "<|im_start|>"
    im_end_token: str = "<|im_end|>"
    assistant_prefix: str = "<|im_start|>assistant\n"
    vision_start_token: str = "<|vision_start|>"
    vision_end_token: str = "<|vision_end|>"


class Qwen3VLChatTemplate(TextEncoderChatTemplate):
    chat_markers: Qwen3VLChatMarkers
    # Which frames one temporal patch merges. Set by
    # `Qwen3VLTextEncoderPreprocessor.build_chat_template` from the module config;
    # the default matches the shipped checkpoints.
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2

    def __init__(self, tokenizer: Any):
        super().__init__(tokenizer)  # resolves bos / eos markers + ids
        self.chat_markers = Qwen3VLChatMarkers()
        # Extra ChatML stop token (eos_token_id comes from the base).
        self.im_end_token_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    def apply_chat_template(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Apply Qwen3-VL ChatML to a raw conversation (text + image + video)."""
        markers = self.chat_markers
        out: list[ConversationItem] = []
        dummy_parts: list[ConversationItem] = []
        prev_role: str | None = None

        def close_turn(role: str) -> None:
            out.append(
                self._build_conversation_item(
                    "text", markers.im_end_token + "\n", role, loss_mask=int(role == "assistant")
                )
            )

        for item in sample:
            role = item.role
            if item.is_dummy:
                dummy_parts.append(item)
                continue

            if role != prev_role:
                if prev_role is not None:
                    close_turn(prev_role)
                out.append(
                    self._build_conversation_item("text", markers.im_start_token + role + "\n", role, loss_mask=0)
                )
                prev_role = role

            if item.type == "text":
                out.append(self._build_conversation_item("text", str(item.value), role))
            elif item.type == "video":
                # HF emits one ``<t seconds><|vision_start|>…<|vision_end|>`` block per
                # temporal patch, not one per clip, so the timestamps line up with the
                # per-frame M-RoPE the backbone rebuilds. A video item already carries
                # the merged vision rows for *every* frame, so split its row count into
                # per-frame segments here.
                out.extend(self._video_segments(item, role))
            elif item.type == "image":
                # One block for the whole image (``T == 1``), as upstream.
                out.append(self._build_conversation_item("text", markers.vision_start_token, role, loss_mask=0))
                out.append(item)  # media row passed through verbatim (keeps value/source/meta)
                out.append(self._build_conversation_item("text", markers.vision_end_token, role, loss_mask=0))
            else:
                raise ValueError(f"Qwen3-VL text encoder only supports text/image/video items, got {item.type!r}")

        if prev_role is not None:
            close_turn(prev_role)
        out.extend(dummy_parts)
        return out

    def _video_segments(self, item: ConversationItem, role: str) -> list[ConversationItem]:
        """One ``<t seconds><|vision_start|>rows<|vision_end|>`` block per frame.

        HF's ``Qwen3VLProcessor.replace_video_token`` builds the prompt that way, and
        the backbone's M-RoPE restarts per frame to match (see
        ``qwen3vl_vision_position_ids``). Emitting a single block for the whole clip
        would hand the model a different token sequence — and a different position
        layout — than the checkpoint was trained on.

        A video item holds **raw patch rows** for every frame in one tensor (the
        vision preprocessor splits the flat ViT output by ``t*h*w``), so one frame
        contributes ``h*w`` rows of that tensor; the merge happens when the vision
        module encodes them, not here.

        Each segment is a *complete* visual item in its own right, so its metadata
        must describe that segment alone: the grid becomes ``[1, h, w]`` and the
        merged-token count is divided by the clip length. Copying the clip's
        ``[t, h, w]`` would have every downstream consumer build ``t`` frames' worth
        of positions and M-RoPE for a single frame's rows.
        """
        markers = self.chat_markers
        grid = (item.meta or {}).get(_OMNI_GRID) or (item.meta or {}).get("grid_thw")
        value = item.value
        num_frames = None
        h = w = 0
        if isinstance(value, torch.Tensor) and value.dim() == 2 and grid is not None:
            t, h, w = (int(x) for x in torch.as_tensor(grid).reshape(-1).tolist())
            patch_rows_per_frame = h * w
            if patch_rows_per_frame > 0 and value.size(0) == t * patch_rows_per_frame:
                num_frames = t
        if num_frames is None:
            # No usable grid: keep the single-block shape rather than guessing a
            # split that would silently misalign rows.
            return [
                self._build_conversation_item("text", markers.vision_start_token, role, loss_mask=0),
                item,
                self._build_conversation_item("text", markers.vision_end_token, role, loss_mask=0),
            ]

        timestamps = _frame_timestamps(item, num_frames, self.temporal_patch_size)
        per_frame = value.size(0) // num_frames
        meta = dict(item.meta or {})
        # A single frame's grid, and the tokens it alone contributes.
        meta[_OMNI_GRID] = [1, h, w]
        if _OMNI_TOKEN_LEN in meta:
            meta[_OMNI_TOKEN_LEN] = int(meta[_OMNI_TOKEN_LEN]) // num_frames
        out: list[ConversationItem] = []
        for frame in range(num_frames):
            out.append(self._build_conversation_item("text", f"<{timestamps[frame]:.1f} seconds>", role, loss_mask=0))
            out.append(self._build_conversation_item("text", markers.vision_start_token, role, loss_mask=0))
            out.append(
                ConversationItem(
                    type=item.type,
                    value=value[frame * per_frame : (frame + 1) * per_frame],
                    role=item.role,
                    meta=dict(meta),
                )
            )
            out.append(self._build_conversation_item("text", markers.vision_end_token, role, loss_mask=0))
        return out

    def apply_generation_prompt(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Append the assistant generation prefix after a templated (turn-closed) prompt."""
        out = list(sample)
        out.append(self._build_conversation_item("text", self.chat_markers.assistant_prefix, "assistant", loss_mask=0))
        return out
