"""Per-module processors + worker-side preprocessor for the Qwen3-Omni vision tower.

Qwen3-Omni ships a **single** ``preprocessor_config.json`` that serves three
processors at once (``feature_extractor_type: WhisperFeatureExtractor`` for the
audio tower, ``image_processor_type: Qwen2VLImageProcessor`` here) and no
``video_preprocessor_config.json`` at all. The video processor resolves to
:class:`Qwen2VLVideoProcessor` through the auto-map and falls back to reading
that same file, which matters: its class default ``patch_size`` is 14 while
Qwen3-Omni uses 16, so both processors must be loaded from the checkpoint
directory and never constructed bare.

Aliased (not subclassed) to the canonical classes so the saved configs keep
their registered ``*_processor_type`` and stay Auto-loadable.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

from ......data.seed_omni.utils.media_metadata import VIDEO_METADATA_KEY
from ....utils.conversation import _OMNI_TOKEN_LEN, ConversationItem, iter_desired_items
from ...module_processing_base import ModulePreprocessorBase
from .configuration import Qwen3OmniVisionEncoderConfig


# Image: Qwen3-Omni reuses the Qwen2-VL image processor.
Qwen3OmniVisionImageProcessor = Qwen2VLImageProcessor
# Video: the auto-map resolves qwen3_omni_moe to the Qwen2-VL video processor.
# Note this is *not* Qwen3VLVideoProcessor — see ``_second_per_grid`` for the
# consequence that matters.
Qwen3OmniVisionVideoProcessor = Qwen2VLVideoProcessor

# Per-item ``grid_thw`` stashed alongside the normalized patches (which go onto
# ``item.value``) by the CPU preprocessor; popped on the main process, which
# then republishes it as the public ``meta["grid_thw"]``.
_OMNI_GRID = "_omni_grid"
# Public meta key, read by the backbone's TMRoPE packer and by the talker's
# media metadata. Not popped.
_SECOND_PER_GRID = "second_per_grid"


def _second_per_grid(item: ConversationItem, temporal_patch_size: int) -> float:
    """Seconds spanned by one temporal grid step, off the clip's own metadata.

    Load-bearing rather than cosmetic. The backbone scales TMRoPE's temporal
    row by ``second_per_grid * position_id_per_seconds`` (13 for Qwen3-Omni), so
    getting it wrong silently misplaces every position id in and after a video
    clip — and, with in-video audio, slides the two streams apart on a clock
    they are supposed to share. The backbone's packer therefore *raises* when a
    video item lacks this key instead of defaulting, and this is the only thing
    that supplies it.

    Upstream computes ``temporal_patch_size / fps`` against the rate that was
    *asked* for. That is the PR #1165 mistake: a clip whose frames were thinned
    to a frame budget, or whose source rate does not divide the requested one,
    is handed frames spaced differently from that rate, and every grid step then
    claims a duration the frames never had.
    :meth:`VideoMetadata.seconds_per_patch` instead measures the spacing of the
    frames actually kept, on the source clip's own axis — which is the axis the
    audio is on too.

    Deriving it from the frames that were kept is correct here because
    Qwen3-Omni uses :class:`Qwen2VLVideoProcessor`, which has
    ``do_sample_frames=False`` and patchifies exactly the frames it is handed,
    so the grid steps line up with those frames one for one.
    """
    metadata = (item.meta or {}).get(VIDEO_METADATA_KEY)
    if metadata is None:
        raise ValueError(
            f"qwen3omni_vision: a video item carries no meta[{VIDEO_METADATA_KEY!r}], so its TMRoPE timeline "
            f"cannot be derived and the clip cannot be placed on the same clock as its audio. Video items must "
            f"come from veomni/data/seed_omni/utils/video.load_video (or state the same metadata themselves)."
        )
    return metadata.seconds_per_patch(temporal_patch_size)


def _store_patches(
    items: list,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    dtype: Any,
    *,
    spatial_merge_size: int,
) -> None:
    """Split flat ViT patches by per-item grid; stash patches on ``value`` + grid on ``meta``.

    Each item also records the merged-token count it contributes, which is what
    ``dyn_bsz`` counts (``_omni_token_len``).
    """
    grids = grid_thw.tolist()
    sizes = [g[0] * g[1] * g[2] for g in grids]
    chunks = torch.split(pixel_values, sizes, dim=0)
    merge_area = spatial_merge_size**2
    for item, patches, grid in zip(items, chunks, grids, strict=True):
        item.value = patches.to(dtype=dtype)
        item.meta[_OMNI_GRID] = grid
        item.meta[_OMNI_TOKEN_LEN] = int(grid[0] * grid[1] * grid[2]) // merge_area


class Qwen3OmniVisionPreprocessor(ModulePreprocessorBase):
    """Worker-side image/video patchify+normalize for the Qwen3-Omni vision tower.

    Holds only the (picklable) HF image / video processors plus a CPU zero-patch
    template — never the model. Writes each item's normalized patches onto
    ``item.value``, stashes ``grid_thw`` on ``meta``, and for video items also
    records ``second_per_grid`` (see :func:`_second_per_grid`).

    When a whole micro-batch has no user image/video, appends one ``is_dummy``
    placeholder carrying zero patches + grid, so the FSDP unit still runs exactly
    one visual forward per step and produces a gradient anchor.
    """

    def __init__(
        self,
        image_processor: Any,
        video_processor: Any,
        dtype: torch.dtype | None = None,
        dummy_pixel_values: torch.Tensor | None = None,
        dummy_grid: list | None = None,
        spatial_merge_size: int = 2,
    ) -> None:
        self._image_processor = image_processor
        self._video_processor = video_processor
        self._dtype = dtype
        self._dummy_pixel_values = dummy_pixel_values
        self._dummy_grid = dummy_grid
        # Merged-token count per item (``_omni_token_len``) needs the same merge
        # factor the ViT applies; the preprocessor gets it from the processor
        # config so it can tag items before any model exists.
        self._spatial_merge_size = int(spatial_merge_size)

    @classmethod
    def from_pretrained(cls, module_path: str, **kwargs: Any) -> Qwen3OmniVisionPreprocessor:
        """Build straight from the checkpoint dir — no model instance needed.

        Both processors read ``preprocessor_config.json`` (Qwen3-Omni ships no
        separate video config), so the checkpoint split must place that file in
        this module's directory.
        """
        overrides = kwargs.pop("config_overrides", None) or {}
        del kwargs
        image_processor = Qwen3OmniVisionImageProcessor.from_pretrained(module_path)
        video_processor = Qwen3OmniVisionVideoProcessor.from_pretrained(module_path)
        # ``spatial_merge_size`` lives on the image processor; fall back to the
        # Qwen2-VL default of 2 when an older processor config omits it.
        merge = int(overrides.get("spatial_merge_size") or getattr(image_processor, "merge_size", 2) or 2)
        return cls(image_processor, video_processor, spatial_merge_size=merge)

    def bind_dummy_inputs(self, config: Qwen3OmniVisionEncoderConfig, dtype: torch.dtype | None = None) -> None:
        """Training-only FSDP-anchor dummy — pure ``(config, dtype)``, no live model."""
        cfg = config.vision_config
        merge = cfg.spatial_merge_size
        t, h, w = 1, 2 * merge, 2 * merge
        pixel_row = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        self._dtype = dtype
        self._spatial_merge_size = int(merge)
        self._dummy_pixel_values = torch.zeros(t * h * w, pixel_row, dtype=dtype)
        self._dummy_grid = [t, h, w]

    def preprocess_conversations(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        for sample in conversation_list:
            image_items = [
                it for it in iter_desired_items([sample], types=["image"], roles=["user"]) if not it.is_dummy
            ]
            video_items = [
                it for it in iter_desired_items([sample], types=["video"], roles=["user"]) if not it.is_dummy
            ]
            if image_items or video_items:
                if image_items:
                    out = self._image_processor(images=[it.value for it in image_items], return_tensors="pt")
                    self._store(image_items, out["pixel_values"], out["image_grid_thw"])
                if video_items:
                    frames = [it.value.video for it in video_items]
                    out = self._video_processor(videos=frames, return_tensors="pt")
                    self._store(video_items, out["pixel_values_videos"], out["video_grid_thw"])
                    temporal_patch_size = self._video_processor.temporal_patch_size
                    for it in video_items:
                        it.meta[_SECOND_PER_GRID] = _second_per_grid(it, temporal_patch_size)
                continue
            if inference:
                continue
            if self._dummy_pixel_values is None:
                raise RuntimeError(
                    f"{type(self).__name__}: dummy inputs not bound — call bind_dummy_inputs() "
                    "before training use (pure inference never reaches this branch)."
                )
            # Per sample: this module's forward still runs on a step with no real
            # media so its FSDP collectives stay in step with the other ranks. It
            # keeps the ``role="user"`` tag of the items it stands in for, so the
            # forward's own filter selects it.
            t, h, w = self._dummy_grid
            sample.append(
                ConversationItem(
                    type="image",
                    value=self._dummy_pixel_values.clone(),
                    role="user",
                    is_dummy=True,
                    meta={
                        _OMNI_GRID: self._dummy_grid,
                        _OMNI_TOKEN_LEN: int(t * h * w) // (self._spatial_merge_size**2),
                    },
                )
            )

    def _store(self, items: list, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> None:
        _store_patches(
            items,
            pixel_values,
            grid_thw,
            self._dtype,
            spatial_merge_size=self._spatial_merge_size,
        )


__all__ = [
    "Qwen3OmniVisionImageProcessor",
    "Qwen3OmniVisionVideoProcessor",
    "Qwen3OmniVisionPreprocessor",
]
