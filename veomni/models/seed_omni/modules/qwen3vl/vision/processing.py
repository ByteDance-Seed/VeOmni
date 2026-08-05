"""Per-module processors + worker-side preprocessor for the Qwen3-VL vision tower.

Qwen3-VL keeps image and video preprocessing in **two distinct processors with
two distinct configs** (mirroring the upstream Qwen3-VL checkpoint). The vision
module loads **both**:

* **image** → :class:`transformers.Qwen2VLImageProcessor`, config
  ``preprocessor_config.json``.  Qwen3-VL *reuses* the Qwen2-VL image processor
  (the upstream ``preprocessor_config.json`` declares
  ``image_processor_type: Qwen2VLImageProcessorFast``, which resolves to
  ``Qwen2VLImageProcessor`` in transformers v5) — there is **no** Qwen3-VL
  specific image processor.
* **video** → :class:`transformers.Qwen3VLVideoProcessor`, config
  ``video_preprocessor_config.json`` (temporal patchify; not a Qwen2-VL class).

Aliased (not subclassed) to the canonical classes so the saved configs keep
their registered ``*_processor_type`` and stay Auto-loadable.

:class:`Qwen3VLVisionPreprocessor` is the picklable, weight-free CPU worker
counterpart (see :class:`~veomni.models.seed_omni.mixins.modulemixin.Preprocessor`)
— built straight off the checkpoint dir, with no model instance involved.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from ....mixins.modulemixin import Preprocessor
from ....utils.conversation import ConversationItem, iter_desired_items
from .configuration import Qwen3VLVisionEncoderConfig


# Image: Qwen3-VL reuses the Qwen2-VL image processor.
Qwen3VLVisionImageProcessor = Qwen2VLImageProcessor
# Video: Qwen3-VL's own (non-Qwen2VL) video processor.
Qwen3VLVisionVideoProcessor = Qwen3VLVideoProcessor

# qwen3vl-specific meta key: per-item ``grid_thw`` stashed alongside the
# normalized patches (which go onto ``item.value``, like siglip/vqvae) by the
# CPU preprocessor (DataLoader worker for training, pre-FSM pass for inference).
# ``_pixels_and_grid`` pops it on the main process.
_OMNI_GRID = "_omni_grid"
_SOURCE = "qwen3vl_vision"


def _video_metadata(items: list, frames: list) -> list[dict]:
    """HF ``video_metadata`` for the handed-over (already decoded) frames.

    The data layer (``seed_omni/video_utils.load_video``) pre-trims each clip to
    ``mm_configs.fps`` purely as a memory bound, so the frames passed here are a
    self-contained clip whose *source* fps **is** ``VideoInputs.video_fps``. We
    forward that as the metadata fps; the HF ``Qwen3VLVideoProcessor`` then
    sub-samples to its own authoritative ``self.fps`` (the model's target rate).
    Without metadata it would default to ``fps=24`` and mangle the clip.
    """
    return [{"total_num_frames": f.shape[0], "fps": it.value.video_fps} for it, f in zip(items, frames)]


def _store_patches(items: list, pixel_values: torch.Tensor, grid_thw: torch.Tensor, dtype: Any) -> None:
    """Split flat ViT patches by per-item grid; stash patches on ``value`` + grid on ``meta``.

    Used by the CPU preprocessor (both training and inference share it) so items
    are left in the preprocessed form that ``_pixels_and_grid`` reads back.
    """
    grids = grid_thw.tolist()
    sizes = [g[0] * g[1] * g[2] for g in grids]
    chunks = torch.split(pixel_values, sizes, dim=0)
    for it, px, g in zip(items, chunks, grids, strict=True):
        it.value = px.to(dtype=dtype)
        it.meta[_OMNI_GRID] = g


class Qwen3VLVisionPreprocessor(Preprocessor):
    """Worker-side image/video patchify+normalize for the Qwen3-VL vision tower.

    Holds only the (picklable) HF image / video processors + a CPU zero-patch
    template — never the model. Runs them on **CPU** (bf16, to halve IPC), writes
    the per-item normalized patches onto ``item.value`` and stashes ``grid_thw`` on
    ``meta``. For each sample without a user image/video, appends a
    ``role="dummy"`` placeholder
    carrying the zero patches + grid (the merger still runs on it in the GPU
    forward for the FSDP gradient anchor).
    """

    def __init__(
        self,
        image_processor: Any,
        video_processor: Any,
        dtype: torch.dtype | None = None,
        dummy_pixel_values: torch.Tensor | None = None,
        dummy_grid: list | None = None,
    ) -> None:
        self._image_processor = image_processor
        self._video_processor = video_processor
        self._dtype = dtype
        self._dummy_pixel_values = dummy_pixel_values  # CPU (t*h*w, pixel_row), model dtype
        self._dummy_grid = dummy_grid  # [t, h, w]

    @classmethod
    def from_pretrained(cls, module_path: str, **kwargs: Any) -> Qwen3VLVisionPreprocessor | None:
        """Build straight from the checkpoint dir — no model instance needed.

        Image and video processors ship as two independent config files
        (``preprocessor_config.json`` / ``video_preprocessor_config.json``); try
        each on its own so one missing file doesn't drop the other modality.
        """
        del kwargs
        image_processor = _try_from_pretrained(Qwen3VLVisionImageProcessor, module_path)
        video_processor = _try_from_pretrained(Qwen3VLVisionVideoProcessor, module_path)
        if image_processor is None and video_processor is None:
            return None
        return cls(image_processor, video_processor)

    def bind_dummy_inputs(self, config: Qwen3VLVisionEncoderConfig, dtype: torch.dtype | None = None) -> None:
        """Training-only FSDP-anchor dummy — pure ``(config, dtype)``, no live model."""
        cfg = config.vision_config
        merge = cfg.spatial_merge_size
        t, h, w = 1, 2 * merge, 2 * merge
        pixel_row = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        self._dtype = dtype
        self._dummy_pixel_values = torch.zeros(t * h * w, pixel_row, dtype=dtype)
        self._dummy_grid = [t, h, w]

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        for sample in conversation_list:
            sample_image_items = list(iter_desired_items([sample], types=["image"], roles=["user"]))
            sample_video_items = list(iter_desired_items([sample], types=["video"], roles=["user"]))
            if sample_image_items or sample_video_items:
                if sample_image_items and self._image_processor is not None:
                    out = self._image_processor(images=[it.value for it in sample_image_items], return_tensors="pt")
                    self._store(sample_image_items, out["pixel_values"], out["image_grid_thw"])
                if sample_video_items and self._video_processor is not None:
                    frames = [it.value.video for it in sample_video_items]
                    out = self._video_processor(
                        videos=frames, video_metadata=_video_metadata(sample_video_items, frames), return_tensors="pt"
                    )
                    self._store(sample_video_items, out["pixel_values_videos"], out["video_grid_thw"])
            elif not inference:
                if self._dummy_pixel_values is None:
                    raise RuntimeError(
                        f"{type(self).__name__}: dummy inputs not bound — call bind_dummy_inputs() "
                        "before training use (pure inference never reaches this branch)."
                    )
                sample.append(
                    ConversationItem(
                        type="image",
                        value=self._dummy_pixel_values,
                        role="dummy",
                        source=_SOURCE,
                        meta={_OMNI_GRID: self._dummy_grid},
                    )
                )

    def _store(self, items: list, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> None:
        _store_patches(items, pixel_values, grid_thw, self._dtype)


def _try_from_pretrained(processor_cls: Any, module_path: str) -> Any | None:
    try:
        return processor_cls.from_pretrained(module_path)
    except Exception:  # noqa: BLE001 — this modality's config file may not exist for this module
        return None


__all__ = [
    "Qwen3VLVisionImageProcessor",
    "Qwen3VLVisionVideoProcessor",
    "Qwen3VLVisionPreprocessor",
]
