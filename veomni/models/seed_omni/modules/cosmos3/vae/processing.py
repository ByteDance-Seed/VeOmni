"""Worker-side preprocessor for :class:`Cosmos3Vae`: decoded frames -> the VAE's input clip.

The data layer hands over every frame of the source clip (``mm_configs.fps:
null``, no ``max_frames``) with its timeline. What is Cosmos3's own business
happens here, on the CPU, before ``dyn_bsz`` prices the sample — the same frames,
pixels and price the DiT port computed:

* keep ``duration * sample_fps`` frames, aligned to ``frame_factor * k +
  frame_factor_remainder`` within ``[min_frames, max_frames]``, as a fixed-stride
  prefix of the source clip, padded with the last frame when it is too short;
* map ``uint8`` to ``[-1, 1]`` and resize to a multiple of ``spatial_scale *
  latent_patch_size``, so the latent tiles into patches;
* price the clip as the reference's token counter does.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

from veomni.data.seed_omni.utils.media_metadata import VIDEO_METADATA_KEY

from ....utils.conversation import _OMNI_TOKEN_LEN, ConversationItem, iter_desired_items
from ...module_processing_base import ModulePreprocessorBase
from ..item_meta import FPS_KEY, LATENT_KEY
from .configuration import Cosmos3VaeConfig


# What a sample costs beyond its prompt and its latent patches, as the reference's
# token counter prices it: the prompt's eos (``und``) and the two vision markers
# (``gen``). The text encoder prices the prompt without its appended
# ``[eos, <|vision_start|>]``, so they are counted here and only here.
EXTRA_TOKENS_PER_SAMPLE = 3


def prefix_frame_indices(
    total_frames: int,
    video_fps: float,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
    factor: int,
    remainder: int,
) -> tuple[list[int], int]:
    """Source indices to keep and how many times to repeat the last one.

    The count is ``veomni.data.multimodal.video_utils.calculate_frame_indices``'s
    (not imported: that module needs PyAV); the indices are a fixed stride from
    frame 0, like cosmos-framework's SFT reader (``frame_selection_mode="first"``).
    A stride that runs off a short clip leaves fewer frames than counted, and the
    shortfall is padded rather than filled with duplicate indices.
    """
    count = max(total_frames / video_fps * fps, math.ceil((min_frames - remainder) / factor) * factor + remainder)
    count = min(count, int((max_frames - remainder) // factor) * factor + remainder)
    count = int((count - remainder) // factor) * factor + remainder
    count = int(max(1, count, remainder if remainder > 0 else factor))
    sample_count = min(count, total_frames)
    pad_count = count - sample_count
    stride = max(1, int(round(total_frames / max(sample_count, 1))))
    indices = list(range(0, total_frames, stride))[:sample_count]
    return indices, pad_count + sample_count - len(indices)


class Cosmos3VaePreprocessor(ModulePreprocessorBase):
    def __init__(self, config: Cosmos3VaeConfig, **kwargs: Any) -> None:
        del kwargs
        self.spatial_scale = config.spatial_scale
        self.temporal_scale = config.temporal_scale
        self.latent_patch_size = config.latent_patch_size
        self.sample_fps = config.sample_fps
        self.min_frames = config.min_frames
        self.max_frames = config.max_frames
        self.frame_factor = config.frame_factor
        self.frame_factor_remainder = config.frame_factor_remainder

    @classmethod
    def from_pretrained(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Cosmos3VaePreprocessor:
        return cls(Cosmos3VaeConfig.from_pretrained(module_path, model_config=config_overrides), **kwargs)

    def preprocess_conversations(
        self,
        conversation_list: list[list[ConversationItem]],
        inference: bool = False,
        **kwargs: Any,
    ) -> None:
        del inference, kwargs
        for item in iter_desired_items(conversation_list or [], types=["video"]):
            if item.meta.get(LATENT_KEY):
                # Replayed from the offline cache: already a latent, already priced.
                continue
            frames, fps = self.select_frames(item)
            item.value = self.to_vae_input(frames)
            item.meta[FPS_KEY] = fps
            item.meta[_OMNI_TOKEN_LEN] = self.token_count(tuple(frames.shape))

    def select_frames(self, item: ConversationItem) -> tuple[torch.Tensor, Optional[float]]:
        """The kept ``(T, C, H, W)`` frames and the source clip's rate, which the mRoPE spaces them by."""
        frames = item.value.video
        metadata = item.meta.get(VIDEO_METADATA_KEY)
        if metadata is None:
            raise ValueError("Cosmos3 picks frames on the source timeline, but the clip carries no video metadata.")
        indices, pad_count = prefix_frame_indices(
            max(1, int(metadata.total_num_frames)),
            float(metadata.fps),
            fps=self.sample_fps,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
            factor=self.frame_factor,
            remainder=self.frame_factor_remainder,
        )
        decoded = {int(source): position for position, source in enumerate(metadata.frames_indices)}
        missing = [index for index in indices if index not in decoded]
        if missing:
            raise ValueError(
                f"Cosmos3 keeps source frames {missing[:4]}... that the loader dropped. Set mm_configs.fps: null "
                "and leave mm_configs.max_frames unset, so every source frame reaches the VAE preprocessor."
            )
        kept = frames[[decoded[index] for index in indices]]
        if pad_count:
            kept = torch.cat([kept, kept[-1:].expand(pad_count, -1, -1, -1)], dim=0)
        return kept, float(metadata.fps)

    def to_vae_input(self, frames: torch.Tensor) -> torch.Tensor:
        """``(T, C, H, W)`` ``uint8`` -> ``(3, T, H', W')`` ``float32`` in ``[-1, 1]`` on the latent-patch grid."""
        if frames.dim() != 4 or frames.shape[1] != 3:
            raise ValueError(f"expected one decoded [T, 3, H, W] clip, got {tuple(frames.shape)}.")
        video = frames.to(torch.float32)
        if frames.dtype == torch.uint8:
            video = video.div(255.0)
        video = video.mul(2.0).sub(1.0)

        grid = self.spatial_scale * self.latent_patch_size
        height, width = video.shape[2], video.shape[3]
        target = (max(grid, round(height / grid) * grid), max(grid, round(width / grid) * grid))
        if target != (height, width):
            video = F.interpolate(video, size=target, mode="bilinear", align_corners=False, antialias=True)
        return video.permute(1, 0, 2, 3).contiguous()

    def token_count(self, frames_shape: tuple[int, ...]) -> int:
        """What the reference's token counter charges for the kept ``(T, C, H, W)`` frames, before the resize."""
        frames, _, height, width = frames_shape
        latent_frames = 1 + (frames - 1) // self.temporal_scale
        patches_h = math.ceil((height // self.spatial_scale) / self.latent_patch_size)
        patches_w = math.ceil((width // self.spatial_scale) / self.latent_patch_size)
        return latent_frames * patches_h * patches_w + EXTRA_TOKENS_PER_SAMPLE


__all__ = ["Cosmos3VaePreprocessor", "prefix_frame_indices"]
