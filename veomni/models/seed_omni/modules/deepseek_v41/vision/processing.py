# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image processor + CPU worker for the DeepSeek-V4.1 vision tower.

Ported from the released ``inference/image_processor.py``. DeepSeek-V4.1 ships no
HuggingFace image processor, so the sidecar is written out here as a
:class:`BaseImageProcessor` subclass -- that is what gives the split checkpoint a
``preprocessor_config.json`` and keeps the resize constants in one place rather
than duplicated between training and inference.

The resize plan is the part worth stating: an image is padded up to whole patches,
then shrunk until its *token* cost fits ``max_image_tokens``. Token cost is not the
patch count -- the aligner downsamples by ``downsample_ratio`` in both axes and the
span carries one newline per row plus two delimiters, so it is
``n_llm_h * (n_llm_w + 1) + 2``. Getting that wrong does not fail loudly; it
silently produces a span of a different length than the one the text side reserved.
"""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers.image_processing_utils import BaseImageProcessor

from ....utils.conversation import ConversationItem, iter_desired_items
from ...module_processing_base import ModulePreprocessorBase
from .configuration import DeepseekV41VisionEncoderConfig


TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)

#: Per-item meta written by the CPU worker and popped on the main process.
_OMNI_VIT_GRID = "_omni_vit_grid"
_OMNI_TOKEN_TYPES = "_omni_token_types"


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """LLM positions one image occupies: the grid, a newline per row, two delimiters."""
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int) -> tuple[int, int]:
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return (
        math.ceil((best_height // patch_size) / downsample_ratio),
        math.ceil((best_width // patch_size) / downsample_ratio),
    )


def solve_resize_ratio(
    height: int, width: int, patch_size: int, downsample_ratio: int, max_n_token: int
) -> tuple[int, int]:
    """Largest aspect-preserving pixel size whose token grid still fits ``max_n_token``."""
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    """Shrink the pixel size until the image costs at most ``max_n_token`` LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
            raise ValueError(
                f"image resize failed to fit max_image_tokens={max_n_token}: "
                f"{best_height}x{best_width} still costs {num_image_tokens(n_llm_h, n_llm_w)} tokens."
            )
    return n_llm_h, n_llm_w, best_height, best_width


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Span layout: ``START`` + (``IMAGE`` * n_llm_w + ``NEW_LINE``) * n_llm_h + ``END``."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


class DeepseekV41VisionImageProcessor(BaseImageProcessor):
    """Resize -> pad -> normalize -> patchify, exactly as the released processor does.

    ``ImageOps.pad`` with grey (127, 127, 127) rather than a plain resize is the
    default path: it preserves aspect ratio and matches training. The one exception
    is an image wider than ``max_wh_ratio``, which is squashed instead -- padding it
    would spend most of the token budget on grey.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        patch_size: int = 14,
        downsample_ratio: int = 3,
        max_image_tokens: int = 1024,
        min_pixels: int = 295936,
        max_wh_ratio: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.max_image_tokens = max_image_tokens
        self.min_pixels = min_pixels
        self.max_wh_ratio = max_wh_ratio

    def plan_grid(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Resize plan for an image of this original size; a pure function of its arguments."""
        patch = self.patch_size
        if self.max_wh_ratio is not None and width > height * self.max_wh_ratio:
            width = int(height * self.max_wh_ratio)
        if 0 < width * height < self.min_pixels:
            scale = (self.min_pixels / (width * height)) ** 0.5
            width = int(width * scale)
            height = int(height * scale)
        best_width = math.ceil(width / patch) * patch
        best_height = math.ceil(height / patch) * patch
        return safe_resize(height, width, best_height, best_width, patch, self.downsample_ratio, self.max_image_tokens)

    def preprocess_one(self, image: Any) -> tuple[torch.Tensor, int, int, int, int]:
        """One image -> ``(patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w)``.

        ``patches`` is ``(n_vit_h * n_vit_w, 3, patch, patch)`` in [-1, 1].

        ``image`` is a PIL image or raw bytes at inference, and the ``(3, H, W)
        uint8`` tensor the SeedOmni V2 data layer decodes to in training. The
        tensor is turned back into PIL so both paths resize identically.
        """
        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(bytes(image)))
        elif isinstance(image, torch.Tensor):
            if image.dtype != torch.uint8 or image.dim() != 3 or image.size(0) != 3:
                raise TypeError(f"expected a (3, H, W) uint8 image tensor, got {image.dtype} {tuple(image.shape)}")
            image = Image.fromarray(np.ascontiguousarray(image.permute(1, 2, 0).cpu().numpy()))
        if not isinstance(image, Image.Image):
            raise TypeError(f"expected a PIL image, raw bytes or a uint8 tensor, got {type(image).__name__}")
        image = image.convert("RGB")

        patch = self.patch_size
        n_llm_h, n_llm_w, best_height, best_width = self.plan_grid(image.width, image.height)
        n_vit_h, n_vit_w = best_height // patch, best_width // patch
        if self.max_wh_ratio is not None and image.width >= self.max_wh_ratio * image.height:
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))

        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
        pixels = (pixels - 0.5) / 0.5
        patches = pixels.reshape(3, n_vit_h, patch, n_vit_w, patch)
        patches = patches.permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, patch, patch)
        return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


class DeepseekV41VisionPreprocessor(ModulePreprocessorBase):
    """Worker-side patchify for the DeepSeek-V4.1 ViT.

    Writes the patches onto ``item.value`` and stashes the ViT grid plus the span's
    token-type vector on ``item.meta``. The token types are computed here rather
    than in the GPU forward because they fix the *sequence length* this item
    contributes, and the text encoder's chat template has to agree with it on the
    CPU side before anything reaches a device.
    """

    def __init__(
        self,
        image_processor: DeepseekV41VisionImageProcessor,
        dtype: torch.dtype | None = None,
        dummy_patches: torch.Tensor | None = None,
        dummy_grid: list[int] | None = None,
        dummy_types: torch.Tensor | None = None,
    ) -> None:
        self._image_processor = image_processor
        self._dtype = dtype
        self._dummy_patches = dummy_patches
        self._dummy_grid = dummy_grid
        self._dummy_types = dummy_types

    @classmethod
    def from_pretrained(cls, module_path: str, **kwargs: Any) -> DeepseekV41VisionPreprocessor:
        del kwargs
        return cls(DeepseekV41VisionImageProcessor.from_pretrained(module_path))

    def bind_dummy_inputs(self, config: DeepseekV41VisionEncoderConfig, dtype: torch.dtype | None = None) -> None:
        """Training-only FSDP anchor: one ``downsample_ratio``-square patch grid.

        Sized to exactly one aligner cell so the anchor costs the ViT one
        ``r x r`` grid and the aligner one row -- the smallest input that still
        exercises every parameter, including the unfold in the aligner.
        """
        ratio = config.downsample_ratio
        patch = config.patch_size
        self._dtype = dtype
        self._dummy_patches = torch.zeros(ratio * ratio, 3, patch, patch, dtype=dtype)
        self._dummy_grid = [ratio, ratio]
        self._dummy_types = image_token_types(1, 1)

    def preprocess_conversations(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs
        saw_real_image = False
        for sample in conversation_list:
            items = [it for it in iter_desired_items([sample], types=["image"], roles=["user"]) if not it.is_dummy]
            for item in items:
                patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self._image_processor.preprocess_one(item.value)
                item.value = patches.to(dtype=self._dtype)
                item.meta[_OMNI_VIT_GRID] = [n_vit_h, n_vit_w]
                item.meta[_OMNI_TOKEN_TYPES] = image_token_types(n_llm_h, n_llm_w)
                saw_real_image = True

        if inference or saw_real_image:
            return
        if self._dummy_patches is None:
            raise RuntimeError(
                f"{type(self).__name__}: dummy inputs not bound — call bind_dummy_inputs() "
                "before training use (pure inference never reaches this branch)."
            )
        self.append_batch_anchor(
            conversation_list,
            ConversationItem(
                type="image",
                value=self._dummy_patches,
                role="user",
                is_dummy=True,
                meta={
                    _OMNI_VIT_GRID: list(self._dummy_grid),
                    _OMNI_TOKEN_TYPES: self._dummy_types.clone(),
                },
            ),
        )


__all__ = [
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_START",
    "TEXT",
    "DeepseekV41VisionImageProcessor",
    "DeepseekV41VisionPreprocessor",
    "image_token_types",
    "num_image_tokens",
]
