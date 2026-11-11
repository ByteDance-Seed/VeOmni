# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Per-module image processor and worker-side CPU preprocessor for :class:`LladaUniImageTokenizer`.

LLaDA2.0-Uni uses the GlmImage dynamic-resolution processor (pixel budget
``min_pixels``/``max_pixels``, ``patch_size=16``, ``merge_size=1``), snapping
any odd grid side in ``[1, 35]`` (which lacks a ``<|reserved_token_{n}|>`` in the
LLaDA2.0-Uni vocabulary) to the next even grid side, matching
``veomni.data.data_transform._llada2_uni_patchify``.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers.models.glm_image.image_processing_glm_image import GlmImageImageProcessor, smart_resize

from ....utils.conversation import ConversationItem, iter_desired_items
from ...module_processing_base import ModulePreprocessorBase


_OMNI_PIXELS = "llada2_uni_image_tokenizer_pixels"
_OMNI_GRID = "_omni_grid"
_OMNI_VQ_IDS = "llada2_uni_vq_ids"
_OMNI_VQ_GRID = "llada2_uni_grid"


def _is_spellable_grid_side(side: int) -> bool:
    """Whether the LLaDA2.0-Uni vocabulary has a ``<|reserved_token_{side}|>`` token.

    In LLaDA2.0-Uni's tokenizer, odd numbers in ``[1, 35]`` are assigned tonamed
    special tokens (``<|endoftext|>``, ``<|mask|>``, ``<|image|>``, etc.), so
    ``<|reserved_token_{n}|>`` exists for even numbers in ``[2, 34]`` and all
    integers ``>= 36``.
    """
    if side < 2:
        return False
    if side < 36 and side % 2 == 1:
        return False
    return True


def _next_spellable_side(side: int) -> int:
    for candidate in range(max(side, 2), side + 8):
        if _is_spellable_grid_side(candidate):
            return candidate
    raise ValueError(f"No reserved token spells an image grid side near {side}.")


def _to_rgb_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] in (1, 3, 4):
            return to_pil_image(image).convert("RGB")
        if image.dim() == 3 and image.shape[-1] in (1, 3, 4):
            return to_pil_image(image.permute(2, 0, 1)).convert("RGB")
    raise TypeError(f"Expected a PIL.Image or 3D image tensor, got {type(image).__name__}.")


def patchify_llada2_uni_images(
    images: List[Any],
    image_processor: GlmImageImageProcessor,
    dtype: Optional[torch.dtype] = None,
) -> tuple[List[torch.Tensor], List[List[int]]]:
    """Patchify images into per-image ``(t*h*w, patch_dim)`` tensors and ``[t, h, w]`` grids."""
    patch_size = int(image_processor.patch_size)
    merge_size = int(getattr(image_processor, "merge_size", 1))
    size = image_processor.size
    shortest_edge = size.get("shortest_edge", 512 * 512) if isinstance(size, dict) else size.shortest_edge
    longest_edge = size.get("longest_edge", 2048 * 2048) if isinstance(size, dict) else size.longest_edge

    patches_list: List[torch.Tensor] = []
    grids_list: List[List[int]] = []
    for raw in images:
        pil = _to_rgb_pil(raw)
        height, width = smart_resize(
            pil.height,
            pil.width,
            factor=patch_size * merge_size,
            min_pixels=shortest_edge,
            max_pixels=longest_edge,
        )
        grid = (height // patch_size, width // patch_size)
        spellable = (_next_spellable_side(grid[0]), _next_spellable_side(grid[1]))
        if spellable == grid:
            out = image_processor(images=[pil], return_tensors="pt", do_resize=True)
        else:
            pil = pil.resize((spellable[1] * patch_size, spellable[0] * patch_size), Image.BICUBIC)
            out = image_processor(images=[pil], return_tensors="pt", do_resize=False)
        px = out["pixel_values"]
        if dtype is not None:
            px = px.to(dtype=dtype)
        patches_list.append(px)
        grids_list.append([int(v) for v in out["image_grid_thw"][0].tolist()])
    return patches_list, grids_list


class LladaUniImageProcessor(GlmImageImageProcessor):
    """GlmImage dynamic-resolution image processor (encode side)."""


class LladaUniImageTokenizerPreprocessor(ModulePreprocessorBase):
    """Worker-side GlmImage normalize + patchify for the LLaDA2.0-Uni VQ tokenizer."""

    def __init__(
        self,
        image_processor: LladaUniImageProcessor,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._image_processor = image_processor
        self._dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "LladaUniImageTokenizerPreprocessor":
        del config_overrides
        image_processor = LladaUniImageProcessor.from_pretrained(module_path, **kwargs)
        return cls(image_processor)

    def bind_dummy_inputs(self, config: Any, dtype: Any = None) -> None:
        del config
        self._dtype = dtype

    def preprocess_conversations(
        self,
        conversation_list: list[list[ConversationItem]],
        inference: bool = False,
        **kwargs: Any,
    ) -> None:
        del inference, kwargs
        items = [
            it
            for it in iter_desired_items(conversation_list, types=["image"])
            if not it.is_dummy
            and not it.meta.get(_OMNI_PIXELS)
            and it.meta.get(_OMNI_VQ_IDS) is None
            and it.value is not None
        ]
        if not items:
            return
        patches_list, grids_list = patchify_llada2_uni_images(
            [it.value for it in items],
            self._image_processor,
            dtype=self._dtype,
        )
        for it, px, grid in zip(items, patches_list, grids_list, strict=True):
            it.value = px
            it.meta[_OMNI_GRID] = grid
            it.meta[_OMNI_PIXELS] = True

    def __call__(self, batch: dict[str, Any], inference: bool = False, **kwargs: Any) -> None:
        self.preprocess_conversations(batch["conversation_list"], inference=inference, **kwargs)


__all__ = [
    "_OMNI_GRID",
    "_OMNI_PIXELS",
    "_OMNI_VQ_GRID",
    "_OMNI_VQ_IDS",
    "LladaUniImageProcessor",
    "LladaUniImageTokenizerPreprocessor",
    "patchify_llada2_uni_images",
]
