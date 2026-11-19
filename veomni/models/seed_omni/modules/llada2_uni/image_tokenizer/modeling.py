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

"""LLaDA2.0-Uni discrete image tokenizer (GlmImage SigLIP-ViT + VQ codec).

Maps an image to **discrete VQ token ids** in the unified LLaDA2.0-Uni
vocabulary (codes ``[0, 16383]`` offset by ``image_token_offset`` = 157184 →
``[157184, 173567]``). The sibling ``text_encoder`` wraps each image's VQ ids
with ``<|image|><|reserved_token_h|><|reserved_token_w|><boi>[VQ…]<|/image|>``
and embeds them with the shared word-embedding table.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
from transformers.models.glm_image.modeling_glm_image import GlmImageVisionModel, GlmImageVQVAE

from ....utils.conversation import ConversationItem, iter_desired_items
from ...module_modeling_base import PretrainedOmniModule
from .configuration import LladaUniImageTokenizerConfig
from .processing import (
    _OMNI_GRID,
    _OMNI_PIXELS,
    _OMNI_VQ_GRID,
    _OMNI_VQ_IDS,
    LladaUniImageProcessor,
    LladaUniImageTokenizerPreprocessor,
    patchify_llada2_uni_images,
)


def extract_image_tokenizer_inputs(
    conversation_list: Optional[list[list[ConversationItem]]],
    *,
    image_processor: Optional[LladaUniImageProcessor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[ConversationItem]]:
    """Collect patchified pixels and ``image_grid_thw`` from conversation image items."""
    items = [
        it
        for it in iter_desired_items(conversation_list, types=["image"])
        if not it.is_dummy and it.value is not None and it.meta.get(_OMNI_VQ_IDS) is None
    ]
    if not items:
        return None, None, []

    if all(it.meta.get(_OMNI_PIXELS) for it in items):
        pixel_values = torch.cat([it.value for it in items], dim=0).to(device=device, dtype=dtype, non_blocking=True)
        image_grid_thw = torch.tensor([it.meta[_OMNI_GRID] for it in items], dtype=torch.long, device=device)
        return pixel_values, image_grid_thw, items

    if image_processor is None:
        raise RuntimeError(
            "LladaUniImageTokenizer: samples carry raw images but no image processor is loaded. "
            "Bind `module._image_processor` before forward/generate."
        )
    patches_list, grids_list = patchify_llada2_uni_images(
        [it.value for it in items],
        image_processor,
        dtype=dtype,
    )
    pixel_values = torch.cat(patches_list, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    image_grid_thw = torch.tensor(grids_list, dtype=torch.long, device=device)
    return pixel_values, image_grid_thw, items


def scatter_image_tokenizer_vq_ids(
    items: List[ConversationItem],
    vq_token_ids: List[torch.Tensor],
    image_grid_thw: torch.Tensor,
) -> None:
    """Attach encoded VQ token ids and ``(h, w)`` grids onto image items' ``meta``."""
    for idx, (item, ids) in enumerate(zip(items, vq_token_ids, strict=True)):
        _, h, w = (int(v) for v in image_grid_thw[idx].tolist())
        item.meta[_OMNI_VQ_IDS] = ids.to(dtype=torch.long)
        item.meta[_OMNI_VQ_GRID] = (h, w)


class InferenceMixin:
    """FSM ``generate`` — encodes user/source images into discrete VQ token ids."""

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if conversation_list is None:
            return {"conversation_list": conversation_list}
        pixel_values, image_grid_thw, items = extract_image_tokenizer_inputs(
            [conversation_list],
            image_processor=self._image_processor,
            device=self.device,
            dtype=self.dtype,
        )
        if pixel_values is None:
            return {"conversation_list": conversation_list}
        outputs = self.encode(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        scatter_image_tokenizer_vq_ids(items, outputs["vq_token_ids"], outputs["image_grid_thw"])
        return {"conversation_list": conversation_list}


class LladaUniImageTokenizer(InferenceMixin, PretrainedOmniModule):
    """GlmImage vision + VQ codec → discrete image token ids (unified vocab)."""

    config_class = LladaUniImageTokenizerConfig
    image_processor_class = LladaUniImageProcessor
    preprocessor_class = LladaUniImageTokenizerPreprocessor
    base_model_prefix = "llada2_uni_image_tokenizer"
    main_input_name = "pixel_values"
    _no_split_modules = ["GlmImageVisionBlock"]
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True

    def __init__(self, config: LladaUniImageTokenizerConfig):
        super().__init__(config)
        self.config = config
        # Pin the frozen GlmImage vision tower to sdpa so it cannot inherit a text-side
        # flash attention kernel name from the global ops config.
        self.visual = GlmImageVisionModel._from_config(config.vision_config, attn_implementation="sdpa")
        self.vqmodel = GlmImageVQVAE._from_config(config.vq_config)
        self.image_token_offset = int(config.image_token_offset)
        self.freeze = bool(getattr(config, "freeze", True))
        self._image_processor: Optional[LladaUniImageProcessor] = None
        self.post_init()
        if self.freeze:
            self.requires_grad_(False)

    def freeze_model(self) -> None:
        if self.freeze:
            self.eval()
            self.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.visual.eval()
            self.vqmodel.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        if self.freeze:
            return
        self.visual.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if self.freeze:
            return
        self.visual.gradient_checkpointing_disable()

    def dummy_inputs(self) -> Dict[str, Any]:
        """Minimal 2x2-patch dummy image so an image-less rank still runs the vision tower under FSDP2."""
        vc = self.config.vision_config
        patch_dim = vc.in_channels * (vc.patch_size**2)
        grid = torch.tensor([[1, 2, 2]], dtype=torch.long, device=self.device)
        pixel_values = torch.zeros(int(grid.prod().item()), patch_dim, device=self.device, dtype=self.dtype)
        return {"pixel_values": pixel_values, "image_grid_thw": grid}

    def encode(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        is_dummy = pixel_values is None
        if is_dummy:
            dummy = self.dummy_inputs()
            pixel_values, image_grid_thw = dummy["pixel_values"], dummy["image_grid_thw"]

        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
            image_grid_thw = image_grid_thw.to(self.device)
            vision_out = self.visual(pixel_values, grid_thw=image_grid_thw)
            hidden = vision_out.last_hidden_state
            hidden_size = hidden.shape[-1]

            split_sizes = image_grid_thw.prod(dim=-1).tolist()
            hidden_chunks = torch.split(hidden, split_sizes, dim=0)

            vq_token_ids: List[torch.Tensor] = []
            for i, hs in enumerate(hidden_chunks):
                grid_t, grid_h, grid_w = image_grid_thw[i].tolist()
                hs = hs.view(grid_t, grid_h, grid_w, hidden_size).permute(0, 3, 1, 2).contiguous()
                indices = self.vqmodel.encode(hs).image_tokens.reshape(-1)
                vq_token_ids.append(indices.to(torch.long) + self.image_token_offset)

        if is_dummy:
            return {"vq_token_ids": [], "image_grid_thw": None, "is_dummy": True}
        return {"vq_token_ids": vq_token_ids, "image_grid_thw": image_grid_thw, "is_dummy": False}

    def forward(self, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        return self.encode(**kwargs)


__all__ = [
    "InferenceMixin",
    "LladaUniImageTokenizer",
    "extract_image_tokenizer_inputs",
    "scatter_image_tokenizer_vq_ids",
]
