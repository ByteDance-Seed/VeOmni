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

"""LLaDA2.0-Uni image decoder OmniModule (VQ token ids → PIL image)."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....utils.conversation import ConversationItem
from ...module_modeling_base import PretrainedOmniModule
from .configuration import LladaUniDecoderConfig


SIGNAL_IMAGE_DONE = "image_complete"


class InferenceMixin:
    """FSM ``generate`` — decodes generated VQ token ids into a PIL image."""

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del generation_kwargs, kwargs
        vq_ids, grid = self._find_generated_vq(conversation_list or [])
        if vq_ids is None or grid is None:
            raise RuntimeError(
                "LladaUniDecoder.generate: no generated VQ tokens found on the conversation "
                "(expected meta['llada2_uni_gen_vq_ids'] + meta['llada2_uni_gen_grid'] from the "
                "VQ-generation stage)."
            )
        h, w = int(grid[0]), int(grid[1])
        image = self.decode(vq_token_ids=vq_ids, h=h, w=w)["image"]
        return {
            FSM_SIGNAL_KEY: SIGNAL_IMAGE_DONE,
            "generated": {"type": "image", "value": image},
            "conversation_list": conversation_list,
        }

    @staticmethod
    def _find_generated_vq(conversation_list: List[ConversationItem]):
        for part in reversed(conversation_list):
            vq = part.meta.get("llada2_uni_gen_vq_ids") if isinstance(part.meta, dict) else None
            grid = part.meta.get("llada2_uni_gen_grid") if isinstance(part.meta, dict) else None
            if vq is not None and grid is not None:
                return vq, grid
        return None, None


class LladaUniDecoder(InferenceMixin, PretrainedOmniModule):
    """VQ token ids → image via SigVQ + ZImage DiT + Flux VAE (on-demand loaded)."""

    config_class = LladaUniDecoderConfig
    base_model_prefix = "llada2_uni_decoder"
    main_input_name = "vq_token_ids"
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True

    def __init__(self, config: LladaUniDecoderConfig):
        super().__init__(config)
        self.config = config
        self._anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._asset_dir: Optional[str] = getattr(config, "_name_or_path", None) or None
        self._custom_decode_fn: Optional[Callable[..., Any]] = None
        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        del gradient_checkpointing_kwargs
        return

    def gradient_checkpointing_disable(self) -> None:
        return

    def set_asset_dir(self, path: str) -> None:
        self._asset_dir = path

    def _resolve_asset_dir(self) -> str:
        asset_dir = self._asset_dir or getattr(self.config, "_name_or_path", None)
        if not asset_dir or not os.path.isdir(os.path.join(asset_dir, "decoder")):
            raise RuntimeError(
                "LladaUniDecoder: cannot find decoder assets. Expected a directory containing "
                "`decoder/`, `vae/`, `image_tokenizer/sigvq_embedding.pt` "
                f"(got asset_dir={asset_dir!r}). Run the convert step and load the module from that dir, "
                "or call `module.set_asset_dir(path)`."
            )
        return asset_dir

    @torch.inference_mode()
    def decode(self, vq_token_ids, h: int, w: int) -> Dict[str, Any]:
        if isinstance(vq_token_ids, torch.Tensor):
            vq_token_ids = vq_token_ids.reshape(-1).tolist()
        offset = int(self.config.image_token_offset)
        codes = [(int(t) - offset) if int(t) >= offset else int(t) for t in vq_token_ids]

        if self._custom_decode_fn is not None:
            image = self._custom_decode_fn(codes, int(h), int(w))
            return {"image": image}

        from veomni.models.transformers.llada2_uni_decoder import decode_vq_tokens

        image = decode_vq_tokens(
            codes,
            int(h),
            int(w),
            self._resolve_asset_dir(),
            self.device,
            resolution_multiplier=int(self.config.resolution_multiplier),
            num_steps=int(self.config.decoder_num_steps),
            decode_mode=str(self.config.decode_mode),
        )
        return {"image": image}

    def forward(self, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        return self.decode(**kwargs)

    @property
    def device(self) -> torch.device:
        return self._anchor.device


__all__ = ["InferenceMixin", "LladaUniDecoder", "SIGNAL_IMAGE_DONE"]
