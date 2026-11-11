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

"""VeOmni-accelerated ``LladaUniImageTokenizer`` — training-graph and offline-cache hooks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .....mixins.base_mixin import BaseMixin
from .....mixins.metric_meter_mixin import MetricMeterMixin
from .....mixins.offline_encoding_mixin import OfflineEncodingMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ..configuration import LladaUniImageTokenizerConfig
from ..modeling import (
    LladaUniImageTokenizer,
    extract_image_tokenizer_inputs,
    scatter_image_tokenizer_vq_ids,
)
from ..processing import LladaUniImageProcessor


class LladaUniImageTokenizerOfflineMixin(OfflineEncodingMixin):
    """Offline-cache support for ``LladaUniImageTokenizer``.

    * ``offline_encode`` runs ``encode`` and writes the discrete VQ token ids and
      ``(h, w)`` grid onto each image item's ``meta`` (clearing the patch tensor
      ``item.value`` so cached parquet rows store only compact int64 VQ ids).
    * ``online_process`` is a weight-free pass-through on meta device: the cached
      VQ ids are already on ``item.meta["llada2_uni_vq_ids"]`` for ``text_encoder``.
    """

    def offline_encode(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.encode(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    def online_process(self, **kwargs: Any) -> Dict[str, Any]:
        del kwargs
        return {}

    @pre_forward("online_process")
    def online_process_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        return {}

    @post_forward("online_process")
    def online_process_post(self, **outputs: Any) -> Dict[str, Any]:
        del outputs
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        return {"conversation_list": conversation}


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks for :class:`LladaUniImageTokenizer`."""

    config: LladaUniImageTokenizerConfig
    device: torch.device
    dtype: torch.dtype
    _image_processor: Optional[LladaUniImageProcessor]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        self._encode_items: List[ConversationItem] = []

    @pre_forward("encode", "offline_encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        pixel_values, image_grid_thw, items = extract_image_tokenizer_inputs(
            conversation_list,
            image_processor=self._image_processor,
            device=self.device,
            dtype=self.dtype,
        )
        self._encode_items = items
        seqlens = [int(g.prod().item()) for g in image_grid_thw] if image_grid_thw is not None else []
        self.metric_meter_set_seqlens("encode", seqlens)
        self.metric_meter_set_seqlens("offline_encode", seqlens)
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        items = self._encode_items
        self._conversation_carrier = None
        self._encode_items = []
        if outputs.get("is_dummy", False):
            return {"conversation_list": conversation}
        vq_token_ids: List[torch.Tensor] = outputs.get("vq_token_ids") or []
        grid_thw = outputs.get("image_grid_thw")
        scatter_image_tokenizer_vq_ids(items, vq_token_ids, grid_thw)
        return {"conversation_list": conversation}

    @post_forward("offline_encode")
    def offline_encode_post(self, **outputs: Any) -> Dict[str, Any]:
        out = self.encode_post(**outputs)
        for sample in out["conversation_list"] or []:
            for item in sample:
                if item.type == "image" and item.meta.get("llada2_uni_vq_ids") is not None:
                    item.value = item.meta["llada2_uni_vq_ids"].detach().cpu()
        return out


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for the frozen GlmImage VQ tokenizer."""

    config: LladaUniImageTokenizerConfig

    def estimate_flops(self, seqlens: List[int]) -> float:
        del seqlens
        return 0.0


class VeOmniMixin(BaseMixin, LladaUniImageTokenizerOfflineMixin, TrainingMixin, MeterMixin):
    config: LladaUniImageTokenizerConfig


class LladaUniImageTokenizerAccelerated(VeOmniMixin, LladaUniImageTokenizer):
    pass


__all__ = [
    "LladaUniImageTokenizerAccelerated",
    "LladaUniImageTokenizerOfflineMixin",
    "TrainingMixin",
    "VeOmniMixin",
]
