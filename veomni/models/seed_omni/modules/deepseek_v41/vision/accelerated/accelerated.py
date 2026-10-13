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

"""Training-graph hooks for the DeepSeek-V4.1 vision tower.

``generate()`` lives on the native :class:`DeepseekV41VisionEncoder`; this file
only carries the pre/post forward hooks that move the conversation carrier in and
the assembled spans out.

No sequence-parallel branch. The ViT's attention is per-image and images are
already packed varlen, so an Ulysses slice would have to respect image boundaries
that vary per sample; the tower is 0.49B against a 40-layer backbone, so it is not
where the step time is. Revisit under G4 if a long-image workload says otherwise.
"""

from typing import Any, Dict, List, Optional

import torch

from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem, iter_desired_items
from ..configuration import DeepseekV41VisionEncoderConfig
from ..modeling import (
    DeepseekV41VisionEncoder,
    _VisualOutputSlot,
    build_deepseek_v41_vit_batch,
    build_deepseek_v41_vit_metadata,
    scatter_deepseek_v41_visual_embeds,
)


class TrainingMixin(TrainingModuleMixin):
    """Graph hooks for the DeepSeek-V4.1 ViT (training path).

    Every user image in the micro-batch goes through one packed forward. When a
    micro-batch holds no image at all, the CPU worker has appended a single
    ``is_dummy`` user-image anchor, which travels the same path -- it carries patches
    and a grid exactly like a real one, so the FSDP collectives still fire and the
    branch stays out of the forward.
    """

    config: DeepseekV41VisionEncoderConfig
    device: torch.device
    dtype: torch.dtype

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        self._visual_output_slots: Optional[List[_VisualOutputSlot]] = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        items = list(iter_desired_items(conversation_list, types=["image"], roles=["user"]))
        real_items = [item for item in items if not item.is_dummy]
        items = real_items or items
        pixel_values, grids, token_types, slots = build_deepseek_v41_vit_batch(
            items, device=self.device, dtype=self.dtype
        )
        self._visual_output_slots = slots
        return {
            "pixel_values": pixel_values,
            "grids": grids,
            "token_types": token_types,
            "vit_metadata": build_deepseek_v41_vit_metadata(
                grids, self.config.rope_dim, self.config.rope_theta, self.device
            ),
        }

    @post_forward("forward")
    def forward_post(self, image_embeds: torch.Tensor) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        slots = self._visual_output_slots
        self._conversation_carrier = None
        self._visual_output_slots = None
        scatter_deepseek_v41_visual_embeds(slots, image_embeds)
        return {"conversation_list": conversation}

    def dummy_inputs(self) -> Dict[str, Any]:
        ratio, patch = self.config.downsample_ratio, self.config.patch_size
        grids = [[ratio, ratio]]
        from ..processing import image_token_types

        return {
            "pixel_values": torch.zeros(ratio * ratio, 3, patch, patch, device=self.device, dtype=self.dtype),
            "grids": grids,
            "token_types": [image_token_types(1, 1)],
            "vit_metadata": build_deepseek_v41_vit_metadata(
                grids, self.config.rope_dim, self.config.rope_theta, self.device
            ),
        }


class VeOmniMixin(BaseMixin, TrainingMixin):
    """``generate()`` already lives on the native encoder, so no inference mixin here."""


class DeepseekV41VisionEncoderAccelerated(VeOmniMixin, DeepseekV41VisionEncoder):
    pass


__all__ = ["DeepseekV41VisionEncoderAccelerated"]
