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

"""Training hooks for :class:`LladaVOmni`.

The objective and the batching are the dense module's -- pack the micro-batch into
one row, count the step's samples from ``on_micro_batches_ready``, divide by their
global total, declare ``loss_is_step_mean``. This module adds only the image
inputs, which need two things the dense path does not:

* **The image tensors are concatenated, not stacked.** ``pixel_values`` is
  ``[num_images, 3, H, W]`` for the whole micro-batch, and ``num_images`` differs
  per sample. Concatenating keeps one leading batch axis and matches what
  ``LLaDAVModel.get_image_features`` expects.
* **A batch with no image sends no image keys at all.** The vendored forward then
  runs the tower once on a dummy tile whenever the world size is above one, which
  is what keeps FSDP's all-gathers matched across ranks whose samples differ in
  whether they carry an image. Passing an empty tensor instead would defeat that
  branch.
"""

from __future__ import annotations

from typing import Any

import torch

from ....mixins.base_mixin import BaseMixin
from ....mixins.inference_module_mixin import InferenceModuleMixin
from ...llada.accelerated.accelerated import TrainingMixin as LladaTrainingMixin
from ..modeling import LladaVOmni


class TrainingMixin(LladaTrainingMixin):
    """Dense LLaDA's hooks plus the image inputs."""

    def batch_pre(self, conversation_list: list[Any] = None, **kwargs: Any) -> dict[str, Any]:
        """Pack the text row exactly as the dense module does, then attach the images.

        The text half is delegated rather than copied: packing, the mask-token
        placement and the varlen kwargs all have to match the dense path, and a
        second copy is how those two would drift apart.
        """
        batch = super().batch_pre(conversation_list=conversation_list, **kwargs)
        if not batch:
            return batch

        pixel_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        device = batch["input_ids"].device
        for sample in conversation_list:
            for item in sample:
                value = getattr(item, "value", None)
                if not isinstance(value, dict) or "input_ids" not in value:
                    continue
                pixels = value.get("pixel_values")
                if pixels is not None:
                    pixel_chunks.append(pixels.to(device=device))
                mask = value.get("image_mask")
                if mask is not None:
                    mask_chunks.append(mask.to(device=device, dtype=torch.bool))
                else:
                    # An image-free sample still occupies its span of the packed row.
                    # Filling that span with zeros keeps the mask aligned with the
                    # text; leaving it out would shift every later sample's slots.
                    span = int(value["input_ids"].numel())
                    mask_chunks.append(torch.zeros(span, dtype=torch.bool, device=device))
                break

        if pixel_chunks:
            # `[n_i, 3, H, W]` each -> one `[sum(n_i), 3, H, W]` batch, which is what
            # the tower reads. Per-sample stacking would need equal image counts.
            batch["pixel_values"] = torch.cat(pixel_chunks, dim=0)
            # Slots follow the same concatenation order as the text row, so the
            # packed mask is the per-sample masks laid end to end. Emitted only
            # together with `pixel_values`: a batch with no image at all must carry
            # neither key, so the vendored forward takes its dummy-tile branch and
            # keeps FSDP's all-gathers matched across ranks that do have images.
            batch["image_mask"] = torch.cat(mask_chunks, dim=0).unsqueeze(0)
            if batch["image_mask"].shape != batch["input_ids"].shape:
                raise ValueError(
                    "LladaVOmni: the packed `image_mask` does not cover the packed row "
                    f"({tuple(batch['image_mask'].shape)} vs {tuple(batch['input_ids'].shape)}). "
                    "Every sample must emit an `image_mask`, including the image-free ones."
                )
        return batch


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceModuleMixin):
    pass


class LladaVOmniAccelerated(VeOmniMixin, LladaVOmni):
    pass


__all__ = ["LladaVOmniAccelerated"]
