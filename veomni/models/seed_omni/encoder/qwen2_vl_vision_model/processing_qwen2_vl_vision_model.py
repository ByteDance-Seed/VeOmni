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

from typing import List, Optional, Union

import torch
from transformers import BatchFeature
from transformers.image_utils import ImageInput, PILImageResampling
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from ..base import BaseEncoderProcessorMixin


class Qwen2VLVisionModelProcessor(BaseEncoderProcessorMixin, Qwen2VLImageProcessor):
    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        BaseEncoderProcessorMixin.__init__(self, **kwargs)
        Qwen2VLImageProcessor.__init__(
            self,
            do_resize=do_resize,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            **kwargs,
        )

    def process(
        self,
        images: Optional[ImageInput] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        output = self.preprocess(images=images, return_tensors=return_tensors, **kwargs)
        pixel_values = output["pixel_values"]
        image_grid_thw = output["image_grid_thw"].type(torch.int32)
        num_image_tokens = image_grid_thw.prod(dim=-1).type(torch.int32) // (self.merge_size**2)
        return BatchFeature(
            data={"features": pixel_values, "num_tokens": num_image_tokens, "grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )
