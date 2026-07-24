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

"""Model-local HunyuanImage 3 image processor for T2I target images.

Resolves a versioned resolution bucket, resizes/crops the target image to that
bucket, and normalizes to ``[-1, 1]`` with the official
``ToTensor + Normalize([0.5], [0.5])`` transform (see Tencent's
``HunyuanImage3ImageProcessor.pil_image_to_tensor`` at revision
``6e9113a692a27a0751d82aba3b2015a876646c03``). Resize/crop/flip parameters are
returned as explicit metadata so the online-VAE path and a posterior cache share
one geometry. No floating Hub Python is imported.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from .resolution_policy import (
    HunyuanImage3ResolutionPolicy,
    ResolutionBucket,
    ResolutionPolicyConfig,
    build_resolution_policy,
)


@dataclass
class ProcessedTargetImage:
    pixel_values: torch.Tensor  # [C, H, W] in [-1, 1]
    bucket: ResolutionBucket
    resize_height: int
    resize_width: int
    crop_top: int
    crop_left: int
    flip: bool
    original_width: int
    original_height: int

    @property
    def base_size(self) -> int:
        return self.bucket.base_size

    @property
    def ratio_index(self) -> int:
        return self.bucket.ratio_index

    @property
    def grid_hw(self) -> tuple[int, int]:
        return self.bucket.grid_hw

    @property
    def transform_metadata(self) -> dict[str, int | bool]:
        # The exact geometry a posterior cache must record to stay online/cache
        # consistent (impl §6.1 / §8.2).
        return {
            "base_size": self.bucket.base_size,
            "ratio_index": self.bucket.ratio_index,
            "image_height": self.bucket.height,
            "image_width": self.bucket.width,
            "resize_height": self.resize_height,
            "resize_width": self.resize_width,
            "crop_top": self.crop_top,
            "crop_left": self.crop_left,
            "flip": self.flip,
            "original_width": self.original_width,
            "original_height": self.original_height,
        }


@dataclass
class HunyuanImage3ImageProcessorConfig:
    resolution_policy: ResolutionPolicyConfig = field(default_factory=ResolutionPolicyConfig)
    vae_spatial_factor: int = 16
    patch_size: int = 1
    default_base_size: int = 1024
    random_flip: bool = False

    def __post_init__(self):
        if isinstance(self.resolution_policy, dict):
            self.resolution_policy = ResolutionPolicyConfig(**self.resolution_policy)


class HunyuanImage3ImageProcessor:
    """Bucket-aware target-image preprocessing for ``single_gen_t2i_v1``."""

    def __init__(self, config: Optional[HunyuanImage3ImageProcessorConfig] = None):
        self.config = config or HunyuanImage3ImageProcessorConfig()
        self.resolution_policy: HunyuanImage3ResolutionPolicy = build_resolution_policy(
            self.config.resolution_policy,
            vae_spatial_factor=self.config.vae_spatial_factor,
            patch_size=self.config.patch_size,
        )
        # Official normalization: ToTensor + Normalize([0.5], [0.5]) -> [-1, 1].
        self.pil_image_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def select_bucket(self, *, width: int, height: int, base_size: Optional[int] = None) -> ResolutionBucket:
        return self.resolution_policy.select_bucket(
            width=width,
            height=height,
            base_size=base_size or self.config.default_base_size,
        )

    def preprocess_target_image(
        self,
        image: Image.Image,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        base_size: Optional[int] = None,
        flip: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
        bucket: Optional[ResolutionBucket] = None,
    ) -> ProcessedTargetImage:
        """Resize/crop ``image`` to its bucket and normalize to ``[-1, 1]``.

        When ``bucket`` is provided (the scheduled multi-resolution data plane,
        impl §6.2) the exact ``base_size`` + ``ratio_index`` geometry is forced
        and the natural aspect-ratio ``select_bucket`` step is skipped, so every
        DP rank resizes its own sample to the identical scheduled bucket. When
        ``bucket`` is ``None`` the behavior is byte-identical to before: the
        bucket is chosen from the sample's own resolution.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("preprocess_target_image expects a PIL image.")
        image = image.convert("RGB")
        original_width = width if width is not None else image.width
        original_height = height if height is not None else image.height
        if original_width <= 0 or original_height <= 0:
            raise ValueError("Image width and height must be positive.")

        if bucket is None:
            bucket = self.select_bucket(width=original_width, height=original_height, base_size=base_size)
        target_height, target_width = bucket.height, bucket.width

        # Aspect-fill resize (scale to cover), then a centered crop -- the VeOmni
        # Text2ImageDataset convention. Crop offsets are recorded so a posterior
        # cache reproduces the exact pixels.
        scale = max(target_width / image.width, target_height / image.height)
        resize_width = max(target_width, round(image.width * scale))
        resize_height = max(target_height, round(image.height * scale))
        resized = image.resize((resize_width, resize_height), Image.BICUBIC)
        crop_top = (resize_height - target_height) // 2
        crop_left = (resize_width - target_width) // 2
        cropped = resized.crop((crop_left, crop_top, crop_left + target_width, crop_top + target_height))

        apply_flip = self.config.random_flip if flip is None else flip
        if apply_flip and flip is None:
            apply_flip = bool(torch.rand((), generator=generator).item() < 0.5)
        if apply_flip:
            cropped = TF.hflip(cropped)

        pixel_values = self.pil_image_to_tensor(cropped)
        if pixel_values.shape[-2:] != (target_height, target_width):
            raise RuntimeError("Preprocessed image does not match the selected bucket geometry.")
        return ProcessedTargetImage(
            pixel_values=pixel_values,
            bucket=bucket,
            resize_height=resize_height,
            resize_width=resize_width,
            crop_top=crop_top,
            crop_left=crop_left,
            flip=bool(apply_flip),
            original_width=original_width,
            original_height=original_height,
        )


__all__ = [
    "HunyuanImage3ImageProcessor",
    "HunyuanImage3ImageProcessorConfig",
    "ProcessedTargetImage",
]
