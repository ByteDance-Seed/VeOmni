# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hunyuan Image 3 processor.

The resize/crop and ``[-1, 1]`` normalization follow Tencent's
``image_processor.py`` at revision ``6e9113a692a27a0751d82aba3b2015a876646c03``.

:class:`HunyuanImage3ImageProcessor` takes the target ``image_size`` as a
**runtime** argument on :meth:`preprocess` rather than baking it in, so per-run
YAML resolution changes need no processor rebuild (and a future bucketing
sampler can vary it per batch). :class:`HunyuanImage3Processor` bundles it with
the tokenizer, mirroring HF's ``ProcessorMixin`` layout (``.tokenizer`` +
``.image_processor``) without inheriting from it, since this image processor is
not a HF ``BaseImageProcessor``.

Aspect-ratio bucketing and the SigLIP2 vision branch are deliberately absent:
T2I trains at a fixed resolution with no ViT, and the vendored machinery for
both was never exercised. An IT2I / I2T capability adds them back against a
concrete forward that needs them.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as TF

from ...auto import build_config


def resize_and_crop(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Aspect-fill (cover) resize + center crop to ``target_size == (W, H)``."""
    tw, th = target_size
    w, h = image.size

    tr = th / tw
    r = h / w

    if r < tr:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    image = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)

    crop_top = int(round((resize_height - th) / 2.0))
    crop_left = int(round((resize_width - tw) / 2.0))

    image = image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))
    return image


@dataclass(frozen=True)
class ProcessedImage:
    """One preprocessed target image: normalized pixels plus its token grid.

    ``image_tensor`` is ``[1, C, H, W]`` in ``[-1, 1]`` (the leading batch axis
    matches the official processor, and the collator stacks over it).
    ``token_height`` / ``token_width`` are the VAE-latent grid the sequence
    layout reserves payload positions for.
    """

    image_tensor: Tensor
    token_height: int
    token_width: int

    @property
    def grid_hw(self) -> tuple[int, int]:
        return self.token_height, self.token_width


class HunyuanImage3ImageProcessor:
    """Fixed-resolution VeOmni T2I image preprocessor.

    Reads only ``config.vae_downsample_factor`` and ``config.patch_size``; their
    product is the pixel stride one latent token covers.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.vae_processor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
            ]
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        *,
        config_kwargs: Optional[dict] = None,
        **_ignored,
    ) -> "HunyuanImage3ImageProcessor":
        """Build from a HuggingFace checkpoint directory.

        ``**_ignored`` swallows the ``padding_side`` / ``trust_remote_code`` /
        ``max_pixels`` kwargs :func:`veomni.models.auto.build_processor` splats
        in for Qwen-VL-style processors -- they have no meaning here.
        """
        return cls(build_config(pretrained_path, **dict(config_kwargs or {})))

    def preprocess(
        self,
        image: Image.Image,
        *,
        image_size: Tuple[int, int],
        random_flip: bool = False,
    ) -> ProcessedImage:
        """Resize/crop ``image`` to ``image_size == (height, width)`` and normalize to ``[-1, 1]``."""
        if not isinstance(image, Image.Image):
            raise TypeError("HunyuanImage3ImageProcessor expects a PIL image.")
        # YAML lists / JSON arrays arrive as ``list``; coerce so both call styles
        # share the same downstream check.
        if isinstance(image_size, list):
            image_size = tuple(image_size)
        if not (
            isinstance(image_size, tuple)
            and len(image_size) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in image_size)
        ):
            raise ValueError(f"image_size must be a (height, width) tuple of positive ints, got {image_size!r}.")

        image = image.convert("RGB")
        target_height, target_width = image_size
        resized_image = resize_and_crop(image, (target_width, target_height))

        # Applied post-resize/crop so the crop geometry is unaffected.
        if random_flip and bool(torch.rand(()).item() < 0.5):
            resized_image = TF.hflip(resized_image)

        image_tensor = self.vae_processor(resized_image)
        if image_tensor.shape[-2:] != (target_height, target_width):
            raise RuntimeError(
                f"Preprocessed image shape {tuple(image_tensor.shape[-2:])} != "
                f"requested (height, width) ({target_height}, {target_width})."
            )

        vae_stride_h = int(self.config.vae_downsample_factor[0]) * int(self.config.patch_size)
        vae_stride_w = int(self.config.vae_downsample_factor[1]) * int(self.config.patch_size)
        if target_height % vae_stride_h or target_width % vae_stride_w:
            raise ValueError(
                f"image_size {image_size} must be a multiple of vae_downsample_factor * patch_size "
                f"= ({vae_stride_h}, {vae_stride_w})."
            )

        return ProcessedImage(
            image_tensor=image_tensor.unsqueeze(0),  # add batch dim (matches official)
            token_height=target_height // vae_stride_h,
            token_width=target_width // vae_stride_w,
        )


class HunyuanImage3Processor:
    """Bundle: text tokenizer + T2I image preprocessor."""

    def __init__(self, *, tokenizer, image_processor: HunyuanImage3ImageProcessor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        *,
        config_kwargs: Optional[dict] = None,
        **_ignored,
    ) -> "HunyuanImage3Processor":
        """Load tokenizer + image processor from a HuggingFace checkpoint directory.

        The tokenizer load is **best-effort**: toy configs ship no tokenizer
        files, and the registered data transform tolerates ``tokenizer=None``
        via a length-based fallback (see ``process_sample_hunyuan_image_3``).
        ``trust_remote_code=False`` is safe -- the checkpoint ships a stock
        tokenizer and our config class is already registered.
        """
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_path, trust_remote_code=False)
        except Exception:
            # Toy path / checkpoints without a tokenizer file.
            tokenizer = None
        return cls(
            tokenizer=tokenizer,
            image_processor=HunyuanImage3ImageProcessor.from_pretrained(pretrained_path, config_kwargs=config_kwargs),
        )


__all__ = [
    "HunyuanImage3ImageProcessor",
    "HunyuanImage3Processor",
    "ProcessedImage",
    "resize_and_crop",
]
