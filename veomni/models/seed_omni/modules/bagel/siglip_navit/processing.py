"""Image processor + worker-side preprocessor for BAGEL SigLIP NaViT.

:class:`BagelSiglipNavitPreprocessor` is the picklable, weight-free CPU worker
counterpart (see :class:`~veomni.models.seed_omni.mixins.module_processor_mixin.Preprocessor`) —
built straight off the checkpoint dir, with no model instance involved. Unlike
Janus, BAGEL ships no separate ``preprocessor_config.json``: the image
processor is fully derived from the module's own ``config.json`` via
:meth:`BagelSiglipNavitProcessor.from_config`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature

from ....mixins.module_processor_mixin import Preprocessor
from ....utils.conversation import ConversationItem, iter_desired_items
from ..sources import BAGEL_SIGLIP_CONTEXT
from .configuration import BagelSiglipNavitConfig


class BagelSiglipNavitProcessor(BaseImageProcessor):
    """BAGEL SigLIP NaViT image processor.

    Converts raw images into flattened patch rows plus the varlen metadata that
    the NaViT tower consumes. Carrier selection and embed scatter stay in the
    module mixin because they are SeedOmni conversation semantics.
    """

    model_input_names = [
        "patchified_pixel_values",
        "patchified_position_ids",
        "cu_seqlens",
        "max_seqlen",
        "token_lens",
    ]

    def __init__(
        self,
        patch_size: int = 14,
        image_size: int = 980,
        min_image_size: int = 378,
        max_pixels: int = 14 * 14 * 9 * 1024,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        vit_max_num_patch_per_side: int = 70,
        num_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.image_size = image_size
        self.min_image_size = min_image_size
        self.max_pixels = max_pixels
        self.image_mean = [0.5, 0.5, 0.5] if image_mean is None else image_mean
        self.image_std = [0.5, 0.5, 0.5] if image_std is None else image_std
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.num_channels = num_channels

    @classmethod
    def from_config(cls, config: Any) -> BagelSiglipNavitProcessor:
        return cls(
            patch_size=int(config.patch_size),
            image_size=int(config.image_size),
            min_image_size=int(config.min_image_size),
            max_pixels=int(config.max_pixels),
            image_mean=list(config.image_mean),
            image_std=list(config.image_std),
            vit_max_num_patch_per_side=int(config.vit_max_num_patch_per_side),
            num_channels=int(config.num_channels),
        )

    def preprocess(
        self,
        images: Any,
        *,
        return_tensors: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        del return_tensors, kwargs
        image_list = images if isinstance(images, list) else [images]

        pixel_values: list[torch.Tensor] = []
        position_ids: list[torch.Tensor] = []
        token_lens: list[int] = []
        for image in image_list:
            image_tensor = _preprocess_image(
                image,
                patch_size=self.patch_size,
                image_size=self.image_size,
                min_image_size=self.min_image_size,
                max_pixels=self.max_pixels,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )
            patches = _patchify_image(image_tensor, self.patch_size)
            positions = _flattened_position_ids(
                image_tensor.shape[-2],
                image_tensor.shape[-1],
                patch_size=self.patch_size,
                max_num_patches_per_side=self.vit_max_num_patch_per_side,
            )
            pixel_values.append(patches)
            position_ids.append(positions)
            token_lens.append(int(patches.shape[0]))

        tensor_device = torch.device("cpu") if device is None else device
        token_lens_tensor = torch.tensor(token_lens, dtype=torch.int32, device=tensor_device)
        pixel_tensor = torch.cat(pixel_values, dim=0)
        if dtype is not None:
            pixel_tensor = pixel_tensor.to(dtype=dtype)
        return BatchFeature(
            data={
                "patchified_pixel_values": pixel_tensor.to(device=tensor_device),
                "patchified_position_ids": torch.cat(position_ids, dim=0).to(device=tensor_device, dtype=torch.long),
                "cu_seqlens": F.pad(torch.cumsum(token_lens_tensor, dim=0), (1, 0)).to(torch.int32),
                "max_seqlen": int(token_lens_tensor.max().item()),
                "token_lens": token_lens_tensor,
            }
        )


def _preprocess_image(
    image: Any,
    *,
    patch_size: int,
    image_size: int,
    min_image_size: int,
    max_pixels: int,
    image_mean: list[float],
    image_std: list[float],
) -> torch.Tensor:
    if isinstance(image, (Image.Image, np.ndarray)):
        pil_image = _to_rgb_pil(image if isinstance(image, Image.Image) else Image.fromarray(image))
        pil_image = _resize_pil(
            pil_image,
            patch_size=patch_size,
            image_size=image_size,
            min_image_size=min_image_size,
            max_pixels=max_pixels,
        )
        tensor = _pil_to_rgb_tensor(pil_image)
    else:
        tensor = _to_rgb_tensor(image)
        tensor = _resize_tensor(
            tensor,
            patch_size=patch_size,
            image_size=image_size,
            min_image_size=min_image_size,
            max_pixels=max_pixels,
        )
    mean = torch.tensor(image_mean, dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
    return tensor.sub(mean).div(std)


def _to_rgb_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, Image.Image):
        pil_image = _to_rgb_pil(image)
        array = np.array(pil_image, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
    if isinstance(image, np.ndarray):
        return _to_rgb_tensor(Image.fromarray(image))
    if torch.is_tensor(image):
        tensor = image.detach().to(dtype=torch.float32)
        if tensor.dim() != 3:
            raise TypeError(f"BAGEL image tensor must be 3-D, got shape {tuple(tensor.shape)}.")
        if tensor.shape[0] in (1, 3, 4):
            pass
        elif tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1).contiguous()
        else:
            raise TypeError(f"Unable to infer channel dimension for BAGEL image tensor {tuple(tensor.shape)}.")
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] == 4:
            alpha = tensor[3:4]
            if alpha.max() > 1:
                alpha = alpha / 255.0
            rgb = tensor[:3]
            if rgb.max() > 1:
                rgb = rgb / 255.0
            tensor = rgb * alpha + (1.0 - alpha)
        if tensor.max() > 1:
            tensor = tensor / 255.0
        return tensor[:3].clamp(0.0, 1.0)
    raise TypeError(f"BAGEL image item value must be PIL, numpy, or tensor, got {type(image).__name__}.")


def _resize_tensor(
    tensor: torch.Tensor,
    *,
    patch_size: int,
    image_size: int,
    min_image_size: int,
    max_pixels: int,
    img_num: int = 1,
) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    new_width, new_height = _target_size(
        width,
        height,
        patch_size=patch_size,
        image_size=image_size,
        min_image_size=min_image_size,
        max_pixels=max_pixels,
        img_num=img_num,
    )
    if new_width == width and new_height == height:
        return tensor
    resized = F.interpolate(
        tensor.unsqueeze(0),
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.squeeze(0).clamp(0.0, 1.0)


def _resize_pil(
    image: Image.Image,
    *,
    patch_size: int,
    image_size: int,
    min_image_size: int,
    max_pixels: int,
    img_num: int = 1,
) -> Image.Image:
    width, height = image.size
    new_width, new_height = _target_size(
        width,
        height,
        patch_size=patch_size,
        image_size=image_size,
        min_image_size=min_image_size,
        max_pixels=max_pixels,
        img_num=img_num,
    )
    if new_width == width and new_height == height:
        return image
    return TVF.resize(
        image,
        (new_height, new_width),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def _target_size(
    width: int,
    height: int,
    *,
    patch_size: int,
    image_size: int,
    min_image_size: int,
    max_pixels: int,
    img_num: int,
) -> tuple[int, int]:
    scale = min(image_size / max(width, height), 1.0)
    scale = max(scale, min_image_size / min(width, height))
    new_width, new_height = _apply_scale(width, height, scale, patch_size)
    if new_width * new_height > max_pixels / img_num:
        scale = max_pixels / img_num / (new_width * new_height)
        new_width, new_height = _apply_scale(new_width, new_height, scale, patch_size)
    if max(new_width, new_height) > image_size:
        scale = image_size / max(new_width, new_height)
        new_width, new_height = _apply_scale(new_width, new_height, scale, patch_size)
    return new_width, new_height


def _to_rgb_pil(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        rgba = image.convert("RGBA")
        white = Image.new(mode="RGB", size=rgba.size, color=(255, 255, 255))
        white.paste(rgba, mask=rgba.split()[3])
        return white
    return image.convert("RGB")


def _pil_to_rgb_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)


def _apply_scale(width: int, height: int, scale: float, stride: int) -> tuple[int, int]:
    new_width = round(width * scale)
    new_height = round(height * scale)
    return (
        max(stride, int(round(new_width / stride) * stride)),
        max(stride, int(round(new_height / stride) * stride)),
    )


def _patchify_image(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    channels, height, width = image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("BAGEL preprocessed image height and width must be divisible by patch_size.")
    image = image.reshape(channels, height // patch_size, patch_size, width // patch_size, patch_size)
    image = torch.einsum("chpwq->hwpqc", image)
    return image.reshape(-1, patch_size**2 * channels)


def _flattened_position_ids(
    height: int,
    width: int,
    *,
    patch_size: int,
    max_num_patches_per_side: int,
) -> torch.Tensor:
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    coords_h = torch.arange(0, num_patches_h, dtype=torch.long)
    coords_w = torch.arange(0, num_patches_w, dtype=torch.long)
    return (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()


_OMNI_POSITION_IDS = "bagel_siglip_navit_position_ids"
_OMNI_TOKEN_LEN = "bagel_siglip_navit_token_len"


class BagelSiglipNavitPreprocessor(Preprocessor):
    """Worker-side image patchify for BAGEL SigLIP NaViT context images."""

    def __init__(
        self,
        image_processor: Any,
        dtype: torch.dtype | None = None,
        dummy_pixel_values: torch.Tensor | None = None,
    ) -> None:
        self._image_processor = image_processor
        self._dtype = dtype
        self._dummy_pixel_values = dummy_pixel_values

    @classmethod
    def from_pretrained(
        cls, module_path: str, *, config_overrides: dict[str, Any] | None = None, **kwargs: Any
    ) -> BagelSiglipNavitPreprocessor:
        """Build straight from the checkpoint dir — no model instance needed.

        BAGEL ships no standalone ``preprocessor_config.json``; the image
        processor is derived from the module's own ``config.json``.

        ``config_overrides`` (the module's YAML ``model_config:`` block) is
        applied on top of the on-disk default before deriving the image
        processor's geometry, so it agrees with the live model's own config.
        """
        del kwargs
        config = BagelSiglipNavitConfig.from_pretrained(module_path, **(config_overrides or {}))
        return cls(BagelSiglipNavitProcessor.from_config(config))

    def bind_dummy_inputs(self, config: BagelSiglipNavitConfig, dtype: torch.dtype | None = None) -> None:
        """Training-only FSDP-anchor dummy — pure ``(config, dtype)``, no live model."""
        patch_dim = int(config.num_channels) * int(config.patch_size) * int(config.patch_size)
        self._dtype = dtype
        self._dummy_pixel_values = torch.zeros(1, patch_dim, dtype=dtype)

    def __call__(
        self,
        conversation_list: list[list[ConversationItem]],
        *,
        inference: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        del generation_kwargs

        image_items: list[ConversationItem] = []
        for sample in conversation_list:
            sample_image_items = list(iter_desired_items([sample], types=["image"], sources=[BAGEL_SIGLIP_CONTEXT]))
            if sample_image_items:
                image_items.extend(sample_image_items)
            elif not inference:
                if self._dummy_pixel_values is None:
                    raise RuntimeError(
                        f"{type(self).__name__}: dummy inputs not bound — call bind_dummy_inputs() "
                        "before training use (pure inference never reaches this branch)."
                    )
                sample.append(
                    ConversationItem(
                        type="image",
                        value=self._dummy_pixel_values.to(dtype=self._dtype).clone(),
                        role="dummy",
                        source=BAGEL_SIGLIP_CONTEXT,
                        meta={
                            _OMNI_POSITION_IDS: torch.zeros(1, dtype=torch.long),
                            _OMNI_TOKEN_LEN: 1,
                        },
                    )
                )

        if not image_items:
            return

        inputs = self._image_processor(
            images=[item.value for item in image_items], return_tensors="pt", dtype=self._dtype
        )
        lengths = inputs["token_lens"].detach().cpu().reshape(-1).tolist()
        pixel_chunks = torch.split(inputs["patchified_pixel_values"], lengths, dim=0)
        position_chunks = torch.split(inputs["patchified_position_ids"], lengths, dim=0)
        for item, pixels, position_ids, length in zip(
            image_items, pixel_chunks, position_chunks, lengths, strict=True
        ):
            item.value = pixels.to(dtype=self._dtype)
            item.source = BAGEL_SIGLIP_CONTEXT
            item.meta[_OMNI_POSITION_IDS] = position_ids.to(dtype=torch.long)
            item.meta[_OMNI_TOKEN_LEN] = int(length)


__all__ = [
    "BagelSiglipNavitProcessor",
    "BagelSiglipNavitPreprocessor",
]
