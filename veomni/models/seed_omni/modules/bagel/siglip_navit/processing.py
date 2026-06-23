"""Image processor helpers for BAGEL SigLIP NaViT."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature


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


__all__ = [
    "BagelSiglipNavitProcessor",
]
