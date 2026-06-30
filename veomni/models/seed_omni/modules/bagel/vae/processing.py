"""Image processor helpers for BAGEL VAE."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature


class BagelVAEProcessor(BaseImageProcessor):
    """BAGEL VAE image processor.

    Owns raw-image resize and normalization for VAE encode. Carrier selection,
    source tagging, and latent scatter stay in the module mixin.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_stride: int = 16,
        max_image_size: int = 1024,
        min_image_size: int = 512,
        max_pixels: int = 14 * 14 * 9 * 1024,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.image_stride = image_stride
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.max_pixels = max_pixels
        self.image_mean = [0.5, 0.5, 0.5] if image_mean is None else image_mean
        self.image_std = [0.5, 0.5, 0.5] if image_std is None else image_std

    @classmethod
    def from_config(cls, config: Any) -> BagelVAEProcessor:
        return cls(
            image_stride=int(config.image_stride),
            max_image_size=int(config.max_image_size),
            min_image_size=int(config.min_image_size),
            max_pixels=int(config.max_pixels),
            image_mean=list(config.image_mean),
            image_std=list(config.image_std),
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
        pixel_values = [
            _preprocess_image(
                image,
                image_stride=self.image_stride,
                max_image_size=self.max_image_size,
                min_image_size=self.min_image_size,
                max_pixels=self.max_pixels,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )
            for image in image_list
        ]
        pixel_shapes = torch.tensor(
            [[int(pixel.shape[-2]), int(pixel.shape[-1])] for pixel in pixel_values],
            dtype=torch.long,
        )
        pixel_values = torch.stack(_pad_to_batch_size(pixel_values), dim=0)

        if dtype is not None:
            pixel_values = pixel_values.to(dtype=dtype)
        if device is not None:
            pixel_values = pixel_values.to(device=device)
            pixel_shapes = pixel_shapes.to(device=device)
        return BatchFeature(data={"pixel_values": pixel_values, "pixel_shapes": pixel_shapes})

    def postprocess(self, images: torch.Tensor | list[torch.Tensor], **kwargs: Any) -> list[Image.Image]:
        del kwargs
        image_list = images if isinstance(images, list) else [images]
        out: list[Image.Image] = []
        for image in image_list:
            if image.dim() == 4:
                out.extend(_decoded_tensor_to_pil(item) for item in image)
            else:
                out.append(_decoded_tensor_to_pil(image))
        return out


def _preprocess_image(
    image: Any,
    *,
    image_stride: int,
    max_image_size: int,
    min_image_size: int,
    max_pixels: int,
    image_mean: list[float],
    image_std: list[float],
) -> torch.Tensor:
    if isinstance(image, (Image.Image, np.ndarray)):
        pil_image = _to_rgb_pil(image if isinstance(image, Image.Image) else Image.fromarray(image))
        pil_image = _resize_pil(
            pil_image,
            image_stride=image_stride,
            max_image_size=max_image_size,
            min_image_size=min_image_size,
            max_pixels=max_pixels,
        )
        tensor = _pil_to_rgb_tensor(pil_image)
    else:
        tensor = _to_rgb_tensor(image)
        tensor = _resize_tensor(
            tensor,
            image_stride=image_stride,
            max_image_size=max_image_size,
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
            raise TypeError(f"BAGEL VAE image tensor must be 3-D, got shape {tuple(tensor.shape)}.")
        if tensor.shape[0] in (1, 3, 4):
            pass
        elif tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1).contiguous()
        else:
            raise TypeError(f"Unable to infer channel dimension for BAGEL VAE image tensor {tuple(tensor.shape)}.")
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
    raise TypeError(f"BAGEL VAE image item value must be PIL, numpy, or tensor, got {type(image).__name__}.")


def _resize_tensor(
    tensor: torch.Tensor,
    *,
    image_stride: int,
    max_image_size: int,
    min_image_size: int,
    max_pixels: int,
) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    new_width, new_height = _target_size(
        width,
        height,
        image_stride=image_stride,
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        max_pixels=max_pixels,
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
    image_stride: int,
    max_image_size: int,
    min_image_size: int,
    max_pixels: int,
    img_num: int = 1,
) -> Image.Image:
    width, height = image.size
    new_width, new_height = _target_size(
        width,
        height,
        image_stride=image_stride,
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        max_pixels=max_pixels,
    )
    if new_width == width and new_height == height:
        return image
    return TVF.resize(
        image,
        (new_height, new_width),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def _pad_to_batch_size(images: list[torch.Tensor]) -> list[torch.Tensor]:
    if not images:
        return images
    max_height = max(int(image.shape[-2]) for image in images)
    max_width = max(int(image.shape[-1]) for image in images)
    padded = []
    for image in images:
        pad_height = max_height - int(image.shape[-2])
        pad_width = max_width - int(image.shape[-1])
        if pad_height or pad_width:
            image = F.pad(image, (0, pad_width, 0, pad_height), value=0.0)
        padded.append(image)
    return padded


def crop_latent_to_image_shape(
    latent: torch.Tensor,
    pixel_shape: torch.Tensor | None,
    *,
    downsample: int,
) -> torch.Tensor:
    if not torch.is_tensor(pixel_shape):
        return latent

    shape = pixel_shape.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    if int(shape.numel()) != 2:
        raise ValueError("BAGEL VAE pixel shape metadata must contain [height, width].")

    downsample = max(int(downsample), 1)
    latent_height = int(shape[0].item()) // downsample
    latent_width = int(shape[1].item()) // downsample
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError(
            "BAGEL VAE pixel shape metadata produces an empty latent crop: "
            f"pixel_shape={tuple(int(v.item()) for v in shape)}, downsample={downsample}."
        )
    if latent_height > int(latent.shape[-2]) or latent_width > int(latent.shape[-1]):
        raise ValueError(
            "BAGEL VAE latent crop exceeds encoded latent shape: "
            f"crop=({latent_height}, {latent_width}), latent_shape={tuple(latent.shape)}."
        )
    return latent[..., :latent_height, :latent_width]


def _target_size(
    width: int,
    height: int,
    *,
    image_stride: int,
    max_image_size: int,
    min_image_size: int,
    max_pixels: int,
) -> tuple[int, int]:
    scale = min(max_image_size / max(width, height), 1.0)
    scale = max(scale, min_image_size / min(width, height))
    new_width, new_height = _apply_scale(width, height, scale, image_stride)
    if new_width * new_height > max_pixels:
        scale = max_pixels / (new_width * new_height)
        new_width, new_height = _apply_scale(new_width, new_height, scale, image_stride)
    if max(new_width, new_height) > max_image_size:
        scale = max_image_size / max(new_width, new_height)
        new_width, new_height = _apply_scale(new_width, new_height, scale, image_stride)
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


def _decoded_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.dim() == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.dim() != 3:
        raise ValueError(f"BAGEL VAE generated image must be rank 3, got shape {tuple(image.shape)}.")
    image = (image.detach().to(dtype=torch.float32).clamp(-1.0, 1.0) * 0.5 + 0.5).clamp(0.0, 1.0)
    if image.shape[0] in (1, 3, 4):
        image = image[:3].permute(1, 2, 0)
    elif image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unable to infer channel dimension for BAGEL VAE image tensor {tuple(image.shape)}.")
    image = (image[..., :3] * 255.0).round().to(dtype=torch.uint8).cpu().numpy()
    return Image.fromarray(image)


__all__ = [
    "BagelVAEProcessor",
    "crop_latent_to_image_shape",
]
