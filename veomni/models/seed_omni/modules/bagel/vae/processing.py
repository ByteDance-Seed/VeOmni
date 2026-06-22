"""Stateless image and latent helpers for BAGEL VAE."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from ....conversation import ConversationItem, is_dummy
from ..carrier_updates import (
    insert_before,
    materialize_carrier_updates,
    replace_fields,
)
from ..sources import BAGEL_GENERATED_LATENT, BAGEL_VAE_CONTEXT


def preprocess_image(
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
        pil_image = to_rgb_pil(image if isinstance(image, Image.Image) else Image.fromarray(image))
        pil_image = resize_pil(
            pil_image,
            image_stride=image_stride,
            max_image_size=max_image_size,
            min_image_size=min_image_size,
            max_pixels=max_pixels,
        )
        tensor = pil_to_rgb_tensor(pil_image)
    else:
        tensor = to_rgb_tensor(image)
        tensor = resize_tensor(
            tensor,
            image_stride=image_stride,
            max_image_size=max_image_size,
            min_image_size=min_image_size,
            max_pixels=max_pixels,
        )
    mean = torch.tensor(image_mean, dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
    return tensor.sub(mean).div(std)


def to_rgb_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, Image.Image):
        pil_image = to_rgb_pil(image)
        array = np.array(pil_image, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
    if isinstance(image, np.ndarray):
        return to_rgb_tensor(Image.fromarray(image))
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


def resize_tensor(
    tensor: torch.Tensor,
    *,
    image_stride: int,
    max_image_size: int,
    min_image_size: int,
    max_pixels: int,
) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    new_width, new_height = target_size(
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


def resize_pil(
    image: Image.Image,
    *,
    image_stride: int,
    max_image_size: int,
    min_image_size: int,
    max_pixels: int,
    img_num: int = 1,
) -> Image.Image:
    width, height = image.size
    new_width, new_height = target_size(
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


def target_size(
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
    new_width, new_height = apply_scale(width, height, scale, image_stride)
    if new_width * new_height > max_pixels:
        scale = max_pixels / (new_width * new_height)
        new_width, new_height = apply_scale(new_width, new_height, scale, image_stride)
    if max(new_width, new_height) > max_image_size:
        scale = max_image_size / max(new_width, new_height)
        new_width, new_height = apply_scale(new_width, new_height, scale, image_stride)
    return new_width, new_height


def to_rgb_pil(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        rgba = image.convert("RGBA")
        white = Image.new(mode="RGB", size=rgba.size, color=(255, 255, 255))
        white.paste(rgba, mask=rgba.split()[3])
        return white
    return image.convert("RGB")


def pil_to_rgb_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)


def apply_scale(width: int, height: int, scale: float, stride: int) -> tuple[int, int]:
    new_width = round(width * scale)
    new_height = round(height * scale)
    return (
        max(stride, int(round(new_width / stride) * stride)),
        max(stride, int(round(new_height / stride) * stride)),
    )


def looks_raw_image_value(value: Any) -> bool:
    if not torch.is_tensor(value):
        return True
    if value.dim() != 3:
        return False
    return int(value.shape[0]) in (1, 3, 4) or int(value.shape[-1]) in (1, 3, 4)


def latent_grid(value: torch.Tensor) -> torch.Tensor:
    latent = value.detach()
    if latent.dim() == 4 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    if latent.dim() != 3:
        raise ValueError(f"BAGEL VAE decode expects latent grid tensors, got shape {tuple(value.shape)}.")
    return latent


def is_latent_grid_value(value: object) -> bool:
    if not torch.is_tensor(value):
        return False
    if value.dim() == 4:
        return int(value.shape[0]) == 1
    return value.dim() == 3


def raw_image_encode_items(conversation_list: list[list[ConversationItem]] | None) -> list[ConversationItem]:
    return [
        item
        for sample in conversation_list or []
        for item in sample
        if item.type == "image"
        and item.role == "assistant"
        and not is_dummy(item)
        and looks_raw_image_value(item.value)
    ]


def raw_context_image_items(
    conversation_list: list[list[ConversationItem]] | None,
) -> list[tuple[list[ConversationItem], ConversationItem]]:
    return [
        (sample, item)
        for sample in conversation_list or []
        for item in sample
        if item.type == "image"
        and item.role == "user"
        and not is_dummy(item)
        and looks_raw_image_value(item.value)
        and item.source in (None, BAGEL_VAE_CONTEXT)
    ]


def prepare_encode_inputs(
    encode_items: list[ConversationItem],
    *,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    pixel_values = [
        preprocess_image(
            item.value,
            image_stride=config.image_stride,
            max_image_size=config.max_image_size,
            min_image_size=config.min_image_size,
            max_pixels=config.max_pixels,
            image_mean=config.image_mean,
            image_std=config.image_std,
        )
        for item in encode_items
    ]
    return {"pixel_values": torch.stack(pixel_values, dim=0).to(device=device, dtype=dtype)}


def scatter_encoded_latents(
    encode_items: list[ConversationItem],
    latents: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if len(encode_items) != int(latents.shape[0]):
        raise RuntimeError("BAGEL VAE image count mismatch during latent scatter.")
    materialize_carrier_updates(
        None,
        [
            replace_fields(item, type="output", value=latent.to(device=device, dtype=dtype))
            for item, latent in zip(encode_items, latents, strict=True)
        ],
    )


def context_encode_image_items(
    context_items: list[tuple[list[ConversationItem], ConversationItem]],
) -> list[ConversationItem]:
    return [item for _, item in context_items]


def insert_context_encoded_latents(
    context_items: list[tuple[list[ConversationItem], ConversationItem]],
    latents: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if len(context_items) != int(latents.shape[0]):
        raise RuntimeError("BAGEL VAE context image count mismatch during latent scatter.")
    samples = [sample for sample, _ in context_items]
    materialize_carrier_updates(
        samples,
        [
            insert_before(
                image_item,
                ConversationItem(
                    type="output",
                    value=latent.to(device=device, dtype=dtype),
                    role="assistant",
                    source=BAGEL_VAE_CONTEXT,
                    meta={},
                ),
            )
            for (_, image_item), latent in zip(context_items, latents, strict=True)
        ],
    )


def latent_decode_items(conversation_list: list[list[ConversationItem]] | None) -> list[ConversationItem]:
    return [
        item
        for sample in conversation_list or []
        for item in sample
        if item.type == "output"
        and item.source == BAGEL_GENERATED_LATENT
        and not is_dummy(item)
        and is_latent_grid_value(item.value)
    ]


def prepare_decode_inputs(
    decode_items: list[ConversationItem],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    latents = [latent_grid(item.value) for item in decode_items]
    return {"latents": torch.stack(latents, dim=0).to(device=device, dtype=dtype)}


def scatter_decoded_images(
    decode_items: list[ConversationItem],
    pixel_values: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if len(decode_items) != int(pixel_values.shape[0]):
        raise RuntimeError("BAGEL VAE latent count mismatch during decoded image scatter.")
    materialize_carrier_updates(
        None,
        [
            replace_fields(item, type="image", value=image.to(device=device, dtype=dtype))
            for item, image in zip(decode_items, pixel_values, strict=True)
        ],
    )


def as_batched_decode_conversation(conversation_list: Any) -> list[list[Any]]:
    if not conversation_list:
        return []
    first = conversation_list[0]
    if isinstance(first, list):
        return conversation_list
    return [conversation_list]


def decoded_tensor_to_pil(image: torch.Tensor) -> Image.Image:
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
    "apply_scale",
    "as_batched_decode_conversation",
    "decoded_tensor_to_pil",
    "context_encode_image_items",
    "insert_context_encoded_latents",
    "latent_grid",
    "latent_decode_items",
    "is_latent_grid_value",
    "looks_raw_image_value",
    "prepare_decode_inputs",
    "prepare_encode_inputs",
    "preprocess_image",
    "pil_to_rgb_tensor",
    "raw_context_image_items",
    "raw_image_encode_items",
    "resize_pil",
    "resize_tensor",
    "scatter_decoded_images",
    "scatter_encoded_latents",
    "target_size",
    "to_rgb_pil",
    "to_rgb_tensor",
]
