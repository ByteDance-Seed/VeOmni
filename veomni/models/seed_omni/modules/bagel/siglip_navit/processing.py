"""Stateless image preprocessing helpers for BAGEL SigLIP NaViT."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from ....conversation import ConversationItem, is_dummy


def preprocess_image(
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
        pil_image = to_rgb_pil(image if isinstance(image, Image.Image) else Image.fromarray(image))
        pil_image = resize_pil(
            pil_image,
            patch_size=patch_size,
            image_size=image_size,
            min_image_size=min_image_size,
            max_pixels=max_pixels,
        )
        tensor = pil_to_rgb_tensor(pil_image)
    else:
        tensor = to_rgb_tensor(image)
        tensor = resize_tensor(
            tensor,
            patch_size=patch_size,
            image_size=image_size,
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


def resize_tensor(
    tensor: torch.Tensor,
    *,
    patch_size: int,
    image_size: int,
    min_image_size: int,
    max_pixels: int,
    img_num: int = 1,
) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    new_width, new_height = target_size(
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


def resize_pil(
    image: Image.Image,
    *,
    patch_size: int,
    image_size: int,
    min_image_size: int,
    max_pixels: int,
    img_num: int = 1,
) -> Image.Image:
    width, height = image.size
    new_width, new_height = target_size(
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


def target_size(
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
    new_width, new_height = apply_scale(width, height, scale, patch_size)
    if new_width * new_height > max_pixels / img_num:
        scale = max_pixels / img_num / (new_width * new_height)
        new_width, new_height = apply_scale(new_width, new_height, scale, patch_size)
    if max(new_width, new_height) > image_size:
        scale = image_size / max(new_width, new_height)
        new_width, new_height = apply_scale(new_width, new_height, scale, patch_size)
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


def patchify_image(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    channels, height, width = image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("BAGEL preprocessed image height and width must be divisible by patch_size.")
    image = image.reshape(channels, height // patch_size, patch_size, width // patch_size, patch_size)
    image = torch.einsum("chpwq->hwpqc", image)
    return image.reshape(-1, patch_size**2 * channels)


def flattened_position_ids(
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


def iter_conversation_items(
    conversation_list: list[ConversationItem] | list[list[ConversationItem]],
) -> list[ConversationItem]:
    if not conversation_list:
        return []
    if isinstance(conversation_list[0], ConversationItem):
        return conversation_list  # type: ignore[return-value]
    return [item for sample in conversation_list for item in sample]  # type: ignore[union-attr]


def is_encoded_image_value(value: Any, *, output_size: int) -> bool:
    return torch.is_tensor(value) and value.dim() == 2 and value.shape[-1] == output_size


def user_raw_image_items(
    conversation_list: list[ConversationItem] | list[list[ConversationItem]],
    *,
    output_size: int,
) -> list[ConversationItem]:
    return [
        item
        for item in iter_conversation_items(conversation_list)
        if item.type == "image"
        and item.role == "user"
        and not is_dummy(item)
        and not is_encoded_image_value(item.value, output_size=output_size)
    ]


def image_items(
    conversation_list: list[list[ConversationItem]] | None,
) -> list[ConversationItem]:
    return [
        item for sample in conversation_list or [] for item in sample if item.type == "image" and not is_dummy(item)
    ]


def prepare_image_batch(
    image_items: list[ConversationItem],
    *,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    pixel_values: list[torch.Tensor] = []
    position_ids: list[torch.Tensor] = []
    token_lens: list[int] = []
    for item in image_items:
        image_tensor = preprocess_image(
            item.value,
            patch_size=config.patch_size,
            image_size=config.image_size,
            min_image_size=config.min_image_size,
            max_pixels=config.max_pixels,
            image_mean=config.image_mean,
            image_std=config.image_std,
        )
        patches = patchify_image(image_tensor, config.patch_size)
        positions = flattened_position_ids(
            image_tensor.shape[-2],
            image_tensor.shape[-1],
            patch_size=config.patch_size,
            max_num_patches_per_side=config.vit_max_num_patch_per_side,
        )
        pixel_values.append(patches)
        position_ids.append(positions)
        token_lens.append(int(patches.shape[0]))

    token_lens_tensor = torch.tensor(token_lens, dtype=torch.int32, device=device)
    return {
        "patchified_pixel_values": torch.cat(pixel_values, dim=0).to(device=device, dtype=dtype),
        "patchified_position_ids": torch.cat(position_ids, dim=0).to(device=device, dtype=torch.long),
        "cu_seqlens": F.pad(torch.cumsum(token_lens_tensor, dim=0), (1, 0)).to(torch.int32),
        "max_seqlen": int(token_lens_tensor.max().item()),
        "token_lens": token_lens_tensor,
    }


def scatter_image_embeds(
    image_items: list[ConversationItem],
    image_embeds: torch.Tensor,
    token_lens: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    offset = 0
    lengths = token_lens.detach().cpu().reshape(-1).tolist()
    if len(lengths) != len(image_items):
        raise RuntimeError("BAGEL SigLIP image count mismatch during feature scatter.")
    for item, length in zip(image_items, lengths, strict=True):
        item.value = image_embeds[offset : offset + int(length)].to(device=device, dtype=dtype)
        offset += int(length)
    if offset != int(image_embeds.shape[0]):
        raise RuntimeError("BAGEL SigLIP token count mismatch during feature scatter.")


__all__ = [
    "apply_scale",
    "flattened_position_ids",
    "image_items",
    "is_encoded_image_value",
    "iter_conversation_items",
    "patchify_image",
    "pil_to_rgb_tensor",
    "prepare_image_batch",
    "preprocess_image",
    "resize_pil",
    "resize_tensor",
    "scatter_image_embeds",
    "target_size",
    "to_rgb_pil",
    "to_rgb_tensor",
    "user_raw_image_items",
]
