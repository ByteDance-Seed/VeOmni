"""Training-time image tensorisation for SeedOmni V2 vision modules.

The raw training ``conversation_list`` carries un-normalised ``(C, H, W)``
``uint8`` image tensors (see :mod:`veomni.data.multimodal.seedomni_transform`).
The vision encoder modules (``JanusSiglip`` understanding tower,
``JanusVqvae`` generation codec) own image *processing* in V2, so they call
:func:`build_pixel_values_batch` inside their ``pre_forward`` to turn the raw
pixels into the normalised ``(B, 3, H, W)`` float batch their forward expects.

Why zeros for absent samples?
-----------------------------
Every active training node must forward on every micro-batch to keep the
FSDP DP/SP graphs aligned (seedomni-v2 invariant 10).  A sample with no
image of the module's modality (e.g. a text-only or generation sample fed to
the understanding tower) contributes a zero placeholder image so the batch
dimension is preserved; the backbone's ``masked_scatter`` then ignores that
row (its placeholder mask is all-False) and the per-module grad-sync anchor
keeps the (zero) gradient flowing.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image


def _uint8_chw_to_pil(image: torch.Tensor) -> Image.Image:
    """``(C, H, W)`` uint8 tensor → RGB :class:`PIL.Image`."""
    if image.dim() != 3:
        raise ValueError(f"expected a (C, H, W) image tensor, got shape {tuple(image.shape)}.")
    arr = image.detach().to("cpu", torch.uint8).permute(1, 2, 0).numpy()  # (H, W, C)
    return Image.fromarray(arr).convert("RGB")


def build_pixel_values_batch(
    per_sample_values: list[list[Any]],
    *,
    processor: Any | None,
    image_size: int,
    num_channels: int,
    device: Any,
    dtype: Any,
) -> torch.Tensor:
    """Tensorise per-sample raw images into a normalised ``(B, 3, H, W)`` batch.

    Args:
        per_sample_values: one list per sample of raw ``(C, H, W)`` uint8
            image tensors (output of
            :func:`veomni.models.seed_omni.conversation.collect_modality_values`).
            At most one image per sample is supported today; samples with
            none get a zero placeholder.
        processor: the module's :class:`~transformers.JanusImageProcessor`
            (resize + normalise).  Required when any sample actually carries
            an image.
        image_size / num_channels: zero-placeholder shape for image-free
            samples.
        device / dtype: target device / dtype of the returned batch.

    Returns:
        ``(B, num_channels, image_size, image_size)`` float tensor.
    """
    batch_size = len(per_sample_values)
    zero = torch.zeros(num_channels, image_size, image_size)

    present_idx: list[int] = []
    present_pil: list[Image.Image] = []
    for i, values in enumerate(per_sample_values):
        if not values:
            continue
        if len(values) > 1:
            raise NotImplementedError(
                "SeedOmni V2 training image extraction currently supports at most one image of a "
                f"given modality per sample; sample {i} has {len(values)}. Multi-image-per-sample "
                "support is a follow-up (needs per-sample placeholder-count routing)."
            )
        present_idx.append(i)
        present_pil.append(_uint8_chw_to_pil(values[0]))

    rows: list[torch.Tensor] = [zero] * batch_size
    if present_pil:
        if processor is None:
            raise RuntimeError(
                "build_pixel_values_batch: samples carry images but the module has no image "
                "processor. OmniTrainer must load the module's `processor_class` and assign it to "
                "`module._processor` before training (see OmniTrainer._build_model_assets)."
            )
        processed = processor(images=present_pil, return_tensors="pt")["pixel_values"]  # (n, C, H, W)
        for slot, i in enumerate(present_idx):
            rows[i] = processed[slot]

    return torch.stack(rows, dim=0).to(device=device, dtype=dtype)
