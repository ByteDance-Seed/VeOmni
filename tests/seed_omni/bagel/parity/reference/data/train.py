"""BAGEL official train-batch and loss helpers for parity reference runs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from tests.seed_omni.parity_suite.core import autocast_for_dtype, make_reference_image, patched_randn_like, to_device

from ..vendor.data.data_utils import (
    get_flattened_position_ids_extrapolate,
    patchify,
    prepare_attention_mask_per_sample,
)
from ..vendor.data.transforms import ImageTransform


_TRAIN_BATCH_KEYS = {
    "sequence_length",
    "sample_lens",
    "packed_text_ids",
    "packed_text_indexes",
    "packed_position_ids",
}

_LATENT_PATCH_SIZE = 2
_LATENT_CHANNELS = 16
_MAX_LATENT_SIZE = 64
_SIGLIP_IMAGE_SIZE = 980
_SIGLIP_MIN_IMAGE_SIZE = 378
_SIGLIP_MAX_PIXELS = 14 * 14 * 9 * 1024
_SIGLIP_PATCH_SIZE = 14
_SIGLIP_MAX_PATCHES_PER_SIDE = 70


@dataclass(frozen=True)
class BagelReferenceTrainLoss:
    """Reduced official training losses used for reference backward."""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor]


def reference_train_batch_from_inputs(
    inputs: Mapping[str, Any],
    *,
    device: torch.device,
    tokenizer: Any | None = None,
    new_token_ids: Mapping[str, int] | None = None,
    loss_mode: str | None = None,
) -> dict[str, Any]:
    """Return an official-style BAGEL training batch on ``device``.

    Explicit ``train_batch`` fixtures are still accepted, but normal Bagel train
    parity recipes use a semantic prompt/loss-mode stimulus and convert it here
    into the packed contract consumed by official ``Bagel.forward``.
    """

    raw_batch = inputs.get("train_batch", inputs.get("official_train_batch"))
    if raw_batch is None:
        raw_batch = build_reference_train_batch_from_stimulus(
            inputs,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            loss_mode=loss_mode,
        )
    if not isinstance(raw_batch, Mapping):
        raise TypeError(f"BAGEL train_batch must be a mapping, got {type(raw_batch).__name__}.")
    missing = sorted(_TRAIN_BATCH_KEYS.difference(raw_batch))
    if missing:
        raise KeyError(f"BAGEL train_batch missing required key(s): {', '.join(missing)}.")
    return dict(to_device(dict(raw_batch), device))


def build_reference_train_batch_from_stimulus(
    inputs: Mapping[str, Any],
    *,
    tokenizer: Any | None,
    new_token_ids: Mapping[str, int] | None = None,
    loss_mode: str | None,
) -> dict[str, Any]:
    """Convert a minimal semantic train stimulus to official packed tensors."""

    if tokenizer is None:
        raise ValueError("BAGEL train stimulus conversion requires the reference tokenizer.")
    mode = str(loss_mode or inputs.get("loss_mode", "ce_mse"))
    if mode not in {"ce", "ce_mse", "text_image_ce"}:
        raise ValueError(
            f"Initial BAGEL train graph parity supports loss_mode='ce', 'ce_mse', or 'text_image_ce', got {mode!r}."
        )

    prompt = _train_prompt_from_stimulus(inputs)
    token_ids = torch.tensor(_train_prompt_token_ids(prompt, tokenizer, new_token_ids), dtype=torch.long)
    if int(token_ids.numel()) < 2:
        raise ValueError("BAGEL CE train parity requires at least two prompt tokens.")

    image_size = int(inputs.get("image_size", 32))
    if image_size % 16 != 0:
        raise ValueError("BAGEL train image_size must be divisible by 16.")
    latent_grid = (image_size // 16, image_size // 16)
    num_latent_tokens = latent_grid[0] * latent_grid[1]
    text_len = int(token_ids.numel())
    include_mse = mode == "ce_mse"
    include_vit = mode == "text_image_ce"
    vit_tokens = _reference_vit_tokens(inputs) if include_vit else None
    vit_len = int(vit_tokens["packed_vit_tokens"].shape[0]) if vit_tokens is not None else 0
    sequence_length = vit_len + text_len + (num_latent_tokens if include_mse else 0)

    packed_text_indexes = torch.arange(vit_len, vit_len + text_len, dtype=torch.long)
    packed_vae_token_indexes = torch.arange(vit_len + text_len, sequence_length, dtype=torch.long)
    ce_loss_indexes = torch.zeros(sequence_length, dtype=torch.bool)
    ce_loss_indexes[packed_text_indexes[:-1]] = True

    split_lens = [text_len, num_latent_tokens] if include_mse else [text_len]
    attn_modes = ["causal", "noise"] if include_mse else ["causal"]
    position_parts = [torch.arange(text_len, dtype=torch.long)]
    if include_vit:
        split_lens = [vit_len, text_len]
        attn_modes = ["full", "causal"]
        position_parts = [
            torch.zeros(vit_len, dtype=torch.long),
            torch.arange(1, text_len + 1, dtype=torch.long),
        ]
    elif include_mse:
        position_parts.append(torch.full((num_latent_tokens,), text_len, dtype=torch.long))

    batch: dict[str, Any] = {
        "sequence_length": sequence_length,
        "sample_lens": [sequence_length],
        "split_lens": split_lens,
        "attn_modes": attn_modes,
        "nested_attention_masks": [prepare_attention_mask_per_sample(split_lens, attn_modes)],
        "packed_text_ids": token_ids,
        "packed_text_indexes": packed_text_indexes,
        "packed_position_ids": torch.cat(position_parts, dim=0),
        "ce_loss_indexes": ce_loss_indexes,
        "packed_label_ids": token_ids[1:].clone(),
        "loss_mode": mode,
        "timestep_shift": 1.0,
    }
    if vit_tokens is not None:
        batch.update(vit_tokens)
    if include_mse:
        mse_loss_indexes = torch.zeros(sequence_length, dtype=torch.bool)
        mse_loss_indexes[packed_vae_token_indexes] = True
        batch.update(
            {
                "padded_images": _normalized_train_image(image_size),
                "patchified_vae_latent_shapes": [latent_grid],
                "packed_latent_position_ids": latent_position_ids(*latent_grid, max_latent_size=_MAX_LATENT_SIZE),
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_timesteps": torch.linspace(-0.5, 0.5, steps=num_latent_tokens, dtype=torch.float32),
                "mse_loss_indexes": mse_loss_indexes,
                "fixed_noise": torch.linspace(
                    -0.25,
                    0.25,
                    steps=num_latent_tokens * _LATENT_PATCH_SIZE * _LATENT_PATCH_SIZE * _LATENT_CHANNELS,
                    dtype=torch.float32,
                ).reshape(num_latent_tokens, _LATENT_PATCH_SIZE * _LATENT_PATCH_SIZE * _LATENT_CHANNELS),
            }
        )
        batch["shifted_timesteps"] = shifted_timesteps(
            batch["packed_timesteps"], timestep_shift=batch["timestep_shift"]
        )
    return batch


def _reference_vit_tokens(inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    image_size = int(inputs.get("image_size", _SIGLIP_IMAGE_SIZE))
    transform = ImageTransform(
        image_size,
        int(inputs.get("min_image_size", _SIGLIP_MIN_IMAGE_SIZE)),
        _SIGLIP_PATCH_SIZE,
        max_pixels=int(inputs.get("max_pixels", _SIGLIP_MAX_PIXELS)),
    )
    image = _train_image_from_stimulus(inputs, image_size=image_size)
    image_tensor = transform(image)
    vit_tokens = patchify(image_tensor, _SIGLIP_PATCH_SIZE)
    vit_position_ids = get_flattened_position_ids_extrapolate(
        image_tensor.size(1),
        image_tensor.size(2),
        _SIGLIP_PATCH_SIZE,
        _SIGLIP_MAX_PATCHES_PER_SIDE,
    )
    return {
        "packed_vit_tokens": vit_tokens,
        "packed_vit_token_indexes": torch.arange(int(vit_tokens.shape[0]), dtype=torch.long),
        "packed_vit_position_ids": vit_position_ids,
        "vit_token_seqlens": torch.tensor([int(vit_tokens.shape[0])], dtype=torch.int32),
    }


def _train_image_from_stimulus(inputs: Mapping[str, Any], *, image_size: int):
    conversation_list = inputs.get("conversation_list")
    if conversation_list is not None:
        for sample in _iter_conversation_samples(conversation_list):
            for item in sample:
                if not isinstance(item, Mapping) or str(item.get("type")) != "image":
                    continue
                value = item.get("value")
                if isinstance(value, Mapping) and str(value.get("kind")) == "image":
                    return make_reference_image(
                        int(value.get("width", image_size)), int(value.get("height", image_size))
                    )
    return make_reference_image(image_size, image_size)


def _train_prompt_token_ids(
    prompt: str,
    tokenizer: Any,
    new_token_ids: Mapping[str, int] | None,
) -> list[int]:
    body = list(tokenizer.encode(prompt))
    start_token_id = _special_token_id(
        tokenizer,
        new_token_ids,
        key="bos_token_id",
        token="<|im_start|>",
        fallback=getattr(tokenizer, "eos_token_id", None),
    )
    eos_token_id = _special_token_id(
        tokenizer,
        new_token_ids,
        key="eos_token_id",
        token="<|im_end|>",
        fallback=getattr(tokenizer, "eos_token_id", None),
    )
    return [start_token_id, *body, eos_token_id]


def _special_token_id(
    tokenizer: Any,
    new_token_ids: Mapping[str, int] | None,
    *,
    key: str,
    token: str,
    fallback: int | None,
) -> int:
    if new_token_ids is not None and key in new_token_ids:
        return int(new_token_ids[key])
    resolved = tokenizer.convert_tokens_to_ids(token) if hasattr(tokenizer, "convert_tokens_to_ids") else None
    if resolved is not None:
        unk = getattr(tokenizer, "unk_token_id", None)
        if unk is None or int(resolved) != int(unk):
            return int(resolved)
    if fallback is None:
        raise ValueError(f"BAGEL train stimulus conversion cannot resolve token {token!r}.")
    return int(fallback)


def _train_prompt_from_stimulus(inputs: Mapping[str, Any]) -> str:
    prompt = inputs.get("prompt")
    if prompt is not None:
        return str(prompt)

    conversation_list = inputs.get("conversation_list")
    if conversation_list is None:
        return "Describe BAGEL in one short sentence."
    for sample in _iter_conversation_samples(conversation_list):
        for item in sample:
            text = _text_from_conversation_spec(item)
            if text is not None:
                return text
    raise ValueError("BAGEL train stimulus conversation_list must contain at least one text item.")


def _iter_conversation_samples(conversation_list: Any) -> Iterator[Any]:
    if not isinstance(conversation_list, list):
        raise TypeError("BAGEL train stimulus conversation_list must be a list.")
    if not conversation_list:
        return
    first = conversation_list[0]
    if isinstance(first, Mapping):
        yield conversation_list
        return
    yield from conversation_list


def _text_from_conversation_spec(item: Any) -> str | None:
    if not isinstance(item, Mapping) or str(item.get("type", "text")) != "text":
        return None
    value = item.get("value")
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return None
    if str(value.get("kind")) == "text":
        return str(value.get("text", ""))
    return None


def _normalized_train_image(image_size: int) -> torch.Tensor:
    image = make_reference_image(image_size, image_size)
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).to(dtype=torch.float32)
    tensor = tensor.div(255.0).sub(0.5).div(0.5)
    return tensor.unsqueeze(0)


def encode_reference_vae_latents(
    *,
    vae_model: nn.Module | None,
    batch: Mapping[str, Any],
    target_device: torch.device | None = None,
    target_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Apply official training VAE encode semantics to ``batch``."""

    encoded = dict(batch)
    padded_images = encoded.pop("padded_images", None)
    if padded_images is None:
        return encoded
    if vae_model is None:
        raise ValueError("BAGEL train batch contains padded_images but the reference VAE is not loaded.")
    vae_model.eval()
    vae_param = next(vae_model.parameters(), None)
    vae_device = vae_param.device if vae_param is not None else padded_images.device
    vae_dtype = vae_param.dtype if vae_param is not None and vae_param.dtype.is_floating_point else torch.float32
    source_device = padded_images.device
    output_device = target_device or source_device
    output_dtype = target_dtype or padded_images.dtype
    with torch.no_grad():
        padded_latent = vae_model.encode(padded_images.to(device=vae_device, dtype=vae_dtype))
    if isinstance(padded_latent, Mapping):
        padded_latent = padded_latent.get("latents", padded_latent.get("padded_latent"))
    if not torch.is_tensor(padded_latent):
        raise TypeError(f"BAGEL VAE encode must return a tensor, got {type(padded_latent).__name__}.")
    encoded["padded_latent"] = padded_latent.to(device=output_device, dtype=output_dtype)
    return encoded


def official_train_forward_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Strip parity-only controls before calling official ``Bagel.forward``."""

    excluded = {"fixed_noise", "shifted_timesteps", "loss_mode", "timestep_shift"}
    return {key: value for key, value in batch.items() if key not in excluded}


def latent_position_ids(height: int, width: int, *, max_latent_size: int = _MAX_LATENT_SIZE) -> torch.Tensor:
    rows = torch.arange(int(height), dtype=torch.long)[:, None] * int(max_latent_size)
    cols = torch.arange(int(width), dtype=torch.long)[None]
    return (rows + cols).flatten()


def shifted_timesteps(timesteps: torch.Tensor, *, timestep_shift: float) -> torch.Tensor:
    values = torch.sigmoid(timesteps.to(dtype=torch.float32))
    shift = float(timestep_shift)
    return shift * values / (1.0 + (shift - 1.0) * values)


def reduce_reference_train_losses(
    loss_dict: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    ce_weight: float = 1.0,
    mse_weight: float = 1.0,
    ce_loss_reweighting: bool = False,
) -> BagelReferenceTrainLoss:
    """Reduce official ``Bagel.forward`` raw losses like ``train.py``."""

    device = _loss_device(loss_dict)
    loss = torch.zeros((), device=device, dtype=torch.float32)
    losses: dict[str, torch.Tensor] = {}

    ce = loss_dict.get("ce")
    if torch.is_tensor(ce):
        if ce_loss_reweighting:
            weights = batch.get("ce_loss_weights")
            if not torch.is_tensor(weights):
                raise ValueError("BAGEL CE loss reweighting requires ce_loss_weights.")
            ce_reduced = (ce * weights.to(device=ce.device, dtype=ce.dtype)).sum() / weights.sum().to(
                device=ce.device, dtype=ce.dtype
            )
        else:
            total_ce_tokens = _num_index_tokens(batch, "ce_loss_indexes", default=ce.numel(), device=ce.device)
            ce_reduced = ce.sum() / total_ce_tokens.to(device=ce.device, dtype=ce.dtype)
        losses["ce"] = ce_reduced.detach()
        loss = loss + ce_reduced.to(device=loss.device, dtype=loss.dtype) * float(ce_weight)
    else:
        losses["ce"] = torch.zeros((), device=device, dtype=torch.float32)

    mse = loss_dict.get("mse")
    if torch.is_tensor(mse):
        total_mse_tokens = _num_index_tokens(batch, "mse_loss_indexes", default=mse.shape[0], device=mse.device)
        mse_reduced = mse.mean(dim=-1).sum() / total_mse_tokens.to(device=mse.device, dtype=mse.dtype)
        losses["mse"] = mse_reduced.detach()
        loss = loss + mse_reduced.to(device=loss.device, dtype=loss.dtype) * float(mse_weight)
    else:
        losses["mse"] = torch.zeros((), device=device, dtype=torch.float32)

    losses["loss"] = loss.detach()
    return BagelReferenceTrainLoss(loss=loss, losses=losses)


def train_options_from_inputs(
    inputs: Mapping[str, Any],
    *,
    batch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge explicit train options with options inferred from the packed batch."""

    raw = inputs.get("train_kwargs", inputs.get("training_kwargs", {})) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"BAGEL train kwargs must be a mapping, got {type(raw).__name__}.")
    options = {
        "ce_weight": float(raw.get("ce_weight", 1.0)),
        "mse_weight": float(raw.get("mse_weight", 1.0)),
        "ce_loss_reweighting": bool(raw.get("ce_loss_reweighting", False)),
    }
    if batch is not None:
        options.update(
            {
                "visual_und": "packed_vit_tokens" in batch,
                "visual_gen": "padded_latent" in batch,
                "fixed_noise": torch.is_tensor(batch.get("fixed_noise")),
            }
        )
    return options


def train_loss_options(train_options: Mapping[str, Any]) -> dict[str, Any]:
    """Return the subset of train options consumed by loss reduction."""

    return {
        "ce_weight": float(train_options.get("ce_weight", 1.0)),
        "mse_weight": float(train_options.get("mse_weight", 1.0)),
        "ce_loss_reweighting": bool(train_options.get("ce_loss_reweighting", False)),
    }


@contextmanager
def reference_train_forward_context(
    model: Any,
    batch: Mapping[str, Any],
    train_options: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Iterator[None]:
    """Apply BAGEL train forward options for one official ``Bagel.forward`` call."""

    original_visual_und = bool(getattr(model.config, "visual_und", False))
    original_visual_gen = bool(getattr(model.config, "visual_gen", False))
    model.config.visual_und = bool(train_options.get("visual_und", False))
    model.config.visual_gen = bool(train_options.get("visual_gen", False))
    fixed_noise = batch.get("fixed_noise")
    noise_context = patched_randn_like(fixed_noise) if train_options.get("fixed_noise") else nullcontext()
    try:
        with autocast_for_dtype(device, dtype), noise_context:
            yield
    finally:
        model.config.visual_und = original_visual_und
        model.config.visual_gen = original_visual_gen


def _loss_device(loss_dict: Mapping[str, Any]) -> torch.device:
    for value in loss_dict.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def _num_index_tokens(
    batch: Mapping[str, Any],
    key: str,
    *,
    default: int,
    device: torch.device,
) -> torch.Tensor:
    indexes = batch.get(key)
    if torch.is_tensor(indexes):
        count = int(indexes.sum().item()) if indexes.dtype == torch.bool else indexes.numel()
    elif indexes is None:
        count = int(default)
    else:
        count = len(indexes)
    if count <= 0:
        raise ValueError(f"BAGEL {key} must contain at least one token when its loss is present.")
    return torch.tensor(float(count), device=device)


__all__ = [
    "BagelReferenceTrainLoss",
    "build_reference_train_batch_from_stimulus",
    "encode_reference_vae_latents",
    "latent_position_ids",
    "official_train_forward_batch",
    "reduce_reference_train_losses",
    "reference_train_batch_from_inputs",
    "reference_train_forward_context",
    "shifted_timesteps",
    "train_loss_options",
    "train_options_from_inputs",
]
