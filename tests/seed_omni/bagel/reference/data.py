"""BAGEL reference-side data helpers for parity tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
from PIL import Image
from torch import nn

from tests.seed_omni.parity_suite.core import to_device
from veomni.models.seed_omni.conversation import ConversationItem, is_dummy


T = TypeVar("T")


_TRAIN_BATCH_KEYS = {
    "sequence_length",
    "sample_lens",
    "packed_text_ids",
    "packed_text_indexes",
    "packed_position_ids",
}


@dataclass(frozen=True)
class BagelReferenceTrainLoss:
    """Reduced official training losses used for reference backward."""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor]


# Inference inputs -------------------------------------------------------------


def conversation_item_text(
    item: ConversationItem,
    tokenizer: Any,
    *,
    input_ids_meta_key: str = "input_ids",
    skip_special_tokens: bool = True,
) -> str:
    """Convert a text conversation item back to a string for official BAGEL."""

    if isinstance(item.value, str):
        return item.value
    input_ids = item.meta.get(input_ids_meta_key, item.value)
    if torch.is_tensor(input_ids):
        return tokenizer.decode(
            input_ids.reshape(-1).detach().cpu().tolist(),
            skip_special_tokens=skip_special_tokens,
        )
    raise TypeError(f"BAGEL reference text input must be str or token tensor, got {type(item.value).__name__}.")


def conversation_item_image(item: ConversationItem) -> Image.Image:
    """Convert an image conversation item back to a PIL image for official BAGEL."""

    if isinstance(item.value, Image.Image):
        return item.value
    raise TypeError(f"BAGEL reference image input must be a PIL image, got {type(item.value).__name__}.")


def conversation_to_interleaved_reference_inputs(
    conversation: Iterable[ConversationItem],
    *,
    tokenizer: Any,
    skip_dummy: bool = True,
    input_ids_meta_key: str = "input_ids",
    skip_special_tokens: bool = True,
) -> list[str | Image.Image]:
    """Convert one V2 conversation sample to official interleaved inputs."""

    inputs: list[str | Image.Image] = []
    for item in conversation:
        if skip_dummy and is_dummy(item):
            continue
        if item.type == "text":
            inputs.append(
                conversation_item_text(
                    item,
                    tokenizer,
                    input_ids_meta_key=input_ids_meta_key,
                    skip_special_tokens=skip_special_tokens,
                )
            )
        elif item.type == "image":
            inputs.append(conversation_item_image(item))
        else:
            raise ValueError(f"BAGEL interleaved input does not support conversation item type {item.type!r}.")
    return inputs


def normalize_reference_kwargs(
    value: Any,
    *,
    alias_fields: Mapping[str, tuple[Callable[[Any], Any], tuple[str, ...]]] | None = None,
    direct_fields: Mapping[str, Callable[[Any], Any]] | None = None,
    pair_fields: Mapping[str, Callable[[Any], Any]] | None = None,
    error_prefix: str = "BAGEL reference kwargs",
) -> dict[str, Any]:
    """Normalize authored kwargs into official BAGEL reference kwargs."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{error_prefix} must be a mapping.")
    kwargs: dict[str, Any] = {}

    for target_key, (caster, source_keys) in (alias_fields or {}).items():
        for source_key in source_keys:
            if source_key in value:
                kwargs[target_key] = caster(value[source_key])
                break
    for key, caster in (direct_fields or {}).items():
        if key in value:
            kwargs[key] = caster(value[key])
    for key, caster in (pair_fields or {}).items():
        if key in value:
            kwargs[key] = caster(value[key])
    return kwargs


def int_pair(value: Any, *, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-2 sequence.")
    return int(value[0]), int(value[1])


def float_pair(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-2 sequence.")
    return float(value[0]), float(value[1])


def first_output_of_type(outputs: Iterable[Any], expected_type: type[T]) -> T | None:
    return next((item for item in outputs if isinstance(item, expected_type)), None)


_ALIAS_GENERATION_FIELDS = {
    "max_think_token_n": (int, ("max_think_token_n", "max_length", "max_new_tokens")),
    "text_temperature": (float, ("text_temperature", "temperature")),
}

_DIRECT_GENERATION_FIELDS = {
    "cfg_img_scale": float,
    "cfg_renorm_min": float,
    "cfg_renorm_type": str,
    "cfg_text_scale": float,
    "do_sample": bool,
    "enable_taylorseer": bool,
    "num_timesteps": int,
    "think": bool,
    "timestep_shift": float,
}


def inferencer_generation_kwargs(value: Any) -> dict[str, Any]:
    """Normalize stimulus generation kwargs for official ``InterleaveInferencer``."""

    kwargs = normalize_reference_kwargs(
        value,
        alias_fields=_ALIAS_GENERATION_FIELDS,
        direct_fields=_DIRECT_GENERATION_FIELDS,
        pair_fields={"cfg_interval": lambda item: float_pair(item, name="cfg_interval")},
        error_prefix="BAGEL inferencer generation_kwargs",
    )
    if value is None:
        return kwargs
    if not isinstance(value, Mapping):
        return kwargs
    image_shapes = _inferencer_image_shapes(value)
    if image_shapes is not None:
        kwargs["image_shapes"] = image_shapes
    return kwargs


def _inferencer_image_shapes(value: Mapping[str, Any]) -> tuple[int, int] | None:
    if "image_shapes" in value:
        return int_pair(value["image_shapes"], name="image_shapes")
    if "image_shape" in value:
        return int_pair(value["image_shape"], name="image_shape")
    height = value.get("image_height", value.get("height"))
    width = value.get("image_width", value.get("width"))
    if height is None and width is None:
        return None
    if height is None or width is None:
        raise ValueError("BAGEL inferencer image shape requires both height and width.")
    return int(height), int(width)


# Training inputs --------------------------------------------------------------


def reference_train_batch_from_inputs(inputs: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
    """Return an official-style BAGEL training batch on ``device``.

    The first train parity cases can pass an explicit ``train_batch`` while the
    conversation-to-``sequence_plan`` adapter is built out. Keeping this boundary
    centralized lets later recipes add semantic conversion without changing the
    reference forward/backward entrypoint.
    """

    raw_batch = inputs.get("train_batch", inputs.get("official_train_batch"))
    if raw_batch is None:
        raise NotImplementedError(
            "BAGEL train reference currently expects stimulus.train_batch or "
            "stimulus.official_train_batch. Conversation-to-official training "
            "batch conversion should be added here."
        )
    if not isinstance(raw_batch, Mapping):
        raise TypeError(f"BAGEL train_batch must be a mapping, got {type(raw_batch).__name__}.")
    missing = sorted(_TRAIN_BATCH_KEYS.difference(raw_batch))
    if missing:
        raise KeyError(f"BAGEL train_batch missing required key(s): {', '.join(missing)}.")
    return dict(to_device(dict(raw_batch), device))


def encode_reference_vae_latents(
    *,
    vae_model: nn.Module | None,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply official training VAE encode semantics to ``batch``."""

    encoded = dict(batch)
    padded_images = encoded.pop("padded_images", None)
    if padded_images is None:
        return encoded
    if vae_model is None:
        raise ValueError("BAGEL train batch contains padded_images but the reference VAE is not loaded.")
    vae_model.eval()
    with torch.no_grad():
        padded_latent = vae_model.encode(padded_images)
    if isinstance(padded_latent, Mapping):
        padded_latent = padded_latent.get("latents", padded_latent.get("padded_latent"))
    if not torch.is_tensor(padded_latent):
        raise TypeError(f"BAGEL VAE encode must return a tensor, got {type(padded_latent).__name__}.")
    encoded["padded_latent"] = padded_latent
    return encoded


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


def train_options_from_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Extract official train-loop scalar options from stimulus inputs."""

    raw = inputs.get("train_kwargs", inputs.get("training_kwargs", {})) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"BAGEL train kwargs must be a mapping, got {type(raw).__name__}.")
    return {
        "ce_weight": float(raw.get("ce_weight", 1.0)),
        "mse_weight": float(raw.get("mse_weight", 1.0)),
        "ce_loss_reweighting": bool(raw.get("ce_loss_reweighting", False)),
    }


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
        count = indexes.numel()
    elif indexes is None:
        count = int(default)
    else:
        count = len(indexes)
    if count <= 0:
        raise ValueError(f"BAGEL {key} must contain at least one token when its loss is present.")
    return torch.tensor(float(count), device=device)


__all__ = [
    "BagelReferenceTrainLoss",
    "conversation_item_image",
    "conversation_item_text",
    "conversation_to_interleaved_reference_inputs",
    "encode_reference_vae_latents",
    "first_output_of_type",
    "float_pair",
    "inferencer_generation_kwargs",
    "int_pair",
    "normalize_reference_kwargs",
    "reduce_reference_train_losses",
    "reference_train_batch_from_inputs",
    "train_options_from_inputs",
]
