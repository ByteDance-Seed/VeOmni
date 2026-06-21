"""BAGEL reference generation option normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import float_pair, int_pair, normalize_reference_kwargs


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


__all__ = ["inferencer_generation_kwargs"]
