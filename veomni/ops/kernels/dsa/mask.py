# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing limitations
# under the License.

"""Translate HF dense masks before fused GLM/DeepSeek-V4 DSA kernels.

Fused indexer/attention rows bake in standard causality and cannot represent
padding or a custom additive mask. Modeling must drop a pure causal mask to
``None`` and refuse anything else, pointing the caller at eager.
"""

from __future__ import annotations

import torch
from torch import Tensor


STANDARD_CAUSAL = "standard_causal"
CUSTOM = "custom"
_PROVENANCE_ATTR = "_veomni_dsa_mask_provenance"


def mark_standard_causal_mask(mask: Tensor | None) -> Tensor | None:
    """Record that ``mask`` is the shape-API standard causal triangle."""
    if mask is not None:
        setattr(mask, _PROVENANCE_ATTR, STANDARD_CAUSAL)
    return mask


def mark_custom_dsa_mask(mask: Tensor) -> Tensor:
    """Record that ``mask`` is caller-declared custom or unknown-unsafe."""
    setattr(mask, _PROVENANCE_ATTR, CUSTOM)
    return mask


def dsa_mask_provenance(mask: Tensor | None) -> str | None:
    """Return the attached provenance, or ``None`` when the mask is unmarked.

    ``None`` itself is the fused ``is_causal`` skip and counts as standard.
    """
    if mask is None:
        return STANDARD_CAUSAL
    provenance = getattr(mask, _PROVENANCE_ATTR, None)
    return provenance if isinstance(provenance, str) else None


def copy_dsa_mask_provenance(source: Tensor | None, dest: Tensor | None) -> Tensor | None:
    """Re-attach provenance after a slice or unsqueeze that drops tensor attrs."""
    if dest is None:
        return None
    provenance = dsa_mask_provenance(source) if source is not None else None
    if provenance == STANDARD_CAUSAL:
        return mark_standard_causal_mask(dest)
    if provenance == CUSTOM:
        return mark_custom_dsa_mask(dest)
    return dest


def _position_ids_break_standard_causal(position_ids: Tensor | None) -> bool:
    """Whether any row is not a unit-increment sequence, matching HF packed detection.

    HF treats ``diff != 1`` as a sample boundary. A reset (``<= 0``) and a gap
    such as ``[[0, 1, 4, 5]]`` are both packed, not a standard causal triangle.
    """
    if position_ids is None or not torch.is_tensor(position_ids) or position_ids.numel() <= 1:
        return False
    ids = position_ids if position_ids.dim() > 1 else position_ids.unsqueeze(0)
    if ids.shape[-1] <= 1:
        return False
    return bool((ids[..., 1:] - ids[..., :-1] != 1).any())


_CREATE_CAUSAL_MASK_POSITIONALS = (
    "config",
    "inputs_embeds",
    "attention_mask",
    "past_key_values",
    "position_ids",
    "or_mask_function",
    "and_mask_function",
    "block_sequence_ids",
)


def _bound_create_causal_mask_args(*args: object, **kwargs: object) -> dict[str, object]:
    """Map HF positional ``create_causal_mask`` args onto the keyword names."""
    bound = dict(kwargs)
    for name, value in zip(_CREATE_CAUSAL_MASK_POSITIONALS, args, strict=False):
        bound.setdefault(name, value)
    return bound


def _has_mask_overlay(bound: dict[str, object]) -> bool:
    """HF extras that overlay a non-standard triangle onto the causal mask."""
    return any(bound.get(name) is not None for name in ("or_mask_function", "and_mask_function", "block_sequence_ids"))


def _breaks_standard_causal(bound: dict[str, object]) -> bool:
    """Whether HF would build anything other than a plain causal triangle."""
    config = bound.get("config")
    if config is not None and not getattr(config, "is_causal", True):
        return True
    if _has_mask_overlay(bound):
        return True
    position_ids = bound.get("position_ids")
    return _position_ids_break_standard_causal(position_ids if isinstance(position_ids, Tensor) else None)


def create_standard_causal_mask(*args, **kwargs) -> Tensor | None:
    """HF ``create_causal_mask``, marked only for a no-padding standard triangle.

    Packed ``position_ids``, overlay functions passed by name or position,
    ``config.is_causal=False``, and HF overlay kwargs are marked custom so
    fused translate can reject them without scanning the 4-D mask. A 2-D
    padding mask stays unmarked and still goes through ``is_standard_causal_mask``.
    """
    from transformers.masking_utils import create_causal_mask

    bound = _bound_create_causal_mask_args(*args, **kwargs)
    if _breaks_standard_causal(bound):
        # Bidirectional skip is ``None``, the same sentinel fused DSA uses for
        # standard causal. Force a materialized mask and mark it custom.
        mask = create_causal_mask(*args, **{**kwargs, "allow_is_causal_skip": False})
        if mask is None:
            raise ValueError("non-standard causal mask must materialize so fused DSA can reject it")
        return mark_custom_dsa_mask(mask)
    mask = create_causal_mask(*args, **kwargs)
    if bound.get("attention_mask") is None:
        return mark_standard_causal_mask(mask)
    return mask


def is_standard_causal_mask(attention_mask: Tensor | None, *, q_len: int, kv_len: int) -> bool:
    """Whether ``attention_mask`` is only the lower-triangular causal constraint.

    ``None`` is the ``is_causal`` skip used when there is no padding. Additive
    or boolean ``[B, 1, Q, K]`` / ``[B, Q, K]`` masks must match the
    decode-aware triangle ``k > q + kv_len - q_len`` on every entry.
    ``H != 1``, padding, a custom overlay, a nonzero additive bias on an
    allowed position, or a finite negative other than ``finfo.min`` is not
    standard. Hard blocks are only ``-inf`` and the dtype minimum.
    """
    if attention_mask is None:
        return True
    mask = attention_mask
    if mask.dim() == 4:
        if mask.shape[1] != 1:
            return False
        mask = mask[:, 0]
    if mask.dim() != 3 or mask.shape[-2] != q_len or mask.shape[-1] != kv_len:
        return False
    q_positions = torch.arange(q_len, device=mask.device)[:, None]
    k_positions = torch.arange(kv_len, device=mask.device)[None, :]
    expected_blocked = (k_positions > q_positions + kv_len - q_len).unsqueeze(0).expand_as(mask)
    if mask.dtype == torch.bool:
        # Accept either convention only when the full triangle matches.
        return bool(torch.equal(mask, ~expected_blocked) or torch.equal(mask, expected_blocked))
    allowed = mask == 0
    blocked = torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min)
    return bool(torch.equal(blocked, expected_blocked) and torch.equal(allowed, ~expected_blocked))


def translate_fused_dsa_mask(
    attention_mask: Tensor | None,
    *,
    q_len: int,
    kv_len: int,
    fused: bool,
    what: str,
) -> Tensor | None:
    """Drop a standard causal mask for fused DSA, or keep it for eager.

    Fused rows must not see padding or custom masks. Those cases raise and tell
    the caller to use the eager implementation instead of ignoring the mask.
    A marked standard-causal mask is dropped without scanning the tensor.
    Unmarked masks still go through ``is_standard_causal_mask``.
    """
    if not fused:
        return attention_mask
    provenance = dsa_mask_provenance(attention_mask)
    if provenance == STANDARD_CAUSAL:
        return None
    if provenance == CUSTOM:
        raise ValueError(
            f"{what} fused implementation only accepts the standard causal mask; "
            "padding or custom masks require the eager implementation."
        )
    if is_standard_causal_mask(attention_mask, q_len=q_len, kv_len=kv_len):
        return None
    raise ValueError(
        f"{what} fused implementation only accepts the standard causal mask; "
        "padding or custom masks require the eager implementation."
    )
