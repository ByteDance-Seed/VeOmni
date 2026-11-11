# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Causal and sliding-window mask helpers with packed-sequence support."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from transformers.cache_utils import Cache
from transformers.configuration_utils import PreTrainedConfig
from transformers.masking_utils import (
    bidirectional_mask_function,
    blockwise_overlay,
    or_masks,
)
from transformers.masking_utils import (
    create_causal_mask as _hf_create_causal_mask,
)
from transformers.masking_utils import (
    create_sliding_window_causal_mask as _hf_create_sliding_window_causal_mask,
)

from veomni.ops.kernels.attention.ulysses import effective_sequence_lengths
from veomni.ops.mask import (
    MagiAttentionMask,
    packed_causal_mask,
    sliding_window_mask,
)


class _AttnImplView:
    """Expose one attention impl for HF mask dispatch without mutating the config."""

    def __init__(self, config: PreTrainedConfig, impl: str, sliding_window: int | None = None) -> None:
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_impl", impl)
        object.__setattr__(self, "_sliding_window", sliding_window)

    def __getattr__(self, name: str) -> Any:
        if name == "sliding_window" and self._sliding_window is not None:
            return self._sliding_window
        return getattr(self._config, name)

    @property
    def _attn_implementation(self) -> str:
        return self._impl


def _resolve_attn_implementation(config: PreTrainedConfig, attn_implementation: str | None = None) -> str:
    """Resolve the active attention implementation from ``config``."""
    if isinstance(attn_implementation, str) and attn_implementation:
        return attn_implementation
    impl = getattr(config, "_attn_implementation", None)
    if isinstance(impl, str) and impl:
        return impl
    ops_config = getattr(config, "ops_config", None)
    if ops_config is not None:
        impl = getattr(ops_config, "attn_implementation", None)
        if isinstance(impl, str) and impl:
            return impl
    return "eager"


def _extract_cu_seqlens(kwargs: dict[str, Any]) -> tuple[Tensor | None, Tensor | None]:
    """Extract packed query/key cumulative sequence lengths from ``kwargs``."""
    cu_q = None
    for key in ("cu_seq_lens_q", "cu_seqlens_q", "cu_seqlens"):
        value = kwargs.get(key)
        if torch.is_tensor(value) and value.numel() >= 2:
            cu_q = value
            break
    if cu_q is None:
        return None, None

    cu_k = None
    for key in ("cu_seq_lens_k", "cu_seqlens_k"):
        value = kwargs.get(key)
        if torch.is_tensor(value) and value.numel() >= 2:
            cu_k = value
            break
    return cu_q, cu_k


def _should_use_packed_mask(cu_seqlens: Tensor | None, attn_impl: str, batch_size: int = 1) -> bool:
    """Return whether ``cu_seqlens`` should route through ``veomni.ops.mask``."""
    if cu_seqlens is None:
        return False
    canonical = attn_impl.removeprefix("veomni_")
    if canonical == "sage_attention":
        return False
    if canonical in ("flex_attention", "magi_attention"):
        return True
    # ``cu_seqlens`` is batch-flattened: one segment per row is not packing.
    return cu_seqlens.numel() - 1 > batch_size


def _resolve_sliding_window(config: PreTrainedConfig, sliding_window: int | None = None) -> int:
    """Resolve and cross-check ``sliding_window`` against ``config.sliding_window``."""
    config_window = getattr(config, "sliding_window", None)
    if sliding_window is not None and config_window is not None and sliding_window != config_window:
        raise ValueError(f"sliding_window={sliding_window} disagrees with config.sliding_window={config_window}.")
    resolved = sliding_window if sliding_window is not None else config_window
    if resolved is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")
    return resolved


def _is_prebuilt_mask(attention_mask: Tensor | BlockMask | MagiAttentionMask | None) -> bool:
    """Return True when ``attention_mask`` is already a 4D mask or backend mask object."""
    if isinstance(attention_mask, (BlockMask, MagiAttentionMask)):
        return True
    return isinstance(attention_mask, Tensor) and attention_mask.ndim == 4


def _resolve_skip_ulysses(attn_impl: str, extra_kwargs: dict[str, Any]) -> bool:
    """Return whether mask construction should keep rank-local sequence lengths."""
    canonical = attn_impl.removeprefix("veomni_")
    return bool(extra_kwargs.get("skip_ulysses", False)) or canonical == "eager"


def _validate_existing_4d_mask(existing_4d: Tensor, q_len: int, attn_impl: str, extra_kwargs: dict[str, Any]) -> None:
    """Validate that a pre-built 3D/4D mask is supported by the backend and matches the post-Ulysses sequence length."""
    canonical = attn_impl.removeprefix("veomni_")
    if canonical not in ("eager", "sdpa"):
        raise ValueError(
            f"packed {attn_impl} attention does not support a pre-built 3D/4D tensor attention_mask; "
            "pass a 2D padding mask or a backend mask object instead"
        )
    eff_q_len, _ = effective_sequence_lengths(
        q_len,
        q_len,
        skip_ulysses=_resolve_skip_ulysses(attn_impl, extra_kwargs),
    )
    if existing_4d.shape[-2] != eff_q_len:
        raise ValueError(
            "packed SDPA/eager attention requires the existing mask query length "
            f"to match hidden_states, got {existing_4d.shape[-2]} and {eff_q_len}"
        )
    if existing_4d.shape[-1] != eff_q_len:
        raise ValueError(
            "packed SDPA/eager attention does not support cached sequences "
            f"(q_len={eff_q_len}, kv_len={existing_4d.shape[-1]}); use a packed-capable impl or disable cache"
        )


def _merge_dense_attention_masks(existing: Tensor, packed: Tensor) -> Tensor:
    """Block packed-forbidden positions; keep existing values on the rest.

    Packed isolation is a visibility overlay. Allowed positions retain padding
    and additive bias from ``existing`` instead of being clipped to zero.
    """
    packed_view = packed
    while packed_view.ndim < existing.ndim:
        packed_view = packed_view.unsqueeze(1)
    while packed_view.ndim > existing.ndim:
        if packed_view.shape[1] != 1:
            raise ValueError(
                f"cannot align packed mask {tuple(packed_view.shape)} with existing {tuple(existing.shape)}"
            )
        packed_view = packed_view.squeeze(1)
    if packed_view.shape[-2:] != existing.shape[-2:]:
        raise ValueError(
            f"packed mask q/kv {tuple(packed_view.shape[-2:])} does not match existing {tuple(existing.shape[-2:])}"
        )
    packed_view = packed_view.expand_as(existing)
    packed_keep = packed_view if packed_view.dtype == torch.bool else packed_view >= 0
    packed_keep = packed_keep.to(dtype=torch.bool)
    if existing.dtype == torch.bool:
        return existing & packed_keep
    # A finite sentinel would win over -inf when the query's own sample is
    # fully masked, allowing attention (and gradients) into another sample.
    return existing.masked_fill(~packed_keep, float("-inf"))


def _finalize_packed_mask(
    mask: Tensor | BlockMask | MagiAttentionMask | None,
    *,
    builder: Callable[..., Tensor | BlockMask | MagiAttentionMask | None],
    q_len: int,
    kv_len: int,
    cu_seqlens: Tensor,
    packed_kwargs: dict[str, Any],
    existing_4d_mask: Tensor | None,
    sliding_window: int | None = None,
) -> Tensor | BlockMask | MagiAttentionMask | None:
    """Merge existing 4D masks and enforce ``-inf`` cross-sample isolation on eager masks."""
    if not isinstance(mask, Tensor):
        if existing_4d_mask is not None:
            raise ValueError(
                f"packed {packed_kwargs['impl']} attention does not support a pre-built 3D/4D tensor attention_mask; "
                "pass a 2D padding mask or a backend mask object instead"
            )
        return mask

    attn_impl = packed_kwargs["impl"]
    canonical = attn_impl.removeprefix("veomni_")
    if existing_4d_mask is not None:
        if canonical == "eager" and existing_4d_mask.dtype == torch.bool:
            from veomni.ops.kernels.attention.mask.shape import _to_eager_additive

            existing_4d_mask = _to_eager_additive(existing_4d_mask, packed_kwargs["dtype"])
        mask = _merge_dense_attention_masks(existing_4d_mask, mask)

    if canonical == "eager" and cu_seqlens.numel() - 1 > packed_kwargs["batch_size"]:
        packed_only_kwargs = dict(packed_kwargs)
        packed_only_kwargs.pop("attention_mask", None)
        packed_only_kwargs["impl"] = "sdpa"
        if sliding_window is not None:
            packed_only = builder(
                q_len,
                kv_len,
                sliding_window=sliding_window,
                cu_seqlens=cu_seqlens,
                **packed_only_kwargs,
            )
        else:
            packed_only = builder(q_len, kv_len, cu_seqlens=cu_seqlens, **packed_only_kwargs)
        assert isinstance(packed_only, Tensor)
        mask = _merge_dense_attention_masks(mask, packed_only)

    return mask


def _build_packed_mask_kwargs(
    *,
    config: PreTrainedConfig,
    inputs_embeds: Tensor,
    attention_mask: Tensor | None,
    or_mask_function: Callable | None,
    and_mask_function: Callable | None,
    block_sequence_ids: Tensor | None,
    attn_impl: str,
    cu_seqlens: Tensor,
    cu_seqlens_k: Tensor | None,
    q_len: int,
    kv_len: int,
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Assemble keyword arguments for ``packed_causal_mask`` / ``sliding_window_mask``."""
    packed_kwargs: dict[str, Any] = {
        "impl": attn_impl,
        "device": inputs_embeds.device,
        "batch_size": inputs_embeds.shape[0],
        "dtype": inputs_embeds.dtype,
        "skip_ulysses": _resolve_skip_ulysses(attn_impl, extra_kwargs),
    }
    if attention_mask is not None and attention_mask.ndim == 2:
        packed_kwargs["attention_mask"] = attention_mask
    if not getattr(config, "is_causal", True):
        packed_kwargs["mask_function"] = bidirectional_mask_function
    if block_sequence_ids is not None:
        block_overlay = blockwise_overlay(block_sequence_ids)
        or_mask_function = block_overlay if or_mask_function is None else or_masks(or_mask_function, block_overlay)
    if or_mask_function is not None:
        packed_kwargs["or_mask_function"] = or_mask_function
    if and_mask_function is not None:
        packed_kwargs["and_mask_function"] = and_mask_function
    if cu_seqlens_k is not None and (q_len != kv_len or cu_seqlens_k.numel() != cu_seqlens.numel()):
        packed_kwargs["cu_seqlens_k"] = cu_seqlens_k
    if "compile_block_mask" in extra_kwargs:
        packed_kwargs["compile_block_mask"] = extra_kwargs["compile_block_mask"]
    return packed_kwargs


def create_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: Tensor,
    attention_mask: Tensor | BlockMask | MagiAttentionMask | None,
    past_key_values: Cache | None,
    position_ids: Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: Tensor | None = None,
    layer_idx: int | None = None,
    allow_is_causal_skip: bool = True,
    **kwargs: Any,
) -> Tensor | BlockMask | MagiAttentionMask | None:
    """Create a causal mask, routing packed ``cu_seq_lens_q`` inputs through ``veomni.ops.mask``."""
    if isinstance(attention_mask, (BlockMask, MagiAttentionMask)):
        return attention_mask

    attn_impl = _resolve_attn_implementation(config, kwargs.get("attn_implementation"))
    cu_seqlens, cu_seqlens_k = _extract_cu_seqlens(kwargs)
    if _should_use_packed_mask(cu_seqlens, attn_impl, inputs_embeds.shape[0]):
        assert cu_seqlens is not None
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_seen > 0:
            raise ValueError("Packed causal attention mask does not support cached sequences.")
        existing_4d = attention_mask if isinstance(attention_mask, Tensor) and attention_mask.ndim >= 3 else None
        q_len = inputs_embeds.shape[1]
        kv_len = q_len + past_seen
        if existing_4d is not None:
            _validate_existing_4d_mask(existing_4d, q_len, attn_impl, kwargs)
        packed_kwargs = _build_packed_mask_kwargs(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask if existing_4d is None else None,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
            block_sequence_ids=block_sequence_ids,
            attn_impl=attn_impl,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            q_len=q_len,
            kv_len=kv_len,
            extra_kwargs=kwargs,
        )
        mask = packed_causal_mask(q_len, kv_len, cu_seqlens=cu_seqlens, **packed_kwargs)
        return _finalize_packed_mask(
            mask,
            builder=packed_causal_mask,
            q_len=q_len,
            kv_len=kv_len,
            cu_seqlens=cu_seqlens,
            packed_kwargs=packed_kwargs,
            existing_4d_mask=existing_4d,
        )

    if _is_prebuilt_mask(attention_mask):
        return attention_mask

    hf_impl = "veomni_sage_attention" if attn_impl == "sage_attention" else attn_impl
    hf_config = config if getattr(config, "_attn_implementation", None) == hf_impl else _AttnImplView(config, hf_impl)
    return _hf_create_causal_mask(
        config=hf_config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        or_mask_function=or_mask_function,
        and_mask_function=and_mask_function,
        block_sequence_ids=block_sequence_ids,
        layer_idx=layer_idx,
        allow_is_causal_skip=allow_is_causal_skip,
    )


def create_sliding_window_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: Tensor,
    attention_mask: Tensor | BlockMask | MagiAttentionMask | None,
    past_key_values: Cache | None,
    position_ids: Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: Tensor | None = None,
    layer_idx: int | None = None,
    allow_is_causal_skip: bool = True,
    sliding_window: int | None = None,
    **kwargs: Any,
) -> Tensor | BlockMask | MagiAttentionMask | None:
    """Create a sliding-window causal mask, routing packed ``cu_seq_lens_q`` inputs through ``veomni.ops.mask``.

    Flash-like backends return ``None`` (or a 2D padding mask) here and do not
    encode the window in the returned mask; callers must still forward
    ``sliding_window=...`` to the attention forward. When ``sliding_window`` is
    provided explicitly, it is cross-checked against ``config.sliding_window``.
    """
    if isinstance(attention_mask, (BlockMask, MagiAttentionMask)):
        return attention_mask

    attn_impl = _resolve_attn_implementation(config, kwargs.get("attn_implementation"))
    cu_seqlens, cu_seqlens_k = _extract_cu_seqlens(kwargs)
    if _should_use_packed_mask(cu_seqlens, attn_impl, inputs_embeds.shape[0]):
        assert cu_seqlens is not None
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_seen > 0:
            raise ValueError("Packed sliding-window attention mask does not support cached sequences.")
        resolved_window = _resolve_sliding_window(config, sliding_window)
        existing_4d = attention_mask if isinstance(attention_mask, Tensor) and attention_mask.ndim >= 3 else None
        q_len = inputs_embeds.shape[1]
        kv_len = q_len + past_seen
        if existing_4d is not None:
            _validate_existing_4d_mask(existing_4d, q_len, attn_impl, kwargs)
        packed_kwargs = _build_packed_mask_kwargs(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask if existing_4d is None else None,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
            block_sequence_ids=block_sequence_ids,
            attn_impl=attn_impl,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            q_len=q_len,
            kv_len=kv_len,
            extra_kwargs=kwargs,
        )
        mask = sliding_window_mask(
            q_len,
            kv_len,
            sliding_window=resolved_window,
            cu_seqlens=cu_seqlens,
            **packed_kwargs,
        )
        return _finalize_packed_mask(
            mask,
            builder=sliding_window_mask,
            q_len=q_len,
            kv_len=kv_len,
            cu_seqlens=cu_seqlens,
            packed_kwargs=packed_kwargs,
            existing_4d_mask=existing_4d,
            sliding_window=resolved_window,
        )

    if _is_prebuilt_mask(attention_mask):
        return attention_mask

    resolved_window = _resolve_sliding_window(config, sliding_window)
    hf_impl = "veomni_sage_attention" if attn_impl == "sage_attention" else attn_impl
    needs_view = (
        getattr(config, "_attn_implementation", None) != hf_impl
        or getattr(config, "sliding_window", None) != resolved_window
    )
    hf_config = _AttnImplView(config, hf_impl, sliding_window=resolved_window) if needs_view else config
    return _hf_create_sliding_window_causal_mask(
        config=hf_config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        or_mask_function=or_mask_function,
        and_mask_function=and_mask_function,
        block_sequence_ids=block_sequence_ids,
        layer_idx=layer_idx,
        allow_is_causal_skip=allow_is_causal_skip,
    )


__all__ = ["create_causal_mask", "create_sliding_window_causal_mask"]
