"""Core utilities for MiniMax H3.

Provides attention dispatch and gradient checkpointing compatible with
the original minimax_h3_dit module.
"""

from __future__ import annotations

import inspect
import os

import torch
import torch.nn.functional as F
from einops import rearrange


# ── Attention backend detection ──────────────────────────────────────

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    import xformers.ops as xops

    XFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    XFORMERS_AVAILABLE = False

try:
    if "enable_gqa" in inspect.signature(torch.nn.functional.scaled_dot_product_attention).parameters:
        TORCH_SUPPORT_GQA = True
    else:
        TORCH_SUPPORT_GQA = False
except Exception:
    TORCH_SUPPORT_GQA = False


def _initialize_attention_priority() -> str:
    env_val = os.environ.get("MINIMAX_H3_ATTENTION_IMPLEMENTATION")
    if env_val is not None:
        return env_val.lower()
    if FLASH_ATTN_3_AVAILABLE:
        return "flash_attention_3"
    if FLASH_ATTN_2_AVAILABLE:
        return "flash_attention_2"
    if SAGE_ATTN_AVAILABLE:
        return "sage_attention"
    if XFORMERS_AVAILABLE:
        return "xformers"
    return "torch"


ATTENTION_IMPLEMENTATION = _initialize_attention_priority()


# ── Rearrange helpers (einops) ───────────────────────────────────────


def _rearrange_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern: str = "b n s d",
    k_pattern: str = "b n s d",
    v_pattern: str = "b n s d",
    target: str = "b n s d",
    dims: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rearrange q/k/v to target pattern.

    Einops-based rearrangement, supports any pattern including merged head
    dims like "b s (n d)".
    """
    dims = {} if dims is None else dims

    def _to_pattern(t: torch.Tensor, pat: str) -> torch.Tensor:
        if pat != target:
            t = rearrange(t, f"{pat} -> {target}", **dims)
        return t

    return _to_pattern(q, q_pattern), _to_pattern(k, k_pattern), _to_pattern(v, v_pattern)


def _rearrange_out(
    out: torch.Tensor,
    out_pattern: str = "b n s d",
    target: str = "b n s d",
    dims: dict | None = None,
) -> torch.Tensor:
    """Reverse of _rearrange_qkv for output."""
    dims = {} if dims is None else dims
    if out_pattern != target:
        out = rearrange(out, f"{target} -> {out_pattern}", **dims)
    return out


# ── Backend implementations ──────────────────────────────────────────


def _torch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern: str = "b n s d",
    k_pattern: str = "b n s d",
    v_pattern: str = "b n s d",
    out_pattern: str = "b n s d",
    dims: dict | None = None,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    q, k, v = _rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, "b n s d", dims)

    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        if TORCH_SUPPORT_GQA:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale, is_causal=is_causal, enable_gqa=True)
        else:
            reps = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale, is_causal=is_causal)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale, is_causal=is_causal)

    return _rearrange_out(out, out_pattern, "b n s d", dims)


def _flash_attn_3_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern: str = "b n s d",
    k_pattern: str = "b n s d",
    v_pattern: str = "b n s d",
    out_pattern: str = "b n s d",
    dims: dict | None = None,
    scale: float | None = None,
    is_causal: bool = False,
    window_size: int | None = None,
) -> torch.Tensor:
    q, k, v = _rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, "b s n d", dims)
    ws = (window_size, window_size) if window_size is not None else (-1, -1)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale, causal=is_causal, window_size=ws)
    if isinstance(out, tuple):
        out = out[0]
    return _rearrange_out(out, out_pattern, "b s n d", dims)


def _flash_attn_2_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern: str = "b n s d",
    k_pattern: str = "b n s d",
    v_pattern: str = "b n s d",
    out_pattern: str = "b n s d",
    dims: dict | None = None,
    scale: float | None = None,
    is_causal: bool = False,
    window_size: int | None = None,
) -> torch.Tensor:
    q, k, v = _rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, "b s n d", dims)
    ws = (window_size, window_size) if window_size is not None else (-1, -1)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale, causal=is_causal, window_size=ws)
    return _rearrange_out(out, out_pattern, "b s n d", dims)


def _sage_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern: str = "b n s d",
    k_pattern: str = "b n s d",
    v_pattern: str = "b n s d",
    out_pattern: str = "b n s d",
    dims: dict | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    q, k, v = _rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, "b n s d", dims)
    out = sageattn(q, k, v, sm_scale=scale)
    return _rearrange_out(out, out_pattern, "b n s d", dims)


def _xformers_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern: str = "b n s d",
    k_pattern: str = "b n s d",
    v_pattern: str = "b n s d",
    out_pattern: str = "b n s d",
    dims: dict | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    q, k, v = _rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, "b s n d", dims)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    return _rearrange_out(out, out_pattern, "b s n d", dims)


# ── Main dispatch ────────────────────────────────────────────────────


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern: str = "b n s d",
    k_pattern: str = "b n s d",
    v_pattern: str = "b n s d",
    out_pattern: str = "b n s d",
    dims: dict | None = None,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = False,
    compatibility_mode: bool = False,
    window_size: int | None = None,
) -> torch.Tensor:
    """Dispatch attention to available backend."""
    if compatibility_mode or (attn_mask is not None) or ATTENTION_IMPLEMENTATION == "torch":
        return _torch_sdpa(
            q,
            k,
            v,
            q_pattern,
            k_pattern,
            v_pattern,
            out_pattern,
            dims,
            attn_mask=attn_mask,
            scale=scale,
            is_causal=is_causal,
        )
    if ATTENTION_IMPLEMENTATION == "flash_attention_3":
        return _flash_attn_3_forward(
            q,
            k,
            v,
            q_pattern,
            k_pattern,
            v_pattern,
            out_pattern,
            dims,
            scale=scale,
            is_causal=is_causal,
            window_size=window_size,
        )
    if ATTENTION_IMPLEMENTATION == "flash_attention_2":
        return _flash_attn_2_forward(
            q,
            k,
            v,
            q_pattern,
            k_pattern,
            v_pattern,
            out_pattern,
            dims,
            scale=scale,
            is_causal=is_causal,
            window_size=window_size,
        )
    if ATTENTION_IMPLEMENTATION == "sage_attention":
        if window_size is not None or is_causal:
            return attention_forward(
                q,
                k,
                v,
                q_pattern,
                k_pattern,
                v_pattern,
                out_pattern,
                dims,
                attn_mask,
                scale,
                is_causal,
                compatibility_mode=True,
            )
        return _sage_attn_forward(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if ATTENTION_IMPLEMENTATION == "xformers":
        if window_size is not None or is_causal:
            return attention_forward(
                q,
                k,
                v,
                q_pattern,
                k_pattern,
                v_pattern,
                out_pattern,
                dims,
                attn_mask,
                scale,
                is_causal,
                compatibility_mode=True,
            )
        return _xformers_forward(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    raise NotImplementedError(f"No available attention implementation (current: {ATTENTION_IMPLEMENTATION}).")


# ── Gradient checkpointing ────────────────────────────────────────────

try:
    import deepspeed

    _HAS_DEEPSPEED = True
except ModuleNotFoundError:
    _HAS_DEEPSPEED = False


def _create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)

    return custom_forward


def _create_custom_forward_use_reentrant(module):
    def custom_forward(*inputs):
        return module(*inputs)

    return custom_forward


def _judge_args_requires_grad(*args) -> bool:
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            return True
    return False


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing: bool,
    use_gradient_checkpointing_offload: bool,
    *args,
    **kwargs,
):
    """Gradient checkpoint wrapper.

    Delegates to deepspeed checkpointing when configured, torch checkpointing
    otherwise. Falls back to direct call when checkpointing is disabled.
    """
    if use_gradient_checkpointing and _HAS_DEEPSPEED and deepspeed.checkpointing.is_configured():
        all_args = args + tuple(kwargs.values())
        if not _judge_args_requires_grad(*all_args):
            return model(*args, **kwargs)
        return deepspeed.checkpointing.checkpoint(
            _create_custom_forward_use_reentrant(model),
            *all_args,
        )
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            return torch.utils.checkpoint.checkpoint(
                _create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
    if use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            _create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    return model(*args, **kwargs)
