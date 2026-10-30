"""Core utilities for MiniMax H3.

Attention binds an instance-local ``attention/standard`` handle through
``config.ops_config``. Gradient checkpointing lives in ``veomni.utils.recompute_utils``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from veomni.ops import VeomniOp


def bind_minimax_attention(module: nn.Module, *, is_causal: bool, impl: str | None = None, config=None) -> None:
    """Attach the configured ``attention/standard`` handle and HF interface attrs."""
    if impl is None:
        impl = config.ops_config.attn_implementation if config is not None else "veomni_sdpa"
    num_heads = module.num_heads
    module.veomni_attn = VeomniOp("attention", "standard", impl)
    module.is_causal = is_causal
    module.layer_idx = getattr(module, "layer_idx", None)
    module.num_key_value_heads = num_heads
    module.num_key_value_groups = 1
    module.config = SimpleNamespace(_attn_implementation=impl)


def minimax_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    sliding_window: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """Run interned attention on ``(B, H, S, D)`` and return the same layout."""
    output, _ = module.veomni_attn(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=scaling,
        sliding_window=sliding_window,
        is_causal=module.is_causal,
        skip_ulysses=True,
        **kwargs,
    )
    return output.transpose(1, 2)


def is_flash_attn_impl(impl: str) -> bool:
    """True when the selected attention impl can take FA varlen kwargs."""
    return "flash_attention" in impl


def is_sdpa_attn_impl(impl: str) -> bool:
    """True when MiniMax attention uses the SDPA path, including its eager row."""
    return impl in ("eager", "sdpa", "veomni_sdpa")
