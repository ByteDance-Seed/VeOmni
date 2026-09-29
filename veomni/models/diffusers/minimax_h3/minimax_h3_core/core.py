"""Core utilities for MiniMax H3.

Attention binds an instance-local ``attention/standard`` handle through
``resolve_op_impl``. Gradient checkpointing stays here for the DiT blocks.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl


def bind_minimax_attention(module: nn.Module, *, is_causal: bool, impl: str | None = None) -> None:
    """Attach the configured ``attention/standard`` handle and HF interface attrs."""
    impl = impl or resolve_op_impl("attn_implementation")
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


def packed_block_diag_mask(cu_seqlens: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    """Boolean ``(1, 1, S, S)`` mask that keeps attention inside packed segments."""
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError(f"cu_seqlens must be a torch.Tensor, got {type(cu_seqlens).__name__}")
    cu_seqlens = cu_seqlens.to(device=device)
    positions = torch.arange(seq_len, device=device)
    segment_ids = torch.bucketize(positions, cu_seqlens[1:], right=True)
    mask = segment_ids[:, None] == segment_ids[None, :]
    return mask.view(1, 1, seq_len, seq_len)


def is_flash_attn_impl(impl: str) -> bool:
    """True when the selected attention impl can take FA varlen kwargs."""
    return "flash_attention" in impl


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
