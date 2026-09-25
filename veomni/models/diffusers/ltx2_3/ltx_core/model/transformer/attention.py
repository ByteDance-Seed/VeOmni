from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Protocol

import torch
from ltx_core.model.transformer.ops import (
    GatedAttentionCallable,
    PreAttentionCallable,
    PytorchGatedAttention,
    PytorchPreAttention,
)
from ltx_core.model.transformer.rope import LTXRopeType

from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl


_SDPA_ATTN_IMPLS = frozenset({"sdpa", "veomni_sdpa"})


class AttentionCallable(Protocol):
    """Unmasked attention. Backends without a mask kernel (FA3/FA4) implement only
    this protocol; backends that support masks too (Pytorch/SDPA, xFormers) are
    structurally usable here and as :class:`MaskedAttentionCallable`."""

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor: ...


class MaskedAttentionCallable(Protocol):
    """Masked attention. Mask is required (not optional) -- the caller has already
    decided this is the masked path and chosen a backend that can serve it. Used
    by :class:`Attention` when its forward receives a non-None ``mask``."""

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor
    ) -> torch.Tensor: ...


def _hf_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    """Expand a 2D/3D/4D LTX mask to HF ``(B, 1|H, T, S)``. Do not change bool vs additive."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"LTX attention mask must be 2D, 3D, or 4D, got {mask.ndim}D")
    return mask


class VeomniLTXAttention:
    """``AttentionCallable`` / ``MaskedAttentionCallable`` adapter over ``attention/standard``."""

    def __init__(self, impl: str) -> None:
        self.veomni_attn = VeomniOp("attention", "standard", impl)
        self.veomni_attn_masked = (
            self.veomni_attn if impl in _SDPA_ATTN_IMPLS else VeomniOp("attention", "standard", "sdpa")
        )
        self.is_causal = False
        self.layer_idx = None
        self.num_key_value_groups = 1
        self.config = SimpleNamespace(_attn_implementation=impl)

    def modules(self):
        return iter(())

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_q, inner = q.shape
        dim_head = inner // heads
        query = q.view(batch, seq_q, heads, dim_head).transpose(1, 2)
        key = k.view(batch, -1, heads, dim_head).transpose(1, 2)
        value = v.view(batch, -1, heads, dim_head).transpose(1, 2)
        attention_mask = None if mask is None else _hf_attention_mask(mask)
        handle = self.veomni_attn_masked if attention_mask is not None else self.veomni_attn
        output, _ = handle(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            is_causal=False,
            skip_ulysses=True,
        )
        return output.reshape(batch, seq_q, heads * dim_head)


def _impl_from_attention_function(fn: "AttentionFunction") -> str:
    if fn is AttentionFunction.AUTOMATIC:
        return resolve_op_impl("attn_implementation")
    if fn is AttentionFunction.FLASH_ATTENTION_3:
        return "flash_attention_3"
    if fn is AttentionFunction.FLASH_ATTENTION_4:
        return "flash_attention_4"
    return "sdpa"


def _impl_from_masked_function(fn: "MaskedAttentionFunction") -> str:
    if fn is MaskedAttentionFunction.AUTOMATIC:
        return resolve_op_impl("attn_implementation")
    return "sdpa"


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_4 = "flash_attention_4"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_FLASH = "sdpa_flash"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
    AUTOMATIC = "automatic"

    def to_callable(self) -> AttentionCallable:
        """Resolve to a VeOmni attention adapter at module construction time.

        ``AUTOMATIC`` reads ``resolve_op_impl("attn_implementation")``. SDPA pin
        names and xformers collapse to ``sdpa``. Flash-3/4 keep those impl names.
        """
        return VeomniLTXAttention(_impl_from_attention_function(self))


class MaskedAttentionFunction(Enum):
    """Backends usable on the masked path. FA names stay off this enum because
    those kernels cannot take a dense mask. ``to_callable`` still returns the
    shared adapter; a non-SDPA unmasked impl falls back to ``sdpa`` when a mask
    is present.
    """

    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
    AUTOMATIC = "automatic"

    def to_callable(self) -> MaskedAttentionCallable:
        """Resolve to the same adapter used by :class:`AttentionFunction`."""
        return VeomniLTXAttention(_impl_from_masked_function(self))


@dataclass(frozen=True)
class AttentionOps:
    """Pluggable callables consumed by :class:`Attention`."""

    attention_function: AttentionCallable = field(default_factory=lambda: AttentionFunction.AUTOMATIC.to_callable())
    masked_attention_function: MaskedAttentionCallable = field(
        default_factory=lambda: MaskedAttentionFunction.AUTOMATIC.to_callable()
    )
    preattention_function: PreAttentionCallable = field(default_factory=PytorchPreAttention)
    gated_attention_function: GatedAttentionCallable = field(default_factory=PytorchGatedAttention)


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        ops: AttentionOps | None = None,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        if ops is None:
            ops = AttentionOps()
        self.rope_type = rope_type
        self.attention_function = ops.attention_function
        self.masked_attention_function = ops.masked_attention_function
        self.preattention_function = ops.preattention_function
        self.gated_attention_function = ops.gated_attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.
        When ``perturbation_mask`` is all zeros, the expensive query/key path
        (linear projections, RMSNorm, RoPE) is skipped entirely and only the
        value projection is used as a pass-through.
        Args:
            x: Query input tensor of shape ``(B, T, query_dim)``.
            context: Key/value context tensor of shape ``(B, S, context_dim)``.
                Falls back to ``x`` (self-attention) when *None*.
            mask: Optional attention mask. Interpretation depends on the attention
                backend (additive bias for xformers/PyTorch SDPA). A non-None
                ``mask`` routes to ``masked_attention_function``; ``None`` keeps
                the unmasked path.
            pe: Rotary positional embeddings applied to both ``q`` and ``k``.
            k_pe: Separate rotary positional embeddings for ``k`` only. When
                *None*, ``pe`` is reused for keys.
            perturbation_mask: Optional mask in ``[0, 1]`` that
                blends the attention output with the raw value projection:
                ``out = attn_out * mask + v * (1 - mask)``.
                **1** keeps the full attention output, **0** bypasses attention
                and passes the value projection through unchanged.
                *None* or all-ones means standard attention; all-zeros skips
                the query/key path entirely for efficiency.
            all_perturbed: Whether all perturbations are active for this block.
        Returns:
            Output tensor of shape ``(B, T, query_dim)``.
        """
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)
            q, k = self.preattention_function(q, k, self, mask, pe, k_pe)
            if mask is None:
                out = self.attention_function(q, k, v, self.heads)  # (B, T, H*D)
            else:
                out = self.masked_attention_function(q, k, v, self.heads, mask)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        # Apply per-head gating if enabled
        if self.to_gate_logits is not None:
            out = self.gated_attention_function(x, out, self)

        return self.to_out(out)
