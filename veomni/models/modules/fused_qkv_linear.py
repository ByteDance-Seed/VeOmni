"""Fused QKV projection module used by the Ulysses ws_push path.

``FusedQKVLinear`` stores Q, K, and V as one row-concatenated Parameter so
``WsPushDispatch.try_resolve_fused`` can feed the pre-packed tensor to the
ws_push kernel without a per-forward ``torch.cat`` or host-side cache. Its
eager ``forward`` remains a normal ``F.linear`` plus Q/K/V slicing fallback.

Checkpoint contract:
* HF checkpoints are loaded through model-level converters that merge
  ``q_proj`` / ``k_proj`` / ``v_proj`` tensors into ``qkv_proj`` before module
  loading.
* VeOmni checkpoints save fused ``qkv_proj`` tensors; HF export must split
  them back with ``scripts/qkv_split_dcp_to_hf.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["FusedQKVLinear"]


class FusedQKVLinear(nn.Module):
    """``nn.Linear``-equivalent that owns one ``[q | k | v]`` Parameter.

    The weight is row-concatenated along ``dim=0``:

    .. code-block::

        weight[ : n_q * head_dim                               ] -> q part
        weight[ n_q * head_dim : (n_q + n_kv) * head_dim       ] -> k part
        weight[ (n_q + n_kv) * head_dim :                      ] -> v part

    The eager path returns ``(q, k, v)``. The ws_push path bypasses
    ``forward`` and passes ``self.weight`` to ``async_ulysses_qkv_projection``.
    """

    def __init__(
        self,
        hidden_size: int,
        n_q: int,
        n_kv: int,
        head_dim: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if n_q <= 0 or n_kv <= 0 or head_dim <= 0:
            raise ValueError(f"n_q, n_kv, head_dim must all be positive; got {n_q}, {n_kv}, {head_dim}")

        self.hidden_size = hidden_size
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self._q_out = n_q * head_dim
        self._kv_out = n_kv * head_dim
        total_out = self._q_out + 2 * self._kv_out
        self.out_features = total_out
        self.in_features = hidden_size

        factory_kwargs = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(torch.empty(total_out, hidden_size, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(total_out, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match ``nn.Linear`` init so a fresh fused layer behaves like three
        # independent Linears whose outputs are concatenated.
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager fallback: one ``F.linear`` followed by Q/K/V slicing."""
        out = F.linear(hidden_states, self.weight, self.bias)
        q = out[..., : self._q_out]
        k = out[..., self._q_out : self._q_out + self._kv_out]
        v = out[..., self._q_out + self._kv_out :]
        return q, k, v

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, n_q={self.n_q}, n_kv={self.n_kv}, "
            f"head_dim={self.head_dim}, bias={self.bias is not None}"
        )

    # Constructors used by patchgen init patches.
    @classmethod
    def from_separate_linears(
        cls,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        head_dim: int,
    ) -> FusedQKVLinear:
        """Create a fused module from initialized HF ``q/k/v_proj`` Linears.

        Patchgen calls this after the original attention ``__init__`` runs.
        """
        from torch.distributed.tensor import DTensor

        for name, proj in (("q_proj", q_proj), ("k_proj", k_proj), ("v_proj", v_proj)):
            if isinstance(proj.weight, DTensor):
                raise RuntimeError(
                    f"from_separate_linears must run before FSDP2 wraps the module; {name}.weight is already a DTensor"
                )
        if q_proj.in_features != k_proj.in_features or q_proj.in_features != v_proj.in_features:
            raise ValueError(
                "q/k/v_proj must share in_features; got "
                f"{q_proj.in_features}, {k_proj.in_features}, {v_proj.in_features}"
            )
        if k_proj.out_features != v_proj.out_features:
            raise ValueError(f"k/v_proj must share out_features; got {k_proj.out_features}, {v_proj.out_features}")
        hidden_size = q_proj.in_features
        n_q = q_proj.out_features // head_dim
        n_kv = k_proj.out_features // head_dim
        if n_q * head_dim != q_proj.out_features:
            raise ValueError(f"q_proj.out_features={q_proj.out_features} not divisible by head_dim={head_dim}")
        if n_kv * head_dim != k_proj.out_features:
            raise ValueError(f"k_proj.out_features={k_proj.out_features} not divisible by head_dim={head_dim}")

        has_bias_q = q_proj.bias is not None
        has_bias_k = k_proj.bias is not None
        has_bias_v = v_proj.bias is not None
        if has_bias_q != has_bias_k or has_bias_q != has_bias_v:
            raise ValueError("q/k/v_proj must agree on bias presence; got mixed")

        fused = cls(
            hidden_size=hidden_size,
            n_q=n_q,
            n_kv=n_kv,
            head_dim=head_dim,
            bias=has_bias_q,
            dtype=q_proj.weight.dtype,
            device=q_proj.weight.device,
        )
        with torch.no_grad():
            fused.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
            if has_bias_q:
                fused.bias.copy_(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0))
        return fused

    # Zero-copy per-projection views used by fallback and backward paths.
    def q_weight_view(self) -> torch.Tensor:
        return self.weight[: self._q_out]

    def k_weight_view(self) -> torch.Tensor:
        return self.weight[self._q_out : self._q_out + self._kv_out]

    def v_weight_view(self) -> torch.Tensor:
        return self.weight[self._q_out + self._kv_out :]

    def q_bias_view(self) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias[: self._q_out]

    def k_bias_view(self) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias[self._q_out : self._q_out + self._kv_out]

    def v_bias_view(self) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias[self._q_out + self._kv_out :]
