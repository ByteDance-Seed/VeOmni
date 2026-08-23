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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime-owned affine-summary providers for lossless GDN KCP.

Open-VeOmni owns the neutral KCP ABI but deliberately does not import private
kernel packages. A runtime that selects an out-of-tree GDN implementation must
register the matching affine summary in every torchrun worker before model
construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import torch


ExternalKcpAffineSummary = Callable[..., torch.Tensor]
ExternalKcpAffinePrepare = Callable[..., None]


@dataclass(frozen=True)
class ExternalKcpAffineSummaryRegistration:
    """One immutable process-local KCP affine provider."""

    provider: ExternalKcpAffineSummary
    identity: str
    prepare: ExternalKcpAffinePrepare | None = None


_EXTERNAL_KCP_AFFINE_SUMMARIES: dict[str, ExternalKcpAffineSummaryRegistration] = {}


def register_external_kcp_affine_summary(
    implementation: str,
    provider: ExternalKcpAffineSummary,
    *,
    identity: str,
    prepare: ExternalKcpAffinePrepare | None = None,
) -> None:
    """Register an exact out-of-tree affine summary without allowing drift.

    ``provider`` must implement ``(key, value, g, beta, *, cu_seqlens,
    use_qk_l2norm, eps) -> hm``. ``prepare``, when present, receives only the
    shape/dtype/device keyword contract accepted by
    :func:`prepare_external_kcp_affine_summary`.
    """

    if not isinstance(implementation, str) or not implementation:
        raise ValueError("KCP external affine implementation name must be a non-empty string")
    if not callable(provider):
        raise TypeError("KCP external affine provider must be callable")
    if not isinstance(identity, str) or not identity:
        raise ValueError("KCP external affine provider identity must be a non-empty string")
    if prepare is not None and not callable(prepare):
        raise TypeError("KCP external affine prepare hook must be callable")

    candidate = ExternalKcpAffineSummaryRegistration(provider=provider, identity=identity, prepare=prepare)
    existing = _EXTERNAL_KCP_AFFINE_SUMMARIES.get(implementation)
    if existing is None:
        _EXTERNAL_KCP_AFFINE_SUMMARIES[implementation] = candidate
        return
    if existing != candidate:
        if existing.identity != identity:
            raise RuntimeError(
                "KCP external affine provider is already registered with a different identity: "
                f"implementation={implementation!r} existing={existing.identity!r} requested={identity!r}"
            )
        raise RuntimeError(
            "KCP external affine provider identity is already bound to a different callable or prepare hook: "
            f"implementation={implementation!r} identity={identity!r}"
        )


def _get_registration(implementation: str) -> ExternalKcpAffineSummaryRegistration:
    registration = _EXTERNAL_KCP_AFFINE_SUMMARIES.get(implementation)
    if registration is None:
        raise RuntimeError(
            "KCP external affine provider is not registered; refusing to mix an out-of-tree GDN kernel "
            "with a different prefix recurrence: "
            f"implementation={implementation!r}"
        )
    return registration


def get_external_kcp_affine_summary_identity(implementation: str) -> str:
    """Return the immutable provider identity for runtime attestation."""

    return _get_registration(implementation).identity


def external_kcp_affine_summary(
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    implementation: str,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_list: Sequence[int] | None = None,
    use_qk_l2norm: bool,
    eps: float,
) -> torch.Tensor:
    """Execute an external affine provider and validate the public KCP ABI."""

    registration = _get_registration(implementation)
    provider_kwargs = {
        "cu_seqlens": cu_seqlens,
        "use_qk_l2norm": use_qk_l2norm,
        "eps": eps,
    }
    if cu_seqlens_list is not None:
        provider_kwargs["cu_seqlens_list"] = tuple(cu_seqlens_list)
    hm = registration.provider(
        key,
        value,
        g,
        beta,
        **provider_kwargs,
    )
    if not isinstance(hm, torch.Tensor):
        raise TypeError(
            "KCP external affine provider must return a tensor: "
            f"implementation={implementation!r} identity={registration.identity!r}"
        )
    num_seqs = int(key.shape[0]) if cu_seqlens is None else int(cu_seqlens.numel() - 1)
    expected_shape = (
        num_seqs,
        int(key.shape[2]),
        int(key.shape[3]),
        int(value.shape[-1]) + int(key.shape[-1]),
    )
    if hm.shape != expected_shape or hm.device != key.device or hm.dtype != torch.float32:
        raise RuntimeError(
            "KCP external affine provider changed the hm contract: "
            f"implementation={implementation!r} identity={registration.identity!r} "
            f"expected=(shape={expected_shape}, device={key.device}, dtype={torch.float32}) "
            f"actual=(shape={tuple(hm.shape)}, device={hm.device}, dtype={hm.dtype})"
        )
    return hm


def prepare_external_kcp_affine_summary(
    implementation: str,
    *,
    device: torch.device,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    key_dtype: torch.dtype,
    value_dtype: torch.dtype,
    g_dtype: torch.dtype,
    beta_dtype: torch.dtype,
) -> None:
    """Prepare a registered provider outside activation checkpointing."""

    registration = _get_registration(implementation)
    if registration.prepare is None:
        return
    registration.prepare(
        device=device,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        g_dtype=g_dtype,
        beta_dtype=beta_dtype,
    )


__all__ = [
    "ExternalKcpAffineSummaryRegistration",
    "external_kcp_affine_summary",
    "get_external_kcp_affine_summary_identity",
    "prepare_external_kcp_affine_summary",
    "register_external_kcp_affine_summary",
]
