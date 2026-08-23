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

"""Shared normalization semantics for gated delta-rule producers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


ExternalL2Norm = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class ExternalGatedDeltaRuleL2Norm:
    """One runtime-owned exact L2Norm implementation.

    Open-VeOmni deliberately does not import private kernel packages.  A
    runtime that registers an out-of-tree GDN provider must register the
    provider's matching L2Norm autograd function as well.  The explicit
    identity prevents a later overlay from silently changing numerical
    semantics after the model has been constructed.
    """

    provider: ExternalL2Norm
    identity: str


_EXTERNAL_L2NORM_PROVIDERS: dict[str, ExternalGatedDeltaRuleL2Norm] = {}


def producer_dtype_l2norm(x: torch.Tensor, *, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Normalize in the producer's active arithmetic context and storage dtype.

    This is the historical NPU GDR expression.  In particular, it must not be
    replaced by an unconditional fp32 reduction followed by a cast: KCP and the
    local GDR core must consume the exact same normalized key tensor.
    """

    original_dtype = x.dtype
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return (x * inv_norm).to(original_dtype)


def register_external_gated_delta_rule_l2norm(
    implementation: str,
    provider: ExternalL2Norm,
    *,
    identity: str,
) -> None:
    """Register an out-of-tree provider's exact L2Norm implementation.

    ``provider`` must accept ``provider(tensor, eps=<float>)`` and return a
    tensor with the same shape, device, and dtype.  Registration is
    fail-closed: replacing either the callable or its immutable identity in a
    live process is rejected.  The registry is process-local: a runtime must
    call this function and attest the identity in every torchrun worker rather
    than rely on parent-process or ``fork`` inheritance.
    """

    if not isinstance(implementation, str) or not implementation:
        raise ValueError("GDN external L2Norm implementation name must be a non-empty string")
    if not callable(provider):
        raise TypeError("GDN external L2Norm provider must be callable")
    if not isinstance(identity, str) or not identity:
        raise ValueError("GDN external L2Norm provider identity must be a non-empty string")

    candidate = ExternalGatedDeltaRuleL2Norm(provider=provider, identity=identity)
    existing = _EXTERNAL_L2NORM_PROVIDERS.get(implementation)
    if existing is None:
        _EXTERNAL_L2NORM_PROVIDERS[implementation] = candidate
        return
    if existing != candidate:
        if existing.identity != identity:
            raise RuntimeError(
                "GDN external L2Norm provider is already registered with a different identity: "
                f"implementation={implementation!r} existing={existing.identity!r} requested={identity!r}"
            )
        raise RuntimeError(
            "GDN external L2Norm provider identity is already bound to a different callable: "
            f"implementation={implementation!r} identity={identity!r}"
        )


def get_external_gated_delta_rule_l2norm_identity(implementation: str) -> str:
    """Return the immutable provider identity for runtime attestation."""

    registration = _EXTERNAL_L2NORM_PROVIDERS.get(implementation)
    if registration is None:
        raise RuntimeError(f"GDN external L2Norm provider is not registered: implementation={implementation!r}")
    return registration.identity


def external_gated_delta_rule_l2norm(
    x: torch.Tensor,
    *,
    implementation: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply the registered exact external norm or fail closed."""

    registration = _EXTERNAL_L2NORM_PROVIDERS.get(implementation)
    if registration is None:
        raise RuntimeError(
            "GDN external L2Norm provider is not registered; refusing to mix an out-of-tree kernel "
            "with Open-VeOmni normalization semantics: "
            f"implementation={implementation!r}"
        )
    output = registration.provider(x, eps=eps)
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "GDN external L2Norm provider must return a tensor: "
            f"implementation={implementation!r} identity={registration.identity!r}"
        )
    if output.shape != x.shape or output.device != x.device or output.dtype != x.dtype:
        raise RuntimeError(
            "GDN external L2Norm provider changed the tensor contract: "
            f"implementation={implementation!r} identity={registration.identity!r} "
            f"input=(shape={tuple(x.shape)}, device={x.device}, dtype={x.dtype}) "
            f"output=(shape={tuple(output.shape)}, device={output.device}, dtype={output.dtype})"
        )
    return output


__all__ = [
    "ExternalGatedDeltaRuleL2Norm",
    "external_gated_delta_rule_l2norm",
    "get_external_gated_delta_rule_l2norm_identity",
    "producer_dtype_l2norm",
    "register_external_gated_delta_rule_l2norm",
]
