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

"""Explicit metadata ABI for Qwen3.5 gated-delta-rule backends.

The model produces a canonical device ``cu_seqlens`` plus optional host and
chunk-index metadata.  Native AscendC consumes all of it; the internal Mojo
provider currently consumes only the device CU tensor and rejects the newer
keyword arguments.  This module is the Open-VeOmni-owned ABI boundary: it
selects a declared capability contract and never relies on a provider's
``**kwargs`` signature or silently falls back to another backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .normalization import external_gated_delta_rule_l2norm, producer_dtype_l2norm


@dataclass(frozen=True)
class GatedDeltaRuleMetadataCapabilities:
    """Metadata accepted by one concrete GDN provider."""

    name: str
    accepts_cu_seqlens: bool = True
    accepts_cu_seqlens_list: bool = False
    accepts_chunk_indices: bool = False
    requires_external_qk_l2norm: bool = False


_CAPABILITIES = {
    # The internal Mojo ABI predates the precomputed host/chunk metadata.  A
    # device CU tensor is the canonical representation and is sufficient for
    # its varlen kernel; passing the newer keywords raises TypeError there.
    "mojo": GatedDeltaRuleMetadataCapabilities("mojo", requires_external_qk_l2norm=True),
    # FLA/FlashQLA expose the historical cu_seqlens-only ABI as well.
    "fla": GatedDeltaRuleMetadataCapabilities("fla"),
    "flash_qla": GatedDeltaRuleMetadataCapabilities("flash_qla"),
    # The vendored Triton NPU ABI consumes the device CU plus the host list,
    # but does not consume the newer chunk-index maps.  Keep it separate from
    # AscendC so ``**kwargs`` in the legacy wrapper cannot hide an ABI drift.
    "npu": GatedDeltaRuleMetadataCapabilities("npu", accepts_cu_seqlens_list=True),
    "npu_ascendc": GatedDeltaRuleMetadataCapabilities(
        "npu_ascendc", accepts_cu_seqlens_list=True, accepts_chunk_indices=True
    ),
}


def get_gated_delta_rule_metadata_capabilities(implementation: str) -> GatedDeltaRuleMetadataCapabilities:
    """Return the explicit metadata contract or fail closed."""

    try:
        return _CAPABILITIES[implementation]
    except KeyError as exc:
        raise RuntimeError(
            "Qwen3.5 GDN backend has no declared metadata ABI; refusing implicit kwargs filtering: "
            f"implementation={implementation!r}"
        ) from exc


def requires_chunked_varlen_metadata(implementation: str) -> bool:
    """Whether a provider needs the host/chunk precomputed varlen plan."""

    return get_gated_delta_rule_metadata_capabilities(implementation).accepts_chunk_indices


def prepare_gated_delta_rule_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    implementation: str,
    force_external: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Select one exact Q/K normalization path for the provider.

    Out-of-tree providers such as Mojo register their matching autograd norm at
    runtime; Open-VeOmni never imports those private packages.  KCP also forces
    normalization outside the local kernel so its affine prefix scan and the
    local GDR core consume the same tensor and autograd edge.  Built-in
    providers use the historical producer expression for that KCP-only case.

    Returns normalized ``(query, key)`` and the flag to pass as
    ``use_qk_l2norm_in_kernel``.
    """

    capabilities = get_gated_delta_rule_metadata_capabilities(implementation)
    if capabilities.requires_external_qk_l2norm:
        query = external_gated_delta_rule_l2norm(query, implementation=implementation, eps=eps)
        key = external_gated_delta_rule_l2norm(key, implementation=implementation, eps=eps)
        return query, key, False
    if force_external:
        query = producer_dtype_l2norm(query, eps=eps)
        key = producer_dtype_l2norm(key, eps=eps)
        return query, key, False
    return query, key, True


def _validate_cu_metadata(
    *,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_list: list[int] | None,
    chunk_indices: dict | None,
    chunk_indices_list: dict | None,
) -> None:
    if cu_seqlens is None:
        raise RuntimeError("GDN backend metadata ABI requires a device cu_seqlens tensor")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "GDN device cu_seqlens must be a 1-D int32/int64 tensor, "
            f"got shape={tuple(cu_seqlens.shape)} dtype={cu_seqlens.dtype}"
        )
    if cu_seqlens_list is not None:
        if len(cu_seqlens_list) != int(cu_seqlens.numel()):
            raise ValueError(
                "GDN host/device cu_seqlens metadata length mismatch: "
                f"host={len(cu_seqlens_list)} device={int(cu_seqlens.numel())}"
            )
        if any(not isinstance(point, int) or isinstance(point, bool) for point in cu_seqlens_list):
            raise TypeError("GDN host cu_seqlens_list must contain plain integers")
        if any(right < left for left, right in zip(cu_seqlens_list, cu_seqlens_list[1:])):
            raise ValueError("GDN host cu_seqlens_list must be non-decreasing")
        # CPU tensors are used by the ABI unit tests and can be compared
        # without an accelerator synchronization.  NPU tensors are deliberately
        # not read back here: the model constructs the host list from the same
        # canonical boundaries and the device tensor from that list.
        if cu_seqlens.device.type == "cpu":
            device_points = [int(point) for point in cu_seqlens.tolist()]
            if device_points != cu_seqlens_list:
                raise ValueError(
                    "GDN host/device cu_seqlens metadata values mismatch: "
                    f"host={cu_seqlens_list!r} device={device_points!r}"
                )
    if (chunk_indices is None) != (chunk_indices_list is None):
        raise ValueError("GDN chunk_indices and chunk_indices_list must be provided together")
    if chunk_indices is not None and chunk_indices_list is not None:
        tensor_keys = set(chunk_indices)
        host_keys = set(chunk_indices_list)
        if tensor_keys != host_keys:
            raise ValueError(
                "GDN chunk metadata key sets differ: "
                f"tensor={sorted(map(str, tensor_keys))} host={sorted(map(str, host_keys))}"
            )


def build_gated_delta_rule_metadata_kwargs(
    implementation: str,
    *,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_list: list[int] | None,
    chunk_indices: dict | None,
    chunk_indices_list: dict | None,
    metadata_is_canonical: bool = False,
) -> dict[str, Any]:
    """Build provider kwargs from the canonical metadata contract.

    Mojo receives only ``cu_seqlens`` because that is its declared ABI.  A
    caller may provide the host CU list only when it explicitly proves that
    the list is the canonical source used to construct the device tensor;
    otherwise unsupported metadata is rejected instead of silently dropped.
    Chunk maps are never accepted by Mojo. Public AscendC receives the full
    plan and therefore retains empty-segment/chunk ownership data.
    """

    capabilities = get_gated_delta_rule_metadata_capabilities(implementation)
    _validate_cu_metadata(
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
    )
    if not capabilities.accepts_chunk_indices and (chunk_indices is not None or chunk_indices_list is not None):
        raise ValueError(
            f"GDN backend does not accept chunk metadata; refusing to drop it: implementation={implementation!r}"
        )
    if not capabilities.accepts_cu_seqlens_list and cu_seqlens_list is not None and not metadata_is_canonical:
        raise ValueError(
            "GDN backend does not accept host CU metadata without an explicit canonical proof; "
            f"implementation={implementation!r}"
        )
    metadata: dict[str, Any] = {"cu_seqlens": cu_seqlens}
    if capabilities.accepts_cu_seqlens_list:
        metadata["cu_seqlens_list"] = cu_seqlens_list
    if capabilities.accepts_chunk_indices:
        metadata["chunk_indices"] = chunk_indices
        metadata["chunk_indices_list"] = chunk_indices_list
    return metadata


def call_chunk_gated_delta_rule(
    kernel: Callable[..., Any],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    implementation: str,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_list: list[int] | None,
    chunk_indices: dict | None,
    chunk_indices_list: dict | None,
    metadata_is_canonical: bool = False,
) -> Any:
    """Invoke a GDN provider with exactly the metadata its ABI declares."""

    kwargs: dict[str, Any] = {
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "output_final_state": output_final_state,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
    }
    kwargs.update(
        build_gated_delta_rule_metadata_kwargs(
            implementation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            metadata_is_canonical=metadata_is_canonical,
        )
    )
    return kernel(query, key, value, **kwargs)


__all__ = [
    "GatedDeltaRuleMetadataCapabilities",
    "build_gated_delta_rule_metadata_kwargs",
    "call_chunk_gated_delta_rule",
    "get_gated_delta_rule_metadata_capabilities",
    "prepare_gated_delta_rule_qk",
    "requires_chunked_varlen_metadata",
]
