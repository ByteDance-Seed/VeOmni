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

"""SDPA backend and SP-aware adapter implementation."""

from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward as hf_sdpa_attention_forward

from .....distributed.parallel_state import get_parallel_state
from ..helper import reject_sdpa_packed_metadata
from ..ulysses import (
    prepare_ulysses_qkv,
    restore_ulysses_output,
    should_apply_ulysses,
)


# Flash / cuDNN drop dense masks. This kernel exists for mask + Ulysses, so pin
# memory-efficient first. MATH stays as the CPU / unsupported-shape fallback.
_SDPA_MASK_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class _LocalGQAModuleView:
    """Expose post-Ulysses GQA metadata without mutating the shared module."""

    __slots__ = ("_module", "num_key_value_groups")

    def __init__(self, module: torch.nn.Module, num_key_value_groups: int) -> None:
        """Store the shared module and expose the local GQA ratio."""
        self._module = module
        self.num_key_value_groups = num_key_value_groups

    def __getattr__(self, name: str):
        """Forward all metadata except the overridden GQA ratio."""
        return getattr(self._module, name)


def _slice_ulysses_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    query_head_count: int,
    local_query_head_count: int,
    group: dist.ProcessGroup,
) -> torch.Tensor | None:
    """Select this rank's query-head mask slice after Ulysses head scattering."""
    if attention_mask is None or attention_mask.ndim < 3:
        return attention_mask

    mask_head_count = attention_mask.shape[-3]
    if mask_head_count in (1, local_query_head_count):
        return attention_mask
    if mask_head_count != query_head_count:
        raise ValueError(
            "SDPA with Ulysses requires the attention-mask head dimension to be 1, "
            f"the local query-head count ({local_query_head_count}), or the global query-head count "
            f"({query_head_count}); got {mask_head_count}."
        )

    head_start = dist.get_rank(group) * local_query_head_count
    return attention_mask.narrow(-3, head_start, local_query_head_count)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    skip_ulysses: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run Transformers SDPA with optional Ulysses exchange.

    ``sliding_window`` is shared-signature metadata and is not forwarded. This
    row supports windowed visibility only when it is already encoded in
    ``attention_mask``. Packed/varlen metadata is rejected because the SDPA
    API has no cumulative-length arguments. ``softcap`` changes logits rather
    than visibility, so a mask cannot encode it; this row rejects explicit
    softcapping rather than silently changing attention semantics.

    Uses memory-efficient SDPA so a dense bool / additive mask stays valid.
    Flash is not tried. Use ``veomni_flash_attention_*`` when the pattern can
    stay in attention kwargs.

    ``skip_ulysses`` opts a call out of sync Ulysses when its tokens are not
    on the SP mesh. Async Ulysses stays outside attention.
    """
    reject_sdpa_packed_metadata(kwargs)
    del sliding_window

    if softcap is not None:
        raise ValueError("veomni_sdpa does not support softcap.")
    if kwargs.get("s_aux") is not None:
        raise ValueError(
            "veomni_sdpa does not implement attention sinks (`s_aux`). "
            "Use veomni_flash_attention_4 or a backend that implements sink-softmax."
        )

    if any(dim == 0 for tensor in (query, key, value) for dim in tensor.shape):
        raise ValueError("SDPA does not support query/key/value tensors with zero dimensions.")

    parallel_state = get_parallel_state()
    ulysses_enabled = should_apply_ulysses(skip_ulysses=skip_ulysses)
    backend_module = module
    if ulysses_enabled:
        query, key, value, query_head_count = prepare_ulysses_qkv(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if key.shape[1] != value.shape[1] or query.shape[1] % key.shape[1] != 0:
            raise ValueError(
                "Post-Ulysses query heads must be divisible by matching key/value heads; "
                f"got query={query.shape[1]}, key={key.shape[1]}, value={value.shape[1]}."
            )
        local_key_value_groups = query.shape[1] // key.shape[1]
        if getattr(module, "num_key_value_groups", 1) != local_key_value_groups:
            backend_module = _LocalGQAModuleView(module, local_key_value_groups)
        attention_mask = _slice_ulysses_attention_mask(
            attention_mask,
            query_head_count=query_head_count,
            local_query_head_count=query.shape[1],
            group=parallel_state.ulysses_group,
        )
    with sdpa_kernel(_SDPA_MASK_BACKENDS):
        output, lse = hf_sdpa_attention_forward(
            backend_module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    if ulysses_enabled:
        output = restore_ulysses_output(output, group=parallel_state.ulysses_group)

    return output, lse
