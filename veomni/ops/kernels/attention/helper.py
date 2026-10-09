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

"""Shared helpers for attention adapters and mask builders."""

import torch


_SDPA_CU_SEQLENS_KEYS = ("cu_seqlens", "cu_seqlens_q", "cu_seqlens_k", "cu_seq_lens_q", "cu_seq_lens_k")
_SDPA_PACKED_METADATA_KEYS = (*_SDPA_CU_SEQLENS_KEYS, "max_length_q", "max_length_k", "max_seqlen_q", "max_seqlen_k")


def strip_sdpa_packed_metadata(
    kwargs: dict,
    *,
    batch_size: int,
    dense_mask: torch.Tensor | None,
) -> dict:
    """Drop packed/varlen arguments absent from the SDPA API.

    The collator always emits this metadata. It is dropped when it describes
    at most one segment per batch row, or when ``dense_mask`` is a 3-D/4-D
    mask, which the caller built to isolate the segments. Several segments in
    a row without such a mask would attend across samples, so that raises.
    Dense mask contents are not inspected.
    """
    packed = [name for name in _SDPA_PACKED_METADATA_KEYS if kwargs.get(name) is not None]
    if not packed:
        return kwargs
    multi_segment = [
        name
        for name in _SDPA_CU_SEQLENS_KEYS
        if torch.is_tensor(kwargs.get(name)) and kwargs[name].numel() - 1 > batch_size
    ]
    if multi_segment and (dense_mask is None or dense_mask.ndim < 3):
        raise ValueError(
            "SDPA received packed metadata with several segments per row ("
            + ", ".join(multi_segment)
            + ") but no dense attention mask that isolates them. Build one with "
            "packed_causal_mask or use a packed-capable attention implementation."
        )
    return {key: value for key, value in kwargs.items() if key not in packed}


def require_all(condition: torch.Tensor, message: str) -> None:
    """Require every element to be true without synchronizing CUDA to Python.

    ``torch._assert_async`` is private but supported by the pinned PyTorch 2.10
    and 2.11 releases. On CUDA, a failed assertion surfaces at a later kernel
    launch and invalidates the process's CUDA context instead of raising the
    catchable ``ValueError`` used by the CPU branch.
    """
    all_true = condition.all()
    if all_true.device.type == "cpu":
        if not bool(all_true):
            raise ValueError(message)
        return

    torch._assert_async(all_true, message)
