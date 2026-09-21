# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""CPU-runnable contract tests for ``precompute_varlen_metadata`` introduced in the
AscendC GDN precomputation path (PR #999).

The module under test pulls in ``torch_npu``, ``fla_npu``, and vendored Triton
kernels at the top level; those are mocked so the tests run on any host (CPU/GPU/Mac).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Mock NPU dependencies so ``flash_gated_delta_rule`` can be imported on CPU.
# ---------------------------------------------------------------------------


def _install_npu_mocks() -> None:
    """Pre-populate ``sys.modules`` with stubs for every NPU / Triton dep."""
    _TL = MagicMock(__path__=[], __spec__=MagicMock())
    _TRITON = MagicMock(language=_TL, __version__="3.2.0", __path__=[], __spec__=MagicMock())

    _TRITON_SUBS = [
        "triton.language.extra",
        "triton.language.extra.libdevice",
        "triton.language.extra.cann",
        "triton.language.extra.cann.extension",
        "triton.runtime",
        "triton.runtime.driver",
    ]

    _MOCKS: dict[str, object] = {
        "torch_npu": MagicMock(),
        "fla_npu": MagicMock(),
        "triton": _TRITON,
        "triton.language": _TL,
    }
    for sub in _TRITON_SUBS:
        _MOCKS[sub] = MagicMock(__path__=[], __spec__=MagicMock())

    for name, mock in _MOCKS.items():
        if name not in sys.modules:
            sys.modules[name] = mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cu_seqlens(*lengths: int) -> torch.LongTensor:
    """Build a cumulative-sequence-lengths tensor from per-sequence lengths."""
    return torch.cumsum(torch.tensor((0,) + lengths, dtype=torch.long), dim=0)


# ---------------------------------------------------------------------------
# precompute_varlen_metadata contract
# ---------------------------------------------------------------------------


class TestPrecomputeVarlenMetadataContract:
    """Verify ``precompute_varlen_metadata`` produces the same metadata as the
    per-layer ``_ensure_varlen_metadata`` fallback."""

    @pytest.fixture(scope="class")
    def module(self):
        _install_npu_mocks()
        from veomni.ops.kernels.gated_delta_rule._ascend import flash_gated_delta_rule as m

        return m

    @pytest.mark.parametrize(
        "lengths,chunk_size,num_heads",
        [
            ([128, 256], 64, 4),
            ([64, 128, 32], 64, 8),
        ],
    )
    def test_metadata_matches_ensure(
        self,
        module,
        lengths: list[int],
        chunk_size: int,
        num_heads: int,
    ) -> None:
        """``precompute_varlen_metadata`` must produce the same keys/values as
        ``_ensure_varlen_metadata`` when called with matching parameters."""
        cu_seqlens = _make_cu_seqlens(*lengths)
        cu_seqlens_list, chunk_indices, chunk_indices_list = module.precompute_varlen_metadata(
            cu_seqlens=cu_seqlens,
            num_heads=num_heads,
            chunk_size=chunk_size,
        )

        # Compare against _ensure_varlen_metadata.
        # _ensure_varlen_metadata needs a gate tensor to derive head count and
        # device info.  Shape: [B, T, H]; h = num_heads so the cumsum-block
        # computation inside both functions produces the same key set.
        B, T = 2, sum(lengths)
        g = torch.zeros(B, T, num_heads)

        _, ref_list, ref_tensor, ref_list_dict = module._ensure_varlen_metadata(
            g=g,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=None,
            chunk_indices=None,
            chunk_indices_list=None,
            chunk_size=chunk_size,
        )

        # cu_seqlens_list
        assert cu_seqlens_list == ref_list

        # Both dicts must cover the same key set
        assert set(chunk_indices.keys()) == set(ref_tensor.keys())
        assert set(chunk_indices_list.keys()) == set(ref_list_dict.keys())

        for key in chunk_indices:
            pre_t = chunk_indices[key]
            ref_t = ref_tensor[key]
            if pre_t is None:
                assert ref_t is None
            else:
                assert ref_t is not None
                assert pre_t.shape == ref_t.shape
                assert pre_t.tolist() == ref_t.tolist(), f"mismatch at key={key}"

        for key in chunk_indices_list:
            pre_l = chunk_indices_list[key]
            ref_l = ref_list_dict[key]
            if pre_l is None:
                assert ref_l is None
            else:
                assert ref_l is not None
                assert pre_l == ref_l, f"mismatch at key={key}"
