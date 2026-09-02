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


# ---------------------------------------------------------------------------
# precompute_varlen_metadata is importable/usable without fla_npu
# ---------------------------------------------------------------------------


class TestPrecomputeVarlenMetadataWithoutFlaNpu:
    """Contract: ``flash_gated_delta_rule`` must import (and its fla_npu-free
    helpers must run) on hosts that do not ship ``fla_npu``.

    This locks in the fix for the Qwen3.5 patchgen-generated NPU forward, which
    imports ``precompute_varlen_metadata`` unconditionally regardless of the
    selected ``chunk_gated_delta_rule_implementation`` — a module-level
    ``import fla_npu`` in ``flash_gated_delta_rule`` would break every non-
    ``npu_ascendc`` NPU training config (all Qwen3.5 NPU configs today).

    Runs in a subprocess because the sibling class above installs a MagicMock
    stub for ``fla_npu`` in the parent interpreter's ``sys.modules``; that
    stub would mask a regression here otherwise.
    """

    def test_import_and_precompute_without_fla_npu(self) -> None:
        import subprocess
        import sys as _sys
        import textwrap

        script = textwrap.dedent(
            """
            import importlib.machinery
            import sys
            from unittest.mock import MagicMock

            def _stub(name):
                m = MagicMock(__path__=[])
                # Give the module a real ModuleSpec so importlib.util.find_spec
                # (used by accelerate.is_npu_available and friends) doesn't
                # raise "X.__spec__ is not set" on a bare MagicMock spec.
                m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
                return m

            # Match the mocks in TestPrecomputeVarlenMetadataContract EXCEPT
            # fla_npu — this test asserts that the module loads without it.
            _TL = _stub("triton.language")
            _TRITON = _stub("triton")
            _TRITON.language = _TL
            _TRITON.__version__ = "3.2.0"
            sys.modules.setdefault("torch_npu", _stub("torch_npu"))
            sys.modules.setdefault("triton", _TRITON)
            sys.modules.setdefault("triton.language", _TL)
            for sub in (
                "triton.language.extra",
                "triton.language.extra.libdevice",
                "triton.language.extra.cann",
                "triton.language.extra.cann.extension",
                "triton.runtime",
                "triton.runtime.driver",
            ):
                sys.modules.setdefault(sub, _stub(sub))

            assert "fla_npu" not in sys.modules, "fla_npu must not be pre-installed"

            from veomni.ops.kernels.gated_delta_rule._ascend.flash_gated_delta_rule import (
                precompute_varlen_metadata,
            )

            # Import alone must not have pulled in fla_npu (top-level import is
            # deferred to the fwd/bwd dispatch sites).
            assert "fla_npu" not in sys.modules, (
                "flash_gated_delta_rule must not import fla_npu at module load time"
            )

            import torch

            cu = torch.cumsum(torch.tensor([0, 128, 256], dtype=torch.long), dim=0)
            cu_list, ci, cil = precompute_varlen_metadata(
                cu_seqlens=cu, num_heads=4, chunk_size=64
            )
            assert cu_list == cu.tolist()
            assert isinstance(ci, dict) and isinstance(cil, dict)
            print("OK")
            """
        )

        result = subprocess.run(
            [_sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"subprocess exited {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
        assert result.stdout.strip().endswith("OK")
