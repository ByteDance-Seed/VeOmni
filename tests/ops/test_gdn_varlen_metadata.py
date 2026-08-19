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

"""CPU-runnable contracts for lightweight GDN varlen metadata precomputation."""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cu_seqlens(*lengths: int) -> torch.LongTensor:
    """Build a cumulative-sequence-lengths tensor from per-sequence lengths."""
    return torch.cumsum(torch.tensor((0,) + lengths, dtype=torch.long), dim=0)


def _load_lightweight_module():
    path = Path(__file__).parents[2] / "veomni/ops/kernels/gated_delta_rule/varlen_metadata.py"
    spec = importlib.util.spec_from_file_location("_test_gdn_varlen_metadata", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_pairs(lengths: list[int], chunk_size: int) -> list[list[int]]:
    return [
        [seq_idx, chunk_idx]
        for seq_idx, length in enumerate(lengths)
        for chunk_idx in range((length + chunk_size - 1) // chunk_size)
    ]


def test_lightweight_module_does_not_import_ascendc_dependencies() -> None:
    code = """
import importlib.machinery
import sys
from types import ModuleType, SimpleNamespace

import torch

torch_npu = ModuleType("torch_npu")
torch_npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
sys.modules["torch_npu"] = torch_npu
torch.npu = SimpleNamespace(config=SimpleNamespace(allow_internal_format=False), is_available=lambda: False)

from veomni.ops.kernels.gated_delta_rule.varlen_metadata import precompute_varlen_metadata

metadata = precompute_varlen_metadata(torch.tensor([0, 64, 192]), num_heads=4)
assert metadata[0] == [0, 64, 192]
assert "fla_npu" not in sys.modules
assert "veomni.ops.kernels.gated_delta_rule._ascend.flash_gated_delta_rule" not in sys.modules
"""
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_npu.py",
        "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py",
    ],
)
def test_generated_npu_models_import_lightweight_metadata(relative_path: str) -> None:
    source = (Path(__file__).parents[2] / relative_path).read_text()
    lightweight_import = "from veomni.ops.kernels.gated_delta_rule.varlen_metadata import precompute_varlen_metadata"
    assert lightweight_import in source
    assert (
        "from veomni.ops.kernels.gated_delta_rule._ascend.flash_gated_delta_rule import precompute_varlen_metadata"
        not in source
    )


# ---------------------------------------------------------------------------
# precompute_varlen_metadata contract
# ---------------------------------------------------------------------------


class TestPrecomputeVarlenMetadataContract:
    """Verify the backend-neutral metadata contract independently."""

    @pytest.fixture(scope="class")
    def module(self):
        return _load_lightweight_module()

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
        """Tensor and host-list metadata must encode identical chunk ordinals."""
        cu_seqlens = _make_cu_seqlens(*lengths)
        cu_seqlens_list, chunk_indices, chunk_indices_list = module.precompute_varlen_metadata(
            cu_seqlens=cu_seqlens,
            num_heads=num_heads,
            chunk_size=chunk_size,
        )

        cumsum_block_size = 1 << (max(1, (1 << 17) // (num_heads * chunk_size)) - 1).bit_length()
        expected_sizes = {16, 32, 64, 128, 608 * 2, chunk_size, cumsum_block_size}

        assert cu_seqlens_list == torch.cumsum(torch.tensor([0, *lengths]), dim=0).tolist()
        assert set(chunk_indices) == {str(size) for size in expected_sizes}
        assert set(chunk_indices_list) == {str(size) for size in expected_sizes}
        for size in expected_sizes:
            expected = _expected_pairs(lengths, size)
            assert chunk_indices[str(size)].tolist() == expected
            assert chunk_indices_list[str(size)] == [item for pair in expected for item in pair]

    @pytest.mark.parametrize(
        ("lengths", "expected"),
        [
            ([0, 64], [[1, 0]]),
            ([0, 0, 128], [[2, 0], [2, 1]]),
            ([0, 0], []),
        ],
    )
    def test_empty_samples_preserve_ordinal_and_all_empty_is_well_formed(
        self, module, lengths: list[int], expected: list[list[int]]
    ) -> None:
        indices = module.prepare_chunk_indices(_make_cu_seqlens(*lengths), chunk_size=64)
        assert indices.shape == (len(expected), 2)
        if expected:
            assert indices.tolist() == expected
        else:
            assert indices.numel() == 0

    def test_tensor_and_host_metadata_keep_empty_sample_ordinal_in_sync(self, module) -> None:
        cu = _make_cu_seqlens(0, 64, 0, 128)
        _, tensor_indices, list_indices = module.precompute_varlen_metadata(cu, num_heads=4, chunk_size=64)
        assert tensor_indices["64"].tolist() == [[1, 0], [3, 0], [3, 1]]
        assert list_indices["64"] == [1, 0, 3, 0, 3, 1]


def test_ascendc_module_compatibly_reexports_metadata_helpers() -> None:
    source = (
        Path(__file__).parents[2] / "veomni/ops/kernels/gated_delta_rule/_ascend/flash_gated_delta_rule.py"
    ).read_text()
    assert "from ..varlen_metadata import (" in source
    for name in ("precompute_varlen_metadata", "prepare_chunk_indices", "prepare_chunk_indices_list"):
        assert name in source


def test_ascendc_fla_npu_import_is_compute_time_only() -> None:
    path = Path(__file__).parents[2] / "veomni/ops/kernels/gated_delta_rule/_ascend/flash_gated_delta_rule.py"
    tree = ast.parse(path.read_text())

    top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all(not any(alias.name == "fla_npu" for alias in node.names) for node in top_level_imports)

    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    ensure = functions["_ensure_fla_npu"]
    assert any(
        isinstance(node, ast.Import) and any(alias.name == "fla_npu" for alias in node.names)
        for node in ast.walk(ensure)
    )
    for name in ("flash_chunk_gated_delta_rule_fwd", "flash_chunk_gated_delta_rule_bwd", "flash_gated_delta_rule"):
        assert any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_ensure_fla_npu"
            for node in ast.walk(functions[name])
        )
