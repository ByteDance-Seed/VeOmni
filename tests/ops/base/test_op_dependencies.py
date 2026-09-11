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

"""Optional-package metadata for built-in operation rows."""

from __future__ import annotations

import pytest

from veomni.ops import OP_REGISTRY


def _entries_for_impl(impl: str):
    """Return every built-in row registered with ``impl``."""
    return [entry for entry in OP_REGISTRY._entries.values() if entry.impl == impl]


@pytest.mark.parametrize(
    ("impl", "requires"),
    (
        ("liger_kernel", ("liger_kernel",)),
        ("triton", ("triton",)),
        ("fused_triton", ("triton",)),
        ("fused_quack", ("quack",)),
        ("fla", ("fla",)),
        ("flash_qla", ("flash_qla",)),
    ),
)
def test_shared_optional_implementations_declare_packages(impl: str, requires: tuple[str, ...]):
    """Rows sharing an implementation name also share its package dependency."""
    entries = _entries_for_impl(impl)
    assert entries
    assert all(entry.requires == requires for entry in entries)


@pytest.mark.parametrize(
    ("op", "variant", "impl", "device", "requires"),
    (
        ("mhc", "pre", "tilelang", "cuda", ("tile_kernels",)),
        ("mhc", "post", "tilelang", "cuda", ("tile_kernels",)),
        ("mhc", "head", "tilelang", "cuda", ("tile_kernels",)),
        ("dsa_attention", "deepseek_v4", "tilelang", "cuda", ("tilelang",)),
        ("dsa_indexer", "deepseek_v4", "tilelang", "cuda", ("tilelang",)),
        ("dsa_attention", "glm", "flashmla_cudnn", "cuda", ("cudnn", "flash_mla")),
        ("dsa_indexer", "glm", "cudnn", "cuda", ("cudnn", "flash_mla")),
        ("moe_experts", "standard", "fused_mlu", "mlu", ("apex",)),
        ("causal_conv1d", "standard", "npu", "npu", ("triton",)),
        ("chunk_gated_delta_rule", "standard", "npu", "npu", ("triton",)),
        (
            "chunk_gated_delta_rule",
            "standard",
            "npu_ascendc",
            "npu",
            ("fla_npu", "triton"),
        ),
    ),
)
def test_specialized_rows_declare_packages(op: str, variant: str, impl: str, device: str, requires: tuple[str, ...]):
    """Rows whose implementation name is ambiguous declare their exact imports."""
    assert OP_REGISTRY._entries[(op, variant, impl, device)].requires == requires
