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

"""Import-isolation tests for the operation registry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]


def test_import_veomni_without_optional_kernel_packages() -> None:
    """Public op packages import without loading any optional kernel backend."""
    script = """
import builtins
import sys

real_import = builtins.__import__
optional_roots = {
    "apex",
    "cuda",
    "cudnn",
    "debugpy",
    "fla",
    "fla_npu",
    "flash_attn",
    "flash_attn_cute",
    "flash_attn_interface",
    "flash_mla",
    "flash_qla",
    "liger_kernel",
    "magi_attention",
    "quack",
    "sageattention",
    "tile_kernels",
    "tilelang",
    "triton",
}


def blocked_import(name, *args, **kwargs):
    if name.partition(".")[0] in optional_roots:
        raise ModuleNotFoundError(f"No module named {name!r} (blocked by test)", name=name)
    return real_import(name, *args, **kwargs)


builtins.__import__ = blocked_import

import veomni
import veomni.ops.batch_invariant
import veomni.ops.kernels.dsa.sparse_mqa_target
import veomni.ops.kernels.dsa.vendor
import veomni.ops.qat
from veomni.ops import OP_REGISTRY

assert not optional_roots.intersection(sys.modules)
for module in (
    "veomni.ops.batch_invariant.triton",
    "veomni.ops.kernels.dsa.vendor.flashmla_cudnn",
    "veomni.ops.kernels.dsa.vendor.tilelang_indexer",
    "veomni.ops.kernels.dsa.vendor.tilelang_sparse_mla",
    "veomni.ops.kernels.dsa.vendor.tilelang_sparse_mla_target",
    "veomni.ops.kernels.moe_experts_lora.shared.triton",
    "veomni.ops.kernels.moe_experts_lora.independent.triton",
    "veomni.ops.qat.quant",
):
    assert module not in sys.modules
for variant in ("shared", "independent"):
    assert "fused_triton" in OP_REGISTRY.list_registered("moe_experts_lora", variant)
    assert "fused_npu" in OP_REGISTRY.list_registered("moe_experts_lora", variant)
"""

    subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, check=True)
