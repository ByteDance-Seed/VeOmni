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


def test_import_veomni_without_triton() -> None:
    """Register every op family without importing the optional Triton package."""
    script = """
import builtins
import sys

real_import = builtins.__import__


def blocked_import(name, *args, **kwargs):
    if name == "triton" or name.startswith("triton."):
        raise ModuleNotFoundError("No module named 'triton' (blocked by test)", name=name)
    return real_import(name, *args, **kwargs)


builtins.__import__ = blocked_import

import veomni
from veomni.ops import OP_REGISTRY

assert "triton" not in sys.modules
assert "veomni.ops.kernels.moe_experts_lora.shared.triton" not in sys.modules
assert "veomni.ops.kernels.moe_experts_lora.independent.triton" not in sys.modules
for variant in ("shared", "independent"):
    assert "fused_triton" in OP_REGISTRY.list_registered("moe_experts_lora", variant)
    assert "fused_npu" in OP_REGISTRY.list_registered("moe_experts_lora", variant)
"""

    subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, check=True)
