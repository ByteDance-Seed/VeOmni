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

"""Documentation policy for VeOmni-owned kernel modules and callables."""

from __future__ import annotations

import ast
from pathlib import Path


KERNELS_ROOT = Path(__file__).parents[3] / "veomni" / "kernels"


def _owned_kernel_modules() -> list[Path]:
    """Return kernel Python modules maintained directly by VeOmni."""
    return [
        path for path in sorted(KERNELS_ROOT.rglob("*.py")) if "vendor" not in path.relative_to(KERNELS_ROOT).parts
    ]


def test_owned_kernel_modules_and_callables_have_docstrings():
    """Require docs on modules, classes, and functions, including private helpers."""
    missing: list[str] = []
    for path in _owned_kernel_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(KERNELS_ROOT.parent.parent)
        if ast.get_docstring(tree) is None:
            missing.append(str(relative))
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and ast.get_docstring(node) is None
            ):
                missing.append(f"{relative}:{node.lineno} {node.name}")

    assert not missing, "Missing kernel docstrings:\n" + "\n".join(missing)
