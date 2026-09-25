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
# See the License for the specific language governing limitations
# under the License.

"""Documentation policy for VeOmni-owned op modules and callables.

The gate requires docs where callers depend on them: module summaries,
module-level types, public functions and methods, and the small op protocol
(``forward`` / ``backward`` / ``wrapper`` / ``__call__``). Nested closures and
ordinary ``_private`` helpers are not gated. Presence of text is not a
substitute for shape, dtype, gradient, device, or fallback contracts on those
surfaces.
"""

from __future__ import annotations

import ast
from pathlib import Path


OPS_ROOT = Path(__file__).parents[3] / "veomni" / "ops"

# Autograd and registry entries other modules call by these names.
_PROTOCOL_FUNCTION_NAMES = frozenset({"forward", "backward", "wrapper", "__call__"})


def _owned_op_modules() -> list[Path]:
    """Return op Python modules maintained directly by VeOmni."""
    return [path for path in sorted(OPS_ROOT.rglob("*.py")) if "vendor" not in path.relative_to(OPS_ROOT).parts]


def _is_public_name(name: str) -> bool:
    return not name.startswith("_")


def _requires_function_doc(name: str) -> bool:
    if name in _PROTOCOL_FUNCTION_NAMES:
        return True
    if name.startswith("__") and name.endswith("__"):
        return False
    return _is_public_name(name)


def missing_docstrings(path: Path, *, relative: str) -> list[str]:
    """Return missing-docstring sites under the public-API / protocol policy."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing: list[str] = []
    if ast.get_docstring(tree) is None:
        missing.append(str(relative))

    func_depth = 0

    def visit(node: ast.AST) -> None:
        nonlocal func_depth
        if isinstance(node, ast.ClassDef):
            if func_depth == 0 and ast.get_docstring(node) is None:
                missing.append(f"{relative}:{node.lineno} {node.name}")
            for child in ast.iter_child_nodes(node):
                visit(child)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if func_depth == 0 and _requires_function_doc(node.name) and ast.get_docstring(node) is None:
                missing.append(f"{relative}:{node.lineno} {node.name}")
            func_depth += 1
            for child in ast.iter_child_nodes(node):
                visit(child)
            func_depth -= 1
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    for child in ast.iter_child_nodes(tree):
        visit(child)
    return missing


def test_owned_op_modules_and_callables_have_docstrings():
    """Require docs on public op surfaces, not every private or nested helper."""
    missing = [
        site
        for path in _owned_op_modules()
        for site in missing_docstrings(path, relative=str(path.relative_to(OPS_ROOT.parent.parent)))
    ]
    assert not missing, "Missing op docstrings:\n" + "\n".join(missing)


def test_docstring_policy_skips_nested_and_private_helpers(tmp_path: Path):
    """Nested closures and ordinary ``_helpers`` are not gated; public API still is."""
    src = '''"""Owned op module."""

class PublicType:
    """Public type."""

    def method(self):
        """Public method."""
        def nested():
            return 1
        return nested()

    def _private_method(self):
        return 0


class _Meta:
    """Saved-state protocol."""


def public_fn():
    """Public function."""
    def nested():
        return 2
    return nested()


def _helper():
    return 3


def forward():
    """Op protocol."""
    return 4


def __init__():
    return 5
'''
    path = tmp_path / "sample.py"
    path.write_text(src, encoding="utf-8")
    assert missing_docstrings(path, relative="sample.py") == []


def test_docstring_policy_flags_public_and_protocol_gaps(tmp_path: Path):
    src = """
class PublicType:
    def method(self):
        return 1

class _Meta:
    pass

def public_fn():
    return 2

def forward():
    return 3

def _helper():
    return 4
"""
    path = tmp_path / "sample.py"
    path.write_text(src, encoding="utf-8")
    missing = missing_docstrings(path, relative="sample.py")
    assert "sample.py" in missing
    assert any(site.endswith(" PublicType") for site in missing)
    assert any(site.endswith(" _Meta") for site in missing)
    assert any(site.endswith(" method") for site in missing)
    assert any(site.endswith(" public_fn") for site in missing)
    assert any(site.endswith(" forward") for site in missing)
    assert not any(site.endswith(" _helper") for site in missing)
