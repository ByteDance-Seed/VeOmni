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

"""Keep device-specific math out of model files."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


REPO_ROOT = Path(__file__).parents[2]
_BRANCH = re.compile(
    r"IS_NPU_AVAILABLE"
    r"|import\s+torch_npu\b"
    r"|get_device_type\(\)\s*==\s*[\"']npu[\"']"
    r"|(?:device_type|\.type)\s*[!=]=\s*[\"']npu[\"']"
)
_ALLOWED = {
    (
        "veomni/models/diffusers/qwen_image/qwen_image_transformer/modeling_qwen_image_transformer.py",
        "qwen_image_rotary_frequencies",
    ),
    (
        "veomni/models/diffusers/qwen_image/qwen_image_transformer/modeling_qwen_image_transformer.py",
        "_prepare_qwen_attention_mask",
    ),
    (
        "veomni/models/diffusers/hunyuanvideo15/hunyuanvideo15_transformer/native/modules/attention.py",
        "attention",
    ),
}


def _model_files() -> list[Path]:
    root = REPO_ROOT / "veomni" / "models"
    files = list(root.rglob("modeling_*.py"))
    files.append(root / "diffusers/hunyuanvideo15/hunyuanvideo15_transformer/native/modules/attention.py")
    return sorted(set(files))


def _enclosing_function(tree: ast.AST, lineno: int) -> str:
    """Return the innermost function whose body covers ``lineno``."""
    match: ast.AST | None = None
    span = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = node.end_lineno or node.lineno
        if node.lineno <= lineno <= end and (span is None or end - node.lineno < span):
            match = node
            span = end - node.lineno
    if isinstance(match, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return match.name
    return "<module>"


def test_modeling_device_branches_stay_on_the_allowlist() -> None:
    """New NPU branches in model files must move into ops or an explicit allowlist entry."""
    found: set[tuple[str, str]] = set()
    unexpected: list[str] = []
    for path in _model_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for lineno, line in enumerate(source.splitlines(), start=1):
            if _BRANCH.search(line) is None:
                continue
            function = _enclosing_function(tree, lineno)
            found.add((relative, function))
            if (relative, function) not in _ALLOWED:
                unexpected.append(f"{relative}:{lineno} {function}: {line.strip()}")

    assert unexpected == []
    assert found == _ALLOWED
