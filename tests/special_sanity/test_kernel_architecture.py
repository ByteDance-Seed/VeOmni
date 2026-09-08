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

"""Architecture guards for kernel-backed code, tests, and documentation."""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml"}
MODEL_ENTRY_ROOTS = (REPO_ROOT / "tasks", REPO_ROOT / "veomni" / "trainer")
# These integration tests still exercise model families that are not registered
# in models_kernel. Remove each exemption with that model family's migration.
RETAINED_MODEL_CONSUMER_TESTS = {
    Path("tests/data/multimodal/test_vlm_data_process.py"),
    Path("tests/distributed/test_dummy_forward.py"),
    Path("tests/distributed/test_torch_compile.py"),
    Path("tests/e2e/test_e2e_parallel.py"),
    Path("tests/lora/test_moe_lora_ep_sharded_stream_load.py"),
    Path("tests/lora/test_moe_lora_trainer.py"),
    Path("tests/lora/test_qwen3_5_moe_lora.py"),
    Path("tests/lora/test_vlm_lora.py"),
    Path("tests/lora/utils.py"),
    Path("tests/optim/test_muon_fsdp2_parity.py"),
    Path("tests/optim/test_muon_fsdp2_smoke.py"),
    Path("tests/tools/training_utils.py"),
    Path("tests/utils/test_model_loader.py"),
}


def _active_text_files() -> list[Path]:
    """Return maintained text files outside the model tree retained for comparison."""
    files: list[Path] = []
    for root_name in (
        ".agents/knowledge",
        ".agents/skills",
        ".github",
        "docs",
        "scripts",
        "tasks",
        "tests",
        "veomni",
    ):
        for path in (REPO_ROOT / root_name).rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(REPO_ROOT)
            if relative.parts[:2] in {("tests", "models"), ("veomni", "models")}:
                continue
            files.append(path)
    files.append(REPO_ROOT / "pyproject.toml")
    files.append(REPO_ROOT / ".coderabbit.yaml")
    files.append(REPO_ROOT / ".pre-commit-config.yaml")
    files.append(REPO_ROOT / "AGENTS.md")
    files.append(REPO_ROOT / "README.md")
    return sorted(set(files))


def _is_retained_model_config_reference(path: Path, line: str) -> bool:
    """Allow tooling rules that still protect the retained model comparison tree."""
    if path == REPO_ROOT / "pyproject.toml":
        return line.startswith('"veomni/' + "models/transformers/")
    if path == REPO_ROOT / ".coderabbit.yaml":
        return line.strip().startswith('- "!veomni/' + "models/transformers/")
    if path == REPO_ROOT / "tests/special_sanity/check_device_api_usage.py":
        return line.strip().startswith('"veomni/' + "models/")
    return False


def test_active_tree_uses_kernel_architecture_paths():
    """Prevent removed package paths and production model-stack imports from returning."""
    removed_kernel_package = re.compile(r"veomni[./]" + r"ops(?:[./]|\b)|tests/" + r"ops(?:/|\b)")
    model_package = re.compile(r"veomni[./]models(?:[./]|\b)")
    stale: list[str] = []
    for path in _active_text_files():
        relative = path.relative_to(REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if _is_retained_model_config_reference(path, line):
                continue
            retained_model_consumer = relative in RETAINED_MODEL_CONSUMER_TESTS and model_package.search(line)
            if removed_kernel_package.search(line) or (model_package.search(line) and not retained_model_consumer):
                stale.append(f"{relative}:{line_number}: {line.strip()}")

    assert not stale, "Stale architecture paths:\n" + "\n".join(stale)


def test_model_entry_points_forward_kernel_selection():
    """Require production model builders to receive the configured kernel selection."""
    missing: list[str] = []
    miswired: list[str] = []
    stale_keyword: list[str] = []
    for root in MODEL_ENTRY_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                is_foundation_builder = isinstance(node.func, ast.Name) and node.func.id == "build_foundation_model"
                is_minimax_builder = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "from_pretrained"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "MiniMaxH3Pipeline"
                )
                if not (is_foundation_builder or is_minimax_builder):
                    continue

                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                location = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                if "kernels_implementation" not in keywords:
                    missing.append(location)
                elif root.name == "trainer" and not ast.unparse(keywords["kernels_implementation"]).endswith(
                    ".model.ops_implementation"
                ):
                    miswired.append(location)
                if "ops_implementation" in keywords:
                    stale_keyword.append(location)

    assert not missing, "Model builders missing kernels_implementation:\n" + "\n".join(missing)
    assert not miswired, "Trainer builders not wired from model.ops_implementation:\n" + "\n".join(miswired)
    assert not stale_keyword, "Model builders using the wrong keyword:\n" + "\n".join(stale_keyword)
