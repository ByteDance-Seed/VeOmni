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
"""Import every HF-derived standalone modeling module.

Cheap smoke gate for transformers upgrades. Each ``modeling_<model>.py`` file is
a standalone copy of upstream modeling with VeOmni customizations, so an upstream
change can make one fail at *import* time — HF's ``@auto_docstring`` validates
patched signatures and return dataclass docstrings while the class body is
executed, and it raises rather than warns for some shapes.

The bitwise logits suite does not cover every family (some have no toy config on
the current accelerator), so importing all 19 HF-derived modeling modules
directly catches import-time regressions across the board.
"""

import ast
import importlib
import pathlib

import pytest

import veomni  # noqa: F401  installs the ops/attention patches the modeling files expect


_VEOMNI_ROOT = pathlib.Path(veomni.__file__).parent
_NON_HF_DIRS = {"flux", "movqgan", "wan"}
_EXPECTED_HF_MODEL_COUNT = 19


def _hf_modeling_modules() -> list[str]:
    modules = []
    for path in sorted((_VEOMNI_ROOT / "models" / "transformers").glob("*/modeling_*.py")):
        if path.parent.name in _NON_HF_DIRS:
            continue
        rel = path.relative_to(_VEOMNI_ROOT.parent)
        modules.append(str(rel.with_suffix("")).replace("/", "."))
    return modules


def _kernelized_functions(module_name: str) -> list[str]:
    path = _VEOMNI_ROOT.parent / f"{module_name.replace('.', '/')}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and getattr(decorator.func, "id", None) == "use_kernelized_func":
                for arg in decorator.args:
                    values = arg.elts if isinstance(arg, (ast.List, ast.Tuple)) else [arg]
                    names.extend(value.id for value in values if isinstance(value, ast.Name))
    return sorted(set(names))


_MODULES = _hf_modeling_modules()
_KERNELIZED_MODULES = [module for module in _MODULES if _kernelized_functions(module)]


def test_generated_modeling_modules_discovered():
    assert len(_MODULES) == _EXPECTED_HF_MODEL_COUNT, (
        f"found {len(_MODULES)} HF-derived modeling files, expected {_EXPECTED_HF_MODEL_COUNT}. Discovered: {_MODULES}"
    )


@pytest.mark.parametrize("module_name", _KERNELIZED_MODULES, ids=lambda name: name.rsplit(".", 1)[-1])
def test_kernelized_functions_keep_hub_decorators(module_name: str):
    path = _VEOMNI_ROOT.parent / f"{module_name.replace('.', '/')}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    for name in _kernelized_functions(module_name):
        decorators = [ast.unparse(decorator) for decorator in functions[name].decorator_list]
        assert any("use_kernel_" in decorator for decorator in decorators), (
            f"{module_name}.{name} is passed to @use_kernelized_func without a Hub kernel decorator"
        )


@pytest.mark.parametrize("module_name", _KERNELIZED_MODULES, ids=lambda name: name.rsplit(".", 1)[-1])
def test_kernelized_modeling_imports_with_real_kernels(module_name: str):
    pytest.importorskip("kernels")
    hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
    if not hub_kernels._kernels_enabled:
        pytest.skip("Transformers Hub kernel decorators are disabled by USE_HUB_KERNELS")

    module = importlib.import_module(module_name)

    for name in _kernelized_functions(module_name):
        assert hasattr(getattr(module, name), "kernel_layer_name"), f"{module_name}.{name}"


@pytest.mark.parametrize("module_name", _MODULES, ids=lambda name: name.rsplit(".", 1)[-1])
def test_generated_modeling_imports(module_name: str):
    importlib.import_module(module_name)
