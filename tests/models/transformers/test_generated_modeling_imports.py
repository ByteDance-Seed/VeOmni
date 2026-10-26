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
"""Import every patchgen-generated modeling module.

Cheap smoke gate for transformers upgrades. The generated files are full copies
of upstream modeling with VeOmni patches spliced in, so an upstream change can
make one fail at *import* time — HF's ``@auto_docstring`` validates patched
signatures and return dataclass docstrings while the class body is executed, and
it raises rather than warns for some shapes.

The bitwise logits suite does not cover this: it only builds the GPU models it
has toy configs for, so an NPU-only generated file (or a family with no toy
config) can be broken without any test noticing. The transformers 5.9 -> 5.16
bump shipped exactly that failure in ``patched_modeling_qwen3_5_npu.py``.
"""

import ast
import importlib
import pathlib

import pytest

import veomni  # noqa: F401  installs the ops/attention patches the generated files expect
from veomni.utils.device import IS_NPU_AVAILABLE


_VEOMNI_ROOT = pathlib.Path(veomni.__file__).parent


def _generated_modules() -> list[str]:
    modules = []
    for path in sorted((_VEOMNI_ROOT / "models" / "transformers").glob("*/generated/patched_modeling_*.py")):
        rel = path.relative_to(_VEOMNI_ROOT.parent)
        modules.append(str(rel.with_suffix("")).replace("/", "."))
    return modules


def _patch_config_count() -> int:
    return len(list((_VEOMNI_ROOT / "models" / "transformers").glob("*/*patch_gen_config.py")))


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


def _import_generated_module(module_name: str):
    if module_name.endswith("_gpu") and IS_NPU_AVAILABLE:
        pytest.skip("GPU modeling may depend on CUDA-only packages absent from the NPU environment")
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        if module_name.endswith("_npu") and not IS_NPU_AVAILABLE and "torch_npu" in str(exc):
            pytest.skip(f"{module_name} needs torch_npu at import time")
        raise


_MODULES = _generated_modules()
_KERNELIZED_MODULES = [module for module in _MODULES if _kernelized_functions(module)]


def test_generated_modeling_modules_discovered():
    # Every patch config emits exactly one generated modeling file, so the two
    # counts must agree. Deriving the expectation this way means a family that
    # stops being generated fails here instead of silently dropping out of the
    # parametrisation below.
    expected = _patch_config_count()
    assert len(_MODULES) == expected, (
        f"found {len(_MODULES)} generated modeling files for {expected} patch configs; "
        f"run `make patchgen`. Discovered: {_MODULES}"
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

    module = _import_generated_module(module_name)

    for name in _kernelized_functions(module_name):
        assert hasattr(getattr(module, name), "kernel_layer_name"), f"{module_name}.{name}"


@pytest.mark.parametrize("module_name", _MODULES, ids=lambda name: name.rsplit(".", 1)[-1])
def test_generated_modeling_imports(module_name: str):
    if module_name.endswith("_gpu") and IS_NPU_AVAILABLE:
        pytest.skip("GPU modeling may depend on CUDA-only packages absent from the NPU environment")
    if module_name.endswith("_npu") and not IS_NPU_AVAILABLE:
        # Most NPU files import fine on GPU hosts (the device split lives inside
        # the patched bodies), but a few pull ``torch_npu`` at module scope.
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            if "torch_npu" in str(exc):
                pytest.skip(f"{module_name} needs torch_npu at import time")
            raise
        return

    importlib.import_module(module_name)
