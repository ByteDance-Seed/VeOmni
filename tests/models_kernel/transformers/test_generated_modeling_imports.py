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

import importlib
import pathlib

import pytest

import veomni  # noqa: F401  installs the ops/attention patches the generated files expect
from veomni.utils.device import IS_NPU_AVAILABLE


_VEOMNI_ROOT = pathlib.Path(veomni.__file__).parent


def _generated_modules() -> list[str]:
    modules = []
    for path in sorted((_VEOMNI_ROOT / "models_kernel" / "transformers").glob("*/generated/patched_modeling_*.py")):
        rel = path.relative_to(_VEOMNI_ROOT.parent)
        modules.append(str(rel.with_suffix("")).replace("/", "."))
    return modules


def _patch_config_count() -> int:
    return len(list((_VEOMNI_ROOT / "models_kernel" / "transformers").glob("*/*patch_gen_config.py")))


_MODULES = _generated_modules()


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
