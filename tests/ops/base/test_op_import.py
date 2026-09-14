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
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

from veomni.utils.import_utils import is_package_available


REPO_ROOT = Path(__file__).parents[3]


def test_import_veomni_without_optional_kernel_packages() -> None:
    """Public op packages import without loading any optional kernel backend."""
    script = """
import builtins
import sys

real_import = builtins.__import__
attempted_optional_imports = []
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
        attempted_optional_imports.append(name)
        raise ModuleNotFoundError(f"No module named {name!r} (blocked by test)", name=name)
    return real_import(name, *args, **kwargs)


builtins.__import__ = blocked_import

import veomni
import veomni.ops.batch_invariant
import veomni.ops.kernels.dsa.sparse_mqa_target
import veomni.ops.kernels.dsa.vendor
import veomni.ops.qat
from veomni.ops import OP_REGISTRY

assert "sageattention" not in {name.partition(".")[0] for name in attempted_optional_imports}
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


def test_dotted_requirement_discovery_does_not_import_parent(tmp_path: Path) -> None:
    """Availability checks find a child module without executing its parent package."""
    package = tmp_path / "review_optional"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('parent package executed')\n", encoding="utf-8")
    (package / "child.py").write_text("VALUE = 1\n", encoding="utf-8")
    script = f"""
import sys

sys.path.insert(0, {str(tmp_path)!r})

from veomni.ops.registry import OpEntry, OpRegistry

registry = OpRegistry()
entry = OpEntry(
    op="probe",
    variant="standard",
    impl="dotted",
    description="Dotted optional-module discovery probe",
    wrapper=lambda value: value,
    requires=("review_optional.child",),
)
registry.register(entry)
registry.register(
    OpEntry(
        op="probe",
        variant="standard",
        impl="missing",
        description="Missing dotted optional-module discovery probe",
        wrapper=lambda value: value,
        requires=("review_optional.missing",),
    )
)

assert registry.list_available("probe", "standard") == ["dotted"]
assert registry.resolve("probe", "standard", "dotted") is entry
try:
    registry.resolve("probe", "standard", "missing")
except RuntimeError as exc:
    assert "review_optional.missing" in str(exc)
else:
    raise AssertionError("missing dotted requirement resolved")
assert "review_optional" not in sys.modules
assert "review_optional.child" not in sys.modules
"""

    subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, check=True)


def test_dotted_requirement_discovery_supports_namespace_and_deep_paths(tmp_path: Path, monkeypatch) -> None:
    """Namespace and intermediate packages remain unexecuted during deep discovery."""
    package = tmp_path / "review_namespace" / "middle"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('middle package executed')\n", encoding="utf-8")
    (package / "child.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert is_package_available("review_namespace.middle.child")
    assert "review_namespace" not in sys.modules
    assert "review_namespace.middle" not in sys.modules


def test_dotted_requirement_discovery_supports_consecutive_namespace_packages(tmp_path: Path, monkeypatch) -> None:
    """Nested namespace packages are discoverable without synthetic imports."""
    from veomni.ops.registry import OpEntry, OpRegistry

    package = tmp_path / "review_nested_namespace" / "middle"
    package.mkdir(parents=True)
    (package / "child.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = OpRegistry()
    entry = OpEntry(
        op="probe",
        variant="standard",
        impl="namespace",
        description="Nested namespace requirement probe",
        wrapper=lambda value: value,
        requires=("review_nested_namespace.middle.child",),
    )
    registry.register(entry)

    assert is_package_available("review_nested_namespace.middle.child")
    assert registry.list_available("probe", "standard") == ["namespace"]
    assert registry.resolve("probe", "standard", "namespace") is entry
    assert "review_nested_namespace" not in sys.modules
    assert "review_nested_namespace.middle" not in sys.modules
    assert "review_nested_namespace.middle.child" not in sys.modules


def test_dotted_requirement_discovery_uses_loaded_parent_path(tmp_path: Path, monkeypatch) -> None:
    """An already-loaded package contributes its runtime search path without reimport."""
    package = tmp_path / "review_loaded"
    package.mkdir()
    (package / "child.py").write_text("VALUE = 1\n", encoding="utf-8")
    loaded_parent = ModuleType("review_loaded")
    loaded_parent.__path__ = [str(package)]
    monkeypatch.setitem(sys.modules, "review_loaded", loaded_parent)

    assert is_package_available("review_loaded.child")
    assert "review_loaded.child" not in sys.modules


def test_dotted_requirement_discovery_rejects_nonpackage_and_failed_parents(tmp_path: Path, monkeypatch) -> None:
    """Loaded modules without paths and failed-import sentinels cannot have children."""
    monkeypatch.setitem(sys.modules, "review_plain", ModuleType("review_plain"))
    assert not is_package_available("review_plain.child")

    package = tmp_path / "review_failed"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "review_failed", None)
    assert not is_package_available("review_failed")
    assert not is_package_available("review_failed.child")


def test_dotted_requirement_discovery_uses_custom_meta_path_finders(monkeypatch) -> None:
    """Child modules exposed only by a custom meta-path finder remain discoverable."""

    class ReviewFinder:
        def find_spec(self, fullname, path, target):
            assert target is None
            if fullname == "review_meta":
                return ModuleSpec(fullname, loader=None, is_package=True)
            if fullname == "review_meta.child":
                return ModuleSpec(fullname, loader=None)
            return None

    monkeypatch.setattr(sys, "meta_path", [ReviewFinder(), *sys.meta_path])

    assert is_package_available("review_meta.child")
    assert "review_meta" not in sys.modules
