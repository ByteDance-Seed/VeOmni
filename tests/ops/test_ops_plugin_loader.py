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

"""CPU-runnable tests for the external ops-backend plugin loader.

No accelerator and no real plugin package required: entry points are injected
explicitly and point at the ``plugin_fixtures`` declarations next to this file.

Run on any host with ``pytest tests/ops/test_ops_plugin_loader.py``.
"""

import sys
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace

import pytest

import veomni.ops  # noqa: F401 -- trigger built-in registrations + plugin mount
from veomni.ops.config import registry as registry_mod
from veomni.ops.config._plugin_loader import PLUGIN_GROUP, load_ops_backend_plugins
from veomni.ops.config.registry import get_op
from veomni.ops.config.singleton import get_ops_config, set_ops_config
from veomni.ops.kernel_registry import KERNEL_REGISTRY


# Make ``plugin_fixtures.*`` importable (tests/ops is not a package).
_TEST_OPS_DIR = Path(__file__).resolve().parent
if str(_TEST_OPS_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_OPS_DIR))


def _ep(name: str, module: str) -> EntryPoint:
    return EntryPoint(name=name, value=f"plugin_fixtures.{module}", group=PLUGIN_GROUP)


@pytest.fixture
def restore_registries():
    """Snapshot and restore both registries + the ops-config singleton.

    Plugin loading mutates global state by design; tests must not leak
    ``testkit`` backends into other test modules.
    """
    ops_snapshot = dict(registry_mod._OPS_REGISTRY)
    kernels_snapshot = {k: dict(v) for k, v in KERNEL_REGISTRY._specs.items()}
    config_snapshot = get_ops_config()
    yield
    registry_mod._OPS_REGISTRY.clear()
    registry_mod._OPS_REGISTRY.update(ops_snapshot)
    KERNEL_REGISTRY._specs.clear()
    KERNEL_REGISTRY._specs.update(kernels_snapshot)
    set_ops_config(config_snapshot)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.usefixtures("restore_registries")
    def test_plugin_registers_in_both_registries(self):
        loaded = load_ops_backend_plugins([_ep("good", "good")])
        assert loaded == ("good",)

        op = get_op("rms_norm")
        assert "testkit" in op.backends
        assert op.backends["testkit"].entry == "plugin_fixtures.kernels:FakeRMSNorm"
        assert op.default != "testkit"  # opt-in only

        assert KERNEL_REGISTRY.resolve("rms_norm", "standard", "testkit") is not None

    @pytest.mark.usefixtures("restore_registries")
    def test_plugin_backend_listed_in_error_message(self):
        load_ops_backend_plugins([_ep("good", "good")])
        set_ops_config(SimpleNamespace(rms_norm_implementation="typo_backend"))

        from veomni.ops.config.registry import apply_per_model_patches

        with pytest.raises(ValueError, match="testkit"):
            apply_per_model_patches(
                SimpleNamespace(FakeRMSNorm=object), "TestModel", targets={"rms_norm": "FakeRMSNorm"}
            )

    @pytest.mark.usefixtures("restore_registries")
    def test_per_model_end_to_end_bind_and_eager(self):
        from plugin_fixtures.kernels import FakeRMSNorm

        load_ops_backend_plugins([_ep("good", "good")])
        from veomni.ops.config.registry import apply_per_model_patches

        hf_module = SimpleNamespace(FakeRMSNorm="original-class")
        set_ops_config(SimpleNamespace(rms_norm_implementation="testkit"))
        apply_per_model_patches(hf_module, "TestModel", targets={"rms_norm": "FakeRMSNorm"})
        assert hf_module.FakeRMSNorm is FakeRMSNorm

        hf_module2 = SimpleNamespace(FakeRMSNorm="original-class")
        set_ops_config(SimpleNamespace(rms_norm_implementation="eager"))
        apply_per_model_patches(hf_module2, "TestModel", targets={"rms_norm": "FakeRMSNorm"})
        assert hf_module2.FakeRMSNorm == "original-class"  # eager keeps the HF default


# ---------------------------------------------------------------------------
# Rejection matrix: every failure mode must leave zero registration behind
# ---------------------------------------------------------------------------


class TestRejections:
    @pytest.mark.usefixtures("restore_registries")
    @pytest.mark.parametrize("module", ["bad_version", "missing_payload", "unknown_key", "unknown_op"])
    def test_malformed_payloads_rejected(self, module):
        assert load_ops_backend_plugins([_ep(module, module)]) == ()
        assert "testkit" not in get_op("rms_norm").backends

    @pytest.mark.usefixtures("restore_registries")
    def test_backend_name_conflict_rejected_builtin_untouched(self):
        assert load_ops_backend_plugins([_ep("conflict", "backend_conflict")]) == ()
        op = get_op("rms_norm")
        assert "liger_kernel" in op.backends
        assert op.backends["liger_kernel"].entry.startswith("liger_kernel")

    @pytest.mark.usefixtures("restore_registries")
    def test_kernel_name_conflict_rejected(self):
        load_ops_backend_plugins([_ep("good", "good")])
        assert load_ops_backend_plugins([_ep("conflict", "kernel_conflict")]) == ()
        # First plugin's kernel still resolves; the duplicate did not land.
        assert KERNEL_REGISTRY.resolve("rms_norm", "standard", "testkit") is not None

    @pytest.mark.usefixtures("restore_registries")
    def test_partial_payload_rejected_atomically(self):
        assert load_ops_backend_plugins([_ep("partial", "partial_bad")]) == ()
        assert "partialkit" not in get_op("rms_norm").backends
        assert "partialkit" not in KERNEL_REGISTRY.list_available("rms_norm", "standard")

    @pytest.mark.usefixtures("restore_registries")
    def test_broken_import_isolated(self):
        assert load_ops_backend_plugins([_ep("boom", "boom")]) == ()
        assert "testkit" not in get_op("rms_norm").backends

    @pytest.mark.usefixtures("restore_registries")
    def test_broken_plugin_does_not_block_later_plugins(self):
        loaded = load_ops_backend_plugins([_ep("a_boom", "boom"), _ep("b_good", "good")])
        assert loaded == ("b_good",)


# ---------------------------------------------------------------------------
# Kill-switch and re-entry safety
# ---------------------------------------------------------------------------


class TestSwitchAndReentry:
    @pytest.mark.usefixtures("restore_registries")
    def test_kill_switch_skips_everything(self, monkeypatch):
        monkeypatch.setenv("VEOMNI_OPS_PLUGINS", "0")
        assert load_ops_backend_plugins([_ep("good", "good")]) == ()
        assert "testkit" not in get_op("rms_norm").backends

    @pytest.mark.usefixtures("restore_registries")
    def test_repeated_load_is_safe(self):
        assert load_ops_backend_plugins([_ep("good", "good")]) == ("good",)
        # Second pass hits name-conflict rejection -> skipped, nothing duplicated.
        assert load_ops_backend_plugins([_ep("good", "good")]) == ()
        op = get_op("rms_norm")
        assert "testkit" in op.backends


# ---------------------------------------------------------------------------
# ``requires`` generalization (generic package probe)
# ---------------------------------------------------------------------------


class TestRequiresGeneralization:
    def test_missing_package_raises_runtime_error_naming_it(self):
        from veomni.ops.config.registry import _check_requires

        with pytest.raises(RuntimeError, match="definitely_not_installed_pkg_xyz"):
            _check_requires(("definitely_not_installed_pkg_xyz",))

    def test_installed_package_passes(self):
        from veomni.ops.config.registry import _check_requires

        _check_requires(("pytest",))
