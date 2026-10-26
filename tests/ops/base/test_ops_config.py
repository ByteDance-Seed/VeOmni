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

"""CPU tests for installed ops-config resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from veomni.ops.config import get_ops_config, resolve_op_impl, set_ops_config


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("initially_configured", [False, True], ids=["none", "existing-config"])
@pytest.mark.parametrize("fail", [False, True], ids=["normal-exit", "exception"])
def test_ops_config_scope_restores_nested_bindings(initially_configured, fail):
    previous = get_ops_config()
    initial = eager_ops_config() if initially_configured else None
    inner = eager_ops_config()
    inner.cross_entropy_loss_implementation = "chunk_loss"
    with ops_config_scope(initial):
        assert get_ops_config() is initial

        def nested_scope():
            with ops_config_scope(inner):
                assert get_ops_config() is inner
                with ops_config_scope(None):
                    assert get_ops_config() is None
                assert get_ops_config() is inner
                # The scoped code may itself install another config (as model builders do).
                set_ops_config(eager_ops_config())
                if fail:
                    raise RuntimeError("construction failed")

        if fail:
            with pytest.raises(RuntimeError, match="construction failed"):
                nested_scope()
        else:
            nested_scope()
        assert get_ops_config() is initial
    assert get_ops_config() is previous


def test_resolve_op_impl_defaults_to_eager():
    set_ops_config(None)
    assert resolve_op_impl("rms_norm_implementation") == "eager"


def test_resolve_op_impl_reads_ops_config():
    set_ops_config(SimpleNamespace(cross_entropy_loss_implementation="chunk_loss"))
    assert resolve_op_impl("cross_entropy_loss_implementation") == "chunk_loss"


def test_resolve_op_impl_returns_none_for_unset_field_and_veomni_op_resolves_variant_default(monkeypatch):
    from veomni.arguments import OpsImplementationConfig
    from veomni.ops import VeomniOp

    monkeypatch.setattr(VeomniOp, "_intern", {})
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", True)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr("veomni.ops.registry.get_device_type", lambda: "npu")
    monkeypatch.setattr("veomni.ops.platform.requirement.IS_NPU_AVAILABLE", True)

    cfg = OpsImplementationConfig()
    with ops_config_scope(cfg):
        assert resolve_op_impl("rms_norm_implementation") is None
        assert VeomniOp("rms_norm", "standard", resolve_op_impl("rms_norm_implementation")).impl == "torch_npu"
        assert VeomniOp("rms_norm", "unweighted", resolve_op_impl("rms_norm_implementation")).impl == "eager"


@pytest.mark.parametrize("impl", [None, "torch_npu", "npu", "eager", "liger_kernel"])
def test_resolve_op_impl_preserves_variant_scoped_npu_compatibility(impl):
    with ops_config_scope(SimpleNamespace(rms_norm_implementation=impl)):
        assert resolve_op_impl("rms_norm_implementation") == impl
        assert resolve_op_impl("rms_norm_implementation", variant="standard") == impl
        expected = "eager" if impl in ("torch_npu", "npu") else impl
        resolved = resolve_op_impl("rms_norm_implementation", variant="unweighted")
        assert resolved == expected
        if expected == "eager":
            from veomni.ops import VeomniOp

            assert VeomniOp("rms_norm", "unweighted", resolved).impl == "eager"
