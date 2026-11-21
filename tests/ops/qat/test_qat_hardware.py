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

"""Training-path hardware precondition for torchao low-precision rows.

Emulated FP4/FP8 rows also run on CPU for op-level tests; only distributed
training on CUDA requires SM80+ (native BF16) for emulated rows and NVIDIA
SM89+ (Ada/Hopper/Blackwell) for native FP8.
"""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import OpsSelection, SiteOverride
from veomni.ops.qat._hardware import reject_qat_without_native_bf16


# The device is monkeypatched: these cases never touch an accelerator.
pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _model(**ops_fields) -> SimpleNamespace:
    ops = SimpleNamespace(**{"linear_implementation": None, "gated_mlp_implementation": None, **ops_fields})
    return SimpleNamespace(config=SimpleNamespace(ops_config=ops, sub_configs={}))


def _nested_model(**ops_fields) -> SimpleNamespace:
    child_ops = SimpleNamespace(**{"linear_implementation": None, "gated_mlp_implementation": None, **ops_fields})
    child = SimpleNamespace(ops_config=child_ops, sub_configs={})
    root_ops = SimpleNamespace(linear_implementation=None, gated_mlp_implementation=None)
    return SimpleNamespace(
        config=SimpleNamespace(ops_config=root_ops, sub_configs={"text_config": None}, text_config=child)
    )


def _device(monkeypatch, *, cuda: bool, capability: int, hip: str | None = None):
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", cuda)
    monkeypatch.setattr("veomni.utils.device.get_gpu_compute_capability", lambda _device=None: capability)
    monkeypatch.setattr(torch.version, "hip", hip)


def test_leaves_a_model_without_torchao_rows_alone(monkeypatch):
    _device(monkeypatch, cuda=True, capability=70)
    reject_qat_without_native_bf16(_model(linear_implementation="qat"))


_TORCHAO_ROWS = ("torchao",)


@pytest.mark.parametrize("impl", _TORCHAO_ROWS)
@pytest.mark.parametrize("field", ["linear_implementation", "gated_mlp_implementation"])
def test_refuses_a_torchao_row_below_sm80(monkeypatch, field, impl):
    _device(monkeypatch, cuda=True, capability=79)
    with pytest.raises(RuntimeError, match="native BF16 support"):
        reject_qat_without_native_bf16(_model(**{field: impl}))


@pytest.mark.parametrize("capability", [80, 89, 90])
def test_allows_a_torchao_row_from_sm80(monkeypatch, capability):
    _device(monkeypatch, cuda=True, capability=capability)
    reject_qat_without_native_bf16(_model(linear_implementation="torchao"))


def test_leaves_a_cpu_host_alone(monkeypatch):
    """The fake quantizer is tensor math and runs on CPU; only training needs BF16 hardware."""
    _device(monkeypatch, cuda=False, capability=0)
    reject_qat_without_native_bf16(_model(linear_implementation="torchao"))


@pytest.mark.parametrize("impl", _TORCHAO_ROWS)
def test_walks_nested_configs(monkeypatch, impl):
    _device(monkeypatch, cuda=True, capability=70)
    with pytest.raises(RuntimeError, match=impl):
        reject_qat_without_native_bf16(_nested_model(linear_implementation=impl))


@pytest.mark.parametrize("field", ["linear_implementation", "gated_mlp_implementation"])
def test_refuses_a_torchao_override_below_sm80(monkeypatch, field):
    override = OpsSelection(
        name="eager",
        overrides=[SiteOverride(site=r"layers\.0\..*", name="torchao", kwargs={"dtype": "nvfp4"})],
    )
    _device(monkeypatch, cuda=True, capability=79)
    with pytest.raises(RuntimeError, match="native BF16 support"):
        reject_qat_without_native_bf16(_model(**{field: override}))

    _device(monkeypatch, cuda=True, capability=80)
    reject_qat_without_native_bf16(_model(**{field: override}))


@pytest.mark.parametrize("field", ["linear_implementation", "gated_mlp_implementation"])
def test_native_fp8_requires_nvidia_sm89_including_overrides(monkeypatch, field):
    fp8 = OpsSelection(name="torchao", kwargs={"dtype": "fp8", "recipe": "rowwise"})
    fp8_override = OpsSelection(
        name="eager",
        overrides=[SiteOverride(site=r"layers\.0\..*", name="torchao", kwargs={"dtype": "fp8"})],
    )
    fp8_emulated = OpsSelection(name="torchao", kwargs={"dtype": "fp8", "emulated": True})

    # SM80 rejects native FP8 (both top-level and override) but permits emulated FP8.
    _device(monkeypatch, cuda=True, capability=80)
    with pytest.raises(RuntimeError, match="SM89"):
        reject_qat_without_native_bf16(_model(**{field: fp8}))
    with pytest.raises(RuntimeError, match="SM89"):
        reject_qat_without_native_bf16(_model(**{field: fp8_override}))
    reject_qat_without_native_bf16(_model(**{field: fp8_emulated}))

    # ROCm rejects native FP8 even with capability >= 89.
    _device(monkeypatch, cuda=True, capability=90, hip="6.0")
    with pytest.raises(RuntimeError, match="SM89"):
        reject_qat_without_native_bf16(_model(**{field: fp8}))

    # NVIDIA SM89+ permits native FP8.
    _device(monkeypatch, cuda=True, capability=89)
    reject_qat_without_native_bf16(_model(**{field: fp8}))
    reject_qat_without_native_bf16(_model(**{field: fp8_override}))


@pytest.mark.parametrize("field", ["linear_implementation", "gated_mlp_implementation"])
def test_chained_overrides_selecting_native_fp8_require_sm89(monkeypatch, field):
    """Later overrides inherit the row chosen by earlier matching overrides, not just the base."""
    chained_name_then_dtype = OpsSelection(
        name="eager",
        overrides=[
            SiteOverride(site=r"layers\.0\.q_a_proj", name="torchao"),
            SiteOverride(site=r"layers\.0\.q_a_proj", kwargs={"dtype": "fp8"}),
        ],
    )

    _device(monkeypatch, cuda=True, capability=80)
    with pytest.raises(RuntimeError, match="SM89"):
        reject_qat_without_native_bf16(_model(**{field: chained_name_then_dtype}))

    _device(monkeypatch, cuda=True, capability=89)
    reject_qat_without_native_bf16(_model(**{field: chained_name_then_dtype}))


def test_bound_veomni_op_handles_resolve_concrete_sites(monkeypatch):
    """Real modules with bound VeomniOp handles inspect concrete sites without cross-site false positives."""
    from veomni.ops import registry
    from veomni.ops.registry import VeomniOp

    orig_is_available = registry.is_package_available
    monkeypatch.setattr(
        registry, "is_package_available", lambda pkg: True if pkg == "torchao" else orig_is_available(pkg)
    )
    monkeypatch.setattr(VeomniOp, "_intern", dict(VeomniOp._intern))

    class _TwoSiteModule(torch.nn.Module):
        def __init__(self, selection: OpsSelection):
            super().__init__()
            self.config = SimpleNamespace(
                ops_config=SimpleNamespace(linear_implementation=selection, gated_mlp_implementation=None),
                sub_configs={},
            )
            self.op_0 = VeomniOp("linear", "standard", selection, site="layers.0.q_a_proj")
            self.op_1 = VeomniOp("linear", "standard", selection, site="layers.1.q_a_proj")

    # Same concrete site hit by chained overrides -> resolves to torchao native FP8 -> SM80 rejected.
    same_site_chained = OpsSelection(
        name="eager",
        overrides=[
            SiteOverride(site=r"layers\.0\.q_a_proj", name="torchao"),
            SiteOverride(site=r"layers\.0\.q_a_proj", kwargs={"dtype": "fp8"}),
        ],
    )
    _device(monkeypatch, cuda=True, capability=80)
    with pytest.raises(RuntimeError, match="SM89"):
        reject_qat_without_native_bf16(_TwoSiteModule(same_site_chained))

    # Disjoint concrete sites: layer 0 is torchao(nvfp4), layer 1 is eager(dtype=fp8 dropped) -> SM80 allowed.
    disjoint_sites = OpsSelection(
        name="eager",
        overrides=[
            SiteOverride(site=r"layers\.0\.q_a_proj", name="torchao", kwargs={"dtype": "nvfp4"}),
            SiteOverride(site=r"layers\.1\.q_a_proj", kwargs={"dtype": "fp8"}),
        ],
    )
    reject_qat_without_native_bf16(_TwoSiteModule(disjoint_sites))
