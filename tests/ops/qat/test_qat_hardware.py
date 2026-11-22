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
def test_native_fp8_requires_nvidia_sm89(monkeypatch, field):
    fp8 = OpsSelection(name="torchao", kwargs={"dtype": "fp8", "recipe": "rowwise"})
    fp8_chained = OpsSelection(
        name="eager",
        overrides=[
            SiteOverride(site=r"layers\.0\.q_a_proj", name="torchao"),
            SiteOverride(site=r"layers\.0\.q_a_proj", kwargs={"dtype": "fp8"}),
        ],
    )
    fp8_emulated = OpsSelection(name="torchao", kwargs={"dtype": "fp8", "emulated": True})

    # SM80 rejects native FP8 (top-level and chained overrides) but permits emulated FP8.
    _device(monkeypatch, cuda=True, capability=80)
    with pytest.raises(RuntimeError, match="SM89"):
        reject_qat_without_native_bf16(_model(**{field: fp8}))
    with pytest.raises(RuntimeError, match="SM89"):
        reject_qat_without_native_bf16(_model(**{field: fp8_chained}))
    reject_qat_without_native_bf16(_model(**{field: fp8_emulated}))

    # ROCm rejects native FP8 even with capability >= 89.
    _device(monkeypatch, cuda=True, capability=90, hip="6.0")
    with pytest.raises(RuntimeError, match="SM89"):
        reject_qat_without_native_bf16(_model(**{field: fp8}))

    # NVIDIA SM89+ permits native FP8.
    _device(monkeypatch, cuda=True, capability=89)
    reject_qat_without_native_bf16(_model(**{field: fp8}))
    reject_qat_without_native_bf16(_model(**{field: fp8_chained}))


def test_bound_handles_are_authoritative_for_unused_families(monkeypatch):
    """Concrete models without a ``linear`` handle (e.g. Gemma3) ignore unused ``linear_implementation`` FP8."""
    from torch import nn

    from veomni.ops import VeomniOp
    from veomni.ops import registry as ops_registry

    monkeypatch.setattr(ops_registry, "is_package_available", lambda _pkg: True)
    monkeypatch.setattr(VeomniOp, "_intern", dict(VeomniOp._intern))

    fp8 = OpsSelection(name="torchao", kwargs={"dtype": "fp8", "recipe": "rowwise"})
    ops = SimpleNamespace(linear_implementation=fp8, gated_mlp_implementation="eager")
    model = nn.Linear(2, 2)
    model.config = SimpleNamespace(ops_config=ops, sub_configs={})
    model.veomni_gated_mlp = VeomniOp("gated_mlp", "standard", "eager")

    _device(monkeypatch, cuda=True, capability=80)
    reject_qat_without_native_bf16(model)

    model.veomni_q_proj = VeomniOp("linear", "standard", fp8, site="layers.0.q_proj")
    with pytest.raises(RuntimeError, match="SM89"):
        reject_qat_without_native_bf16(model)
