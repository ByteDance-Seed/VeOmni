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

"""Training-path hardware precondition for the torchao QAT rows.

The fake quantizer runs on CPU, which is what the op-level tests use; only the
distributed training GEMM needs native BF16, so the check must refuse an SM80-
host but leave a CPU host alone.
"""

from types import SimpleNamespace

import pytest

from veomni.ops.qat._hardware import reject_qat_without_native_bf16


def _model(**ops_fields) -> SimpleNamespace:
    ops = SimpleNamespace(linear_implementation=None, swiglu_mlp_implementation=None, **ops_fields)
    return SimpleNamespace(config=SimpleNamespace(ops_config=ops, sub_configs={}))


def _nested_model(**ops_fields) -> SimpleNamespace:
    child_ops = SimpleNamespace(linear_implementation=None, swiglu_mlp_implementation=None, **ops_fields)
    child = SimpleNamespace(ops_config=child_ops, sub_configs={})
    root_ops = SimpleNamespace(linear_implementation=None, swiglu_mlp_implementation=None)
    return SimpleNamespace(
        config=SimpleNamespace(ops_config=root_ops, sub_configs={"text_config": None}, text_config=child)
    )


def _device(monkeypatch, *, cuda: bool, capability: int):
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", cuda)
    monkeypatch.setattr("veomni.utils.device.get_gpu_compute_capability", lambda _device=None: capability)


def test_leaves_a_model_without_torchao_rows_alone(monkeypatch):
    _device(monkeypatch, cuda=True, capability=70)
    reject_qat_without_native_bf16(_model(linear_implementation="qat"))


@pytest.mark.parametrize("field", ["linear_implementation", "swiglu_mlp_implementation"])
def test_refuses_a_torchao_row_below_sm80(monkeypatch, field):
    _device(monkeypatch, cuda=True, capability=79)
    with pytest.raises(RuntimeError, match="native BF16 support"):
        reject_qat_without_native_bf16(_model(**{field: "torchao_mxfp4"}))


@pytest.mark.parametrize("capability", [80, 89, 90])
def test_allows_a_torchao_row_from_sm80(monkeypatch, capability):
    _device(monkeypatch, cuda=True, capability=capability)
    reject_qat_without_native_bf16(_model(linear_implementation="torchao_nvfp4"))


def test_leaves_a_cpu_host_alone(monkeypatch):
    """The fake quantizer is tensor math and runs on CPU; only training needs BF16 hardware."""
    _device(monkeypatch, cuda=False, capability=0)
    reject_qat_without_native_bf16(_model(linear_implementation="torchao_mxfp4"))


def test_walks_nested_configs(monkeypatch):
    _device(monkeypatch, cuda=True, capability=70)
    with pytest.raises(RuntimeError, match="linear_implementation"):
        reject_qat_without_native_bf16(_nested_model(linear_implementation="torchao_mxfp4"))
