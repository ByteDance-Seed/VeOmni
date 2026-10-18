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

"""Unset kernel fields take the host default; explicit values are never rewritten."""

from __future__ import annotations

import pytest

import veomni.utils.device as device
import veomni.utils.import_utils as import_utils
from veomni.arguments import OpsImplementationConfig


_HOST_FIELDS = (
    "moe_implementation",
    "cross_entropy_loss_implementation",
    "rms_norm_implementation",
    "swiglu_mlp_implementation",
    "rotary_pos_emb_implementation",
    "rotary_pos_emb_vision_implementation",
    "load_balancing_loss_implementation",
)

_EXPECTED_DEFAULTS: dict[str, dict[str, str]] = {
    "gpu": {
        "moe_implementation": "fused_triton",
        "cross_entropy_loss_implementation": "liger_kernel",
        "rms_norm_implementation": "liger_kernel",
        "swiglu_mlp_implementation": "liger_kernel",
        "rotary_pos_emb_implementation": "liger_kernel",
        "rotary_pos_emb_vision_implementation": "eager",
        "load_balancing_loss_implementation": "triton",
    },
    "npu": {
        "moe_implementation": "fused_npu",
        "cross_entropy_loss_implementation": "chunk_loss",
        "rms_norm_implementation": "torch_npu",
        "swiglu_mlp_implementation": "eager",
        "rotary_pos_emb_implementation": "torch_npu",
        "rotary_pos_emb_vision_implementation": "torch_npu",
        "load_balancing_loss_implementation": "eager",
    },
    "mlu": {
        "moe_implementation": "fused_triton",
        "cross_entropy_loss_implementation": "eager",
        "rms_norm_implementation": "eager",
        "swiglu_mlp_implementation": "eager",
        "rotary_pos_emb_implementation": "eager",
        "rotary_pos_emb_vision_implementation": "eager",
        "load_balancing_loss_implementation": "eager",
    },
}


def _set_host(monkeypatch: pytest.MonkeyPatch, host: str, *, apex_mlu: bool = False) -> None:
    monkeypatch.delenv("MODELING_BACKEND", raising=False)
    monkeypatch.setattr(device, "IS_NPU_AVAILABLE", host == "npu")
    monkeypatch.setattr(import_utils, "is_torch_mlu_available", lambda: host == "mlu")
    monkeypatch.setattr(import_utils, "is_apex_mlu_available", lambda: apex_mlu)


def test_host_fields_default_to_none():
    defaults = {name: f.default for name, f in OpsImplementationConfig.__dataclass_fields__.items()}
    assert {name: defaults[name] for name in _HOST_FIELDS} == dict.fromkeys(_HOST_FIELDS)


@pytest.mark.parametrize("host", ["gpu", "npu", "mlu"])
def test_unset_fields_take_the_host_default(host: str, monkeypatch: pytest.MonkeyPatch):
    _set_host(monkeypatch, host)
    config = OpsImplementationConfig()
    assert {name: getattr(config, name) for name in _HOST_FIELDS} == _EXPECTED_DEFAULTS[host]


def test_unset_moe_takes_fused_mlu_with_apex(monkeypatch: pytest.MonkeyPatch):
    _set_host(monkeypatch, "mlu", apex_mlu=True)
    assert OpsImplementationConfig().moe_implementation == "fused_mlu"


@pytest.mark.parametrize("field_name", _HOST_FIELDS)
def test_explicit_eager_is_kept_on_npu(field_name: str, monkeypatch: pytest.MonkeyPatch):
    _set_host(monkeypatch, "npu")
    assert getattr(OpsImplementationConfig(**{field_name: "eager"}), field_name) == "eager"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("rms_norm_implementation", "torch_npu"),
        ("rotary_pos_emb_implementation", "torch_npu"),
        ("rotary_pos_emb_vision_implementation", "torch_npu"),
        ("moe_implementation", "fused_npu"),
        ("cross_entropy_loss_implementation", "chunk_loss"),
    ],
)
def test_explicit_npu_value_is_kept_on_npu(field_name: str, value: str, monkeypatch: pytest.MonkeyPatch):
    _set_host(monkeypatch, "npu")
    assert getattr(OpsImplementationConfig(**{field_name: value}), field_name) == value


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("moe_implementation", "fused_triton"),
        ("cross_entropy_loss_implementation", "liger_kernel"),
        ("rms_norm_implementation", "liger_kernel"),
        ("swiglu_mlp_implementation", "liger_kernel"),
        ("rotary_pos_emb_implementation", "liger_kernel"),
        ("load_balancing_loss_implementation", "triton"),
    ],
)
def test_explicit_gpu_default_raises_on_npu(field_name: str, value: str, monkeypatch: pytest.MonkeyPatch):
    _set_host(monkeypatch, "npu")
    with pytest.raises(ValueError, match=f"{field_name}='{value}' is not supported on Ascend NPU"):
        OpsImplementationConfig(**{field_name: value})


@pytest.mark.parametrize(
    "field_name",
    ["rms_norm_implementation", "rotary_pos_emb_implementation", "rotary_pos_emb_vision_implementation"],
)
def test_explicit_npu_value_raises_off_npu(field_name: str, monkeypatch: pytest.MonkeyPatch):
    _set_host(monkeypatch, "gpu")
    with pytest.raises(ValueError, match=f"{field_name}='torch_npu' requires Ascend NPU"):
        OpsImplementationConfig(**{field_name: "torch_npu"})


def test_explicit_gpu_value_is_kept_on_mlu(monkeypatch: pytest.MonkeyPatch):
    _set_host(monkeypatch, "mlu")
    config = OpsImplementationConfig(rms_norm_implementation="liger_kernel")
    assert config.rms_norm_implementation == "liger_kernel"


def test_installed_torch_npu_without_runtime_keeps_gpu_defaults(monkeypatch: pytest.MonkeyPatch):
    """A packaged ``torch_npu`` with autoload off is not an NPU host."""
    monkeypatch.delenv("MODELING_BACKEND", raising=False)
    monkeypatch.setattr(import_utils, "is_torch_npu_available", lambda: True)
    monkeypatch.setattr(device, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(import_utils, "is_torch_mlu_available", lambda: False)
    assert OpsImplementationConfig().moe_implementation == "fused_triton"
