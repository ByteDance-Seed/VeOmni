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

import veomni.ops.platform.gpu as gpu_mod
import veomni.ops.platform.requirement as requirement_mod
import veomni.ops.registry as registry_mod
import veomni.utils.device as device
import veomni.utils.import_utils as import_utils
from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import OpsImplementationConfig
from veomni.ops import VeomniOp
from veomni.ops.platform import GpuKernelRequirement


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


_HOST_FIELDS = (
    "moe_implementation",
    "cross_entropy_loss_implementation",
    "rms_norm_implementation",
    "swiglu_mlp_implementation",
    "rotary_pos_emb_implementation",
    "rotary_pos_emb_vision_implementation",
    "load_balancing_loss_implementation",
)

_FIELD_OP_VARIANTS: dict[str, tuple[str, str]] = {
    "moe_implementation": ("moe_experts", "standard"),
    "cross_entropy_loss_implementation": ("cross_entropy_loss", "standard"),
    "rms_norm_implementation": ("rms_norm", "standard"),
    "swiglu_mlp_implementation": ("swiglu_mlp", "standard"),
    "rotary_pos_emb_implementation": ("rope", "partial"),
    "rotary_pos_emb_vision_implementation": ("rope", "full"),
    "load_balancing_loss_implementation": ("load_balancing_loss", "standard"),
}

_EXPECTED_DEFAULTS: dict[str, dict[str, str]] = {
    "gpu": {
        "moe_implementation": "fused_triton",
        "cross_entropy_loss_implementation": "liger_kernel",
        "rms_norm_implementation": "liger_kernel",
        "swiglu_mlp_implementation": "liger_kernel",
        "rotary_pos_emb_implementation": "liger_kernel",
        "rotary_pos_emb_vision_implementation": "liger_kernel",
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


def _set_host(monkeypatch: pytest.MonkeyPatch, host: str, *, apex_mlu: bool = False, mlu_liger: bool = False) -> None:
    monkeypatch.delenv("MODELING_BACKEND", raising=False)
    monkeypatch.setattr(VeomniOp, "_intern", {})
    monkeypatch.setattr(device, "IS_NPU_AVAILABLE", host == "npu")
    monkeypatch.setattr(import_utils, "is_torch_mlu_available", lambda: host == "mlu")
    monkeypatch.setattr(import_utils, "is_apex_mlu_available", lambda: apex_mlu)
    monkeypatch.setattr(import_utils, "is_liger_kernel_mlu_available", lambda: mlu_liger)
    monkeypatch.setattr(
        registry_mod, "get_device_type", lambda: GpuKernelRequirement.device if host == "gpu" else host
    )
    monkeypatch.setattr(gpu_mod, "IS_CUDA_AVAILABLE", host == "gpu")
    monkeypatch.setattr(requirement_mod, "IS_NPU_AVAILABLE", host == "npu")
    monkeypatch.setattr(requirement_mod, "IS_MLU_AVAILABLE", host == "mlu")
    # Simulated host defaults must not depend on packages installed on the real host.
    # Missing-package fallback is covered by the registry tests.
    monkeypatch.setattr(registry_mod, "is_package_available", lambda _package: True)
    if host == "gpu":
        monkeypatch.setattr(gpu_mod, "get_gpu_compute_capability", lambda: 90)


def test_host_fields_default_to_none():
    defaults = {name: f.default for name, f in OpsImplementationConfig.__dataclass_fields__.items()}
    assert {name: defaults[name] for name in _HOST_FIELDS} == dict.fromkeys(_HOST_FIELDS)


@pytest.mark.parametrize("host", ["gpu", "npu", "mlu"])
def test_unset_fields_resolve_to_the_host_default(host: str, monkeypatch: pytest.MonkeyPatch):
    _set_host(monkeypatch, host)
    config = OpsImplementationConfig()
    assert {name: getattr(config, name) for name in _HOST_FIELDS} == dict.fromkeys(_HOST_FIELDS)
    resolved = {
        name: VeomniOp(op, variant, getattr(config, name)).impl for name, (op, variant) in _FIELD_OP_VARIANTS.items()
    }
    assert resolved == _EXPECTED_DEFAULTS[host]


def test_unset_moe_resolves_to_fused_mlu_with_apex(monkeypatch: pytest.MonkeyPatch):
    _set_host(monkeypatch, "mlu", apex_mlu=True)
    config = OpsImplementationConfig()
    assert config.moe_implementation is None
    assert VeomniOp("moe_experts", "standard", config.moe_implementation).impl == "fused_mlu"


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
    _set_host(monkeypatch, "mlu", mlu_liger=True)
    config = OpsImplementationConfig(rms_norm_implementation="liger_kernel")
    assert config.rms_norm_implementation == "liger_kernel"


def test_explicit_liger_without_mlu_build_raises_on_mlu(monkeypatch: pytest.MonkeyPatch):
    """Upstream GPU liger does not run on MLU; only the Cambricon build satisfies an explicit selection."""
    _set_host(monkeypatch, "mlu")
    with pytest.raises(ValueError, match="requires the Cambricon MLU liger_kernel build"):
        OpsImplementationConfig(rms_norm_implementation="liger_kernel")


def test_unset_fields_resolve_to_mlu_liger_defaults(monkeypatch: pytest.MonkeyPatch):
    """With the Cambricon liger build, unset liger-capable fields default to liger_kernel on MLU."""
    _set_host(monkeypatch, "mlu", mlu_liger=True)
    config = OpsImplementationConfig()
    assert {name: getattr(config, name) for name in _HOST_FIELDS} == dict.fromkeys(_HOST_FIELDS)
    with ops_config_scope(config):
        resolved = {
            name: VeomniOp(op, variant, resolve_op_impl(name)).impl
            for name, (op, variant) in _FIELD_OP_VARIANTS.items()
        }
    expected = _EXPECTED_DEFAULTS["mlu"] | {
        "cross_entropy_loss_implementation": "liger_kernel",
        "rms_norm_implementation": "liger_kernel",
        "swiglu_mlp_implementation": "liger_kernel",
        "rotary_pos_emb_implementation": "liger_kernel",
        "rotary_pos_emb_vision_implementation": "liger_kernel",
    }
    assert resolved == expected


def test_installed_torch_npu_without_runtime_keeps_gpu_defaults(monkeypatch: pytest.MonkeyPatch):
    """A packaged ``torch_npu`` with autoload off is not an NPU host."""
    _set_host(monkeypatch, "gpu")
    monkeypatch.setattr(import_utils, "is_torch_npu_available", lambda: True)
    config = OpsImplementationConfig()
    assert config.moe_implementation is None
    assert VeomniOp("moe_experts", "standard", config.moe_implementation).impl == "fused_triton"
