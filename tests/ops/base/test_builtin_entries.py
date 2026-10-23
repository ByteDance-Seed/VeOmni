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

"""Canonical metadata contract for built-in operation registry rows."""

from __future__ import annotations

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import OP_REGISTRY
from veomni.ops.platform import (
    GpuKernelRequirement,
    MluKernelRequirement,
    NpuKernelRequirement,
    NvidiaGpuPlatform,
    RocmGpuPlatform,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


_ANY = None
_GPU = ("gpu", (("nvidia", None, None), ("rocm", None, None)))
_GPU_SM70 = ("gpu", (("nvidia", 70, None), ("rocm", None, None)))
_GPU_SM80 = ("gpu", (("nvidia", 80, None), ("rocm", None, None)))
_GPU_NVIDIA = ("gpu", (("nvidia", None, None),))
_GPU_NVIDIA_SM80 = ("gpu", (("nvidia", 80, None),))
_GPU_NVIDIA_SM90 = ("gpu", (("nvidia", 90, None),))
_GPU_NVIDIA_SM100 = ("gpu", (("nvidia", 100, None),))
_GPU_NVIDIA_SM90_TO_SM100 = ("gpu", (("nvidia", 90, 100),))
_NPU = ("npu", ())
_MLU = ("mlu", ())


_CRITICAL_BUILTIN_ENTRIES = (
    ("attention", "standard", "flash_attention_2", "cuda", _GPU_SM80, ("flash_attn",)),
    ("attention", "standard", "flash_attention_2", "npu", _NPU, ()),
    ("attention", "standard", "flash_attention_3", "cuda", _GPU_NVIDIA_SM90, ("flash_attn_interface",)),
    ("attention", "standard", "flash_attention_4", "cuda", _GPU_NVIDIA_SM90, ("flash_attn.cute",)),
    ("attention", "standard", "veomni_flash_attention_2", "cuda", _GPU_SM80, ("flash_attn",)),
    ("attention", "standard", "veomni_flash_attention_2", "npu", _NPU, ()),
    ("attention", "standard", "veomni_flash_attention_3", "cuda", _GPU_NVIDIA_SM90, ("flash_attn_interface",)),
    ("attention", "standard", "veomni_flash_attention_4", "cuda", _GPU_NVIDIA_SM90, ("flash_attn.cute",)),
    ("attention", "standard", "veomni_flex_attention", "cuda", _GPU_NVIDIA, ()),
    (
        "attention",
        "standard",
        "veomni_magi_attention",
        "cuda",
        _GPU_NVIDIA_SM100,
        ("magi_attention", "flash_attn_cute", "debugpy"),
    ),
    ("dsa_attention", "deepseek_v4", "tilelang", "cuda", _GPU_NVIDIA_SM90, ("tilelang",)),
    ("dsa_attention", "deepseek_v4", "cann", "npu", _NPU, ("cann_ops_transformer",)),
    ("dsa_indexer", "deepseek_v4", "tilelang", "cuda", _GPU_NVIDIA_SM90, ("tilelang",)),
    ("dsa_indexer", "deepseek_v4", "cann", "npu", _NPU, ("cann_ops_transformer",)),
    ("mhc", "pre", "tilelang", "cuda", _GPU_NVIDIA_SM90, ("tile_kernels",)),
    ("mhc", "pre", "cann", "npu", _NPU, ("cann_ops_transformer",)),
    ("mhc", "post", "tilelang", "cuda", _GPU_NVIDIA_SM90, ("tile_kernels",)),
    ("mhc", "post", "cann", "npu", _NPU, ("cann_ops_transformer",)),
    ("mhc", "head", "tilelang", "cuda", _GPU_NVIDIA_SM90, ("tile_kernels",)),
    (
        "chunk_gated_delta_rule",
        "standard",
        "flash_qla",
        "cuda",
        _GPU_NVIDIA_SM90_TO_SM100,
        ("flash_qla",),
    ),
    ("chunk_gated_delta_rule", "standard", "triton", "npu", _NPU, ("triton",)),
    ("moe_experts", "standard", "fused_triton", "cuda", _GPU_SM70, ("triton",)),
    ("moe_experts", "standard", "fused_quack", "cuda", _GPU_NVIDIA_SM90, ("quack",)),
    ("moe_experts", "standard", "fused_npu", "npu", _NPU, ()),
    ("moe_experts", "gpt_oss", "fused_quack", "cuda", _GPU_NVIDIA_SM90, ("quack",)),
    ("moe_experts_lora", "shared", "fused_triton", "cuda", _GPU_SM70, ("triton",)),
    ("moe_experts_lora", "shared", "fused_npu", "npu", _NPU, ()),
    ("moe_experts_lora", "independent", "fused_triton", "cuda", _GPU_SM70, ("triton",)),
    ("moe_experts_lora", "independent", "fused_npu", "npu", _NPU, ()),
)


def _requirement_metadata(requirement):
    """Normalize requirements into data independent of shared preset objects."""
    if requirement is None:
        return None
    if isinstance(requirement, GpuKernelRequirement):
        platforms = []
        for platform in requirement.platforms:
            if isinstance(platform, NvidiaGpuPlatform):
                platforms.append(("nvidia", platform.min_cc, platform.max_cc))
            elif isinstance(platform, RocmGpuPlatform):
                platforms.append(("rocm", None, None))
            else:
                raise AssertionError(f"unexpected GPU platform type: {type(platform).__name__}")
        return "gpu", tuple(platforms)
    if isinstance(requirement, NpuKernelRequirement):
        return "npu", ()
    if isinstance(requirement, MluKernelRequirement):
        return "mlu", ()
    raise AssertionError(f"unexpected requirement type: {type(requirement).__name__}")


def test_builtin_entry_invariants_and_critical_hardware_gates():
    """Verify structural invariants across all built-in rows plus critical hardware/dependency gates."""
    assert OP_REGISTRY._entries, "OP_REGISTRY must not be empty"
    actual = {
        (*key, _requirement_metadata(entry.requirement), entry.requires) for key, entry in OP_REGISTRY._entries.items()
    }
    for op, variant, impl, device, req_meta, _requires in actual:
        if device == "any":
            assert req_meta is None, f"{op}/{variant}/{impl} on device='any' must not set hardware requirement"
        else:
            expected_kind = "gpu" if device == "cuda" else device
            assert req_meta is not None and req_meta[0] == expected_kind, (
                f"{op}/{variant}/{impl} on device={device!r} has mismatched requirement {req_meta}"
            )

    missing_critical = set(_CRITICAL_BUILTIN_ENTRIES) - actual
    assert not missing_critical, f"Missing or altered critical built-in entries: {sorted(missing_critical)}"


_BUILTIN_DEFAULT_ENTRIES = {
    ("cross_entropy_loss", "standard", "liger_kernel", "cuda"),
    ("cross_entropy_loss", "standard", "chunk_loss", "npu"),
    ("load_balancing_loss", "standard", "triton", "cuda"),
    ("moe_experts", "standard", "fused_triton", "cuda"),
    ("moe_experts", "standard", "fused_mlu", "mlu"),
    ("moe_experts", "standard", "fused_triton", "mlu"),
    ("moe_experts", "standard", "fused_npu", "npu"),
    ("rms_norm", "standard", "liger_kernel", "cuda"),
    ("rms_norm", "standard", "torch_npu", "npu"),
    ("rms_norm", "deepseek_v4", "liger_kernel", "cuda"),
    ("rms_norm", "deepseek_v4", "torch_npu", "npu"),
    ("rms_norm", "offset", "liger_kernel", "cuda"),
    ("rms_norm", "offset", "torch_npu", "npu"),
    ("rms_norm", "unweighted", "liger_kernel", "cuda"),
    ("rope", "full", "liger_kernel", "cuda"),
    ("rope", "full", "torch_npu", "npu"),
    ("rope", "partial", "liger_kernel", "cuda"),
    ("rope", "partial", "torch_npu", "npu"),
    ("swiglu_mlp", "standard", "liger_kernel", "cuda"),
    ("swiglu_mlp", "geglu", "liger_kernel", "cuda"),
}


def test_builtin_default_entries_are_exact():
    """Only the intended device rows set default=True."""
    actual = {key for key, entry in OP_REGISTRY._entries.items() if entry.default}
    assert actual == _BUILTIN_DEFAULT_ENTRIES
