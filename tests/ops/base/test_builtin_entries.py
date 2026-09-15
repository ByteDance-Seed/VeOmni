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

from veomni.ops import OP_REGISTRY
from veomni.ops.platform import (
    GpuKernelRequirement,
    MluKernelRequirement,
    NpuKernelRequirement,
    NvidiaGpuPlatform,
    RocmGpuPlatform,
)


_ANY = None
_GPU = ("gpu", (("nvidia", None, None), ("rocm", None, None)))
_GPU_SM70 = ("gpu", (("nvidia", 70, None), ("rocm", None, None)))
_GPU_SM80 = ("gpu", (("nvidia", 80, None), ("rocm", None, None)))
_GPU_NVIDIA_SM80 = ("gpu", (("nvidia", 80, None),))
_GPU_NVIDIA_SM90 = ("gpu", (("nvidia", 90, None),))
_GPU_NVIDIA_SM90_TO_SM100 = ("gpu", (("nvidia", 90, 100),))
_NPU = ("npu", ())
_MLU = ("mlu", ())


_BUILTIN_ENTRIES = (
    ("async_ulysses_qkv", "standard", "eager", "any", _ANY, ()),
    ("async_ulysses_qkv", "dit", "eager", "any", _ANY, ()),
    ("async_ulysses_o", "standard", "eager", "any", _ANY, ()),
    ("async_ulysses_o", "dit", "eager", "any", _ANY, ()),
    ("attention", "standard", "eager", "any", _ANY, ()),
    ("attention", "standard", "sdpa", "any", _ANY, ()),
    ("attention", "standard", "flash_attention_2", "cuda", _GPU_SM80, ("flash_attn",)),
    ("attention", "standard", "flash_attention_2", "npu", _NPU, ()),
    ("attention", "standard", "flash_attention_2", "mlu", _MLU, ("flash_attn",)),
    ("attention", "standard", "flash_attention_3", "cuda", _GPU_NVIDIA_SM90, ("flash_attn_interface",)),
    ("attention", "standard", "flash_attention_4", "cuda", _GPU_NVIDIA_SM90, ("flash_attn.cute",)),
    ("attention", "standard", "flex_attention", "any", _ANY, ()),
    (
        "attention",
        "standard",
        "magi_attention",
        "cuda",
        _GPU_NVIDIA_SM90,
        ("magi_attention", "flash_attn_cute", "cuda.bindings", "debugpy"),
    ),
    ("attention", "standard", "veomni_flash_attention_2", "cuda", _GPU_SM80, ("flash_attn",)),
    ("attention", "standard", "veomni_flash_attention_2", "npu", _NPU, ()),
    ("attention", "standard", "veomni_flash_attention_2", "mlu", _MLU, ("flash_attn",)),
    (
        "attention",
        "standard",
        "veomni_flash_attention_3",
        "cuda",
        _GPU_NVIDIA_SM90,
        ("flash_attn_interface",),
    ),
    ("attention", "standard", "veomni_flash_attention_4", "cuda", _GPU_NVIDIA_SM90, ("flash_attn.cute",)),
    ("attention", "standard", "veomni_flex_attention", "any", _ANY, ()),
    (
        "attention",
        "standard",
        "veomni_magi_attention",
        "cuda",
        _GPU_NVIDIA_SM90,
        ("magi_attention", "flash_attn_cute", "cuda.bindings", "debugpy"),
    ),
    ("attention", "standard", "veomni_sage_attention", "cuda", _GPU_NVIDIA_SM80, ("sageattention",)),
    ("attention", "standard", "veomni_sdpa", "any", _ANY, ()),
    ("dsa_attention", "deepseek_v4", "eager", "any", _ANY, ()),
    ("dsa_attention", "deepseek_v4", "tilelang", "cuda", _GPU_NVIDIA_SM90, ("tilelang",)),
    ("dsa_attention", "glm", "eager", "any", _ANY, ()),
    ("dsa_attention", "glm", "flashmla_cudnn", "cuda", _GPU_NVIDIA_SM90, ("cudnn", "flash_mla")),
    ("dsa_indexer", "deepseek_v4", "eager", "any", _ANY, ()),
    ("dsa_indexer", "deepseek_v4", "tilelang", "cuda", _GPU_NVIDIA_SM90, ("tilelang",)),
    ("dsa_indexer", "glm", "eager", "any", _ANY, ()),
    ("dsa_indexer", "glm", "cudnn", "cuda", _GPU_NVIDIA_SM90, ("cudnn", "flash_mla")),
    ("rms_norm_gated", "standard", "eager", "any", _ANY, ()),
    ("rms_norm_gated", "standard", "fla", "cuda", _GPU, ("fla",)),
    ("rms_norm_gated", "standard", "fla", "mlu", _MLU, ("fla",)),
    ("rms_norm_gated", "standard", "npu", "npu", _NPU, ()),
    ("causal_conv1d", "standard", "eager", "any", _ANY, ()),
    ("causal_conv1d", "standard", "fla", "cuda", _GPU, ("fla",)),
    ("causal_conv1d", "standard", "fla", "mlu", _MLU, ("fla",)),
    ("causal_conv1d", "standard", "npu", "npu", _NPU, ("triton",)),
    ("chunk_gated_delta_rule", "standard", "eager", "any", _ANY, ()),
    ("chunk_gated_delta_rule", "standard", "fla", "cuda", _GPU, ("fla",)),
    ("chunk_gated_delta_rule", "standard", "fla", "mlu", _MLU, ("fla",)),
    (
        "chunk_gated_delta_rule",
        "standard",
        "flash_qla",
        "cuda",
        _GPU_NVIDIA_SM90_TO_SM100,
        ("flash_qla",),
    ),
    ("chunk_gated_delta_rule", "standard", "npu", "npu", _NPU, ("triton",)),
    ("chunk_gated_delta_rule", "standard", "npu_ascendc", "npu", _NPU, ("fla_npu", "triton")),
    ("load_balancing_loss", "standard", "eager", "any", _ANY, ()),
    ("load_balancing_loss", "standard", "triton", "cuda", _GPU, ("triton",)),
    ("cross_entropy_loss", "standard", "eager", "any", _ANY, ()),
    ("cross_entropy_loss", "standard", "liger_kernel", "cuda", _GPU, ("liger_kernel",)),
    ("cross_entropy_loss", "standard", "chunk_loss", "any", _ANY, ()),
    ("mhc", "pre", "eager", "any", _ANY, ()),
    ("mhc", "pre", "tilelang", "cuda", _GPU_NVIDIA_SM90, ("tile_kernels",)),
    ("mhc", "post", "eager", "any", _ANY, ()),
    ("mhc", "post", "tilelang", "cuda", _GPU_NVIDIA_SM90, ("tile_kernels",)),
    ("mhc", "head", "eager", "any", _ANY, ()),
    ("mhc", "head", "tilelang", "cuda", _GPU_NVIDIA_SM90, ("tile_kernels",)),
    ("moe_experts", "standard", "eager", "any", _ANY, ()),
    ("moe_experts", "standard", "fused_triton", "cuda", _GPU_SM70, ("triton",)),
    ("moe_experts", "standard", "fused_triton", "mlu", _MLU, ("triton",)),
    ("moe_experts", "standard", "fused_quack", "cuda", _GPU_NVIDIA_SM90, ("quack",)),
    ("moe_experts", "standard", "fused_npu", "npu", _NPU, ()),
    ("moe_experts", "standard", "fused_mlu", "mlu", _MLU, ("apex",)),
    ("moe_experts", "gpt_oss", "eager", "any", _ANY, ()),
    ("moe_experts", "gpt_oss", "fused_quack", "cuda", _GPU_NVIDIA_SM90, ("quack",)),
    ("moe_experts_lora", "shared", "eager", "any", _ANY, ()),
    ("moe_experts_lora", "shared", "fused_triton", "cuda", _GPU_SM70, ("triton",)),
    ("moe_experts_lora", "shared", "fused_npu", "npu", _NPU, ()),
    ("moe_experts_lora", "independent", "eager", "any", _ANY, ()),
    ("moe_experts_lora", "independent", "fused_triton", "cuda", _GPU_SM70, ("triton",)),
    ("moe_experts_lora", "independent", "fused_npu", "npu", _NPU, ()),
    ("rms_norm", "standard", "eager", "any", _ANY, ()),
    ("rms_norm", "standard", "liger_kernel", "cuda", _GPU, ("liger_kernel",)),
    ("rms_norm", "standard", "npu", "npu", _NPU, ()),
    ("rms_norm", "standard", "triton", "cuda", _GPU, ("triton",)),
    ("rms_norm", "deepseek_v4", "eager", "any", _ANY, ()),
    ("rms_norm", "deepseek_v4", "liger_kernel", "cuda", _GPU, ("liger_kernel",)),
    ("rms_norm", "qwen3_5", "eager", "any", _ANY, ()),
    ("rms_norm", "qwen3_5", "liger_kernel", "cuda", _GPU, ("liger_kernel",)),
    ("rms_norm", "qwen3_5", "npu", "npu", _NPU, ()),
    ("rms_norm", "unweighted", "eager", "any", _ANY, ()),
    ("rms_norm", "unweighted", "liger_kernel", "cuda", _GPU, ("liger_kernel",)),
    ("rope", "full", "eager", "any", _ANY, ()),
    ("rope", "full", "liger_kernel", "cuda", _GPU, ("liger_kernel",)),
    ("rope", "full", "npu", "npu", _NPU, ()),
    ("rope", "partial", "eager", "any", _ANY, ()),
    ("rope", "partial", "npu", "npu", _NPU, ()),
    ("rope", "deepseek_v4", "eager", "any", _ANY, ()),
    ("rope", "deepseek_v4", "triton", "cuda", _GPU, ("triton",)),
    ("rope", "wan", "eager", "any", _ANY, ()),
    ("rope", "wan", "triton", "cuda", _GPU, ("triton",)),
    ("rope", "wan", "npu", "npu", _NPU, ()),
    ("swiglu_mlp", "standard", "eager", "any", _ANY, ()),
    ("swiglu_mlp", "standard", "liger_kernel", "cuda", _GPU, ("liger_kernel",)),
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


def test_builtin_entry_catalog_is_exact():
    """Built-in rows expose the intended device, platform, and package metadata."""
    actual = {
        (*key, _requirement_metadata(entry.requirement), entry.requires) for key, entry in OP_REGISTRY._entries.items()
    }
    expected = set(_BUILTIN_ENTRIES)

    assert len(expected) == len(_BUILTIN_ENTRIES)
    assert actual == expected
