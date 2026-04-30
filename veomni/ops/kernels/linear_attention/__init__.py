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
"""Linear-attention kernels: gated RMSNorm, causal conv1d, chunk gated delta rule.

These three ops back Qwen3.5's ``GatedDeltaNet`` linear-attention layer. Unlike
the kernel families in sibling sub-packages, none of them have a torch eager
fallback that supports varlen training: HF's "eager" path here is essentially
"raise at the first packed-sequence step". The non-eager backends come from
the ``flash-linear-attention`` (``fla``) library, plus an alternative
``flash_qla`` implementation of ``chunk_gated_delta_rule`` from QwenLM.

Selection is driven by three fields on ``OpsImplementationConfig``:

- ``rms_norm_gated_implementation``    -> ``OpSlot("rms_norm_gated", "standard")``
- ``causal_conv1d_implementation``     -> ``OpSlot("causal_conv1d", "standard")``
- ``chunk_gated_delta_rule_implementation`` ->
  ``OpSlot("chunk_gated_delta_rule", "standard")``

Currently ``rms_norm_gated`` and ``causal_conv1d`` ship a single non-eager
backend (``fla``); ``chunk_gated_delta_rule`` additionally accepts
``flash_qla`` for users who install the optional ``flash-qla`` extra
(``pip install veomni[flash-qla]`` / ``uv sync --extra flash-qla``).
"""

from __future__ import annotations

from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


# ── rms_norm_gated (FLA FusedRMSNormGated) ───────────────────────────────────


def _fla_fused_rms_norm_gated_factory():
    """Return the ``FusedRMSNormGated`` *class* from ``fla.modules``.

    The kernel is consumed inside ``Qwen3_5GatedDeltaNet.__init__`` like a
    constructor — ``self.norm = veomni_rms_norm_gated(dim, eps=..., ...)``
    — so the slot stores the class itself, not an instance. Lazily imported
    via the factory so hosts without ``flash-linear-attention`` installed
    can still load the module.
    """
    from fla.modules import FusedRMSNormGated

    return FusedRMSNormGated


KERNEL_REGISTRY.register(
    KernelSpec(
        name="fla",
        op_name="rms_norm_gated",
        variant="standard",
        factory=_fla_fused_rms_norm_gated_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="flash-linear-attention FusedRMSNormGated (RMSNorm + SiLU gate fused)",
    )
)


# ── causal_conv1d (FLA Triton causal conv) ───────────────────────────────────


def _fla_causal_conv1d_factory():
    """Return ``fla.modules.convolution.causal_conv1d`` — the Triton varlen
    depthwise conv used by Qwen3.5's GatedDeltaNet pre-mixer."""
    from fla.modules.convolution import causal_conv1d

    return causal_conv1d


KERNEL_REGISTRY.register(
    KernelSpec(
        name="fla",
        op_name="causal_conv1d",
        variant="standard",
        factory=_fla_causal_conv1d_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="flash-linear-attention causal conv1d (Triton, varlen-aware)",
    )
)


# ── chunk_gated_delta_rule (FLA + FlashQLA) ──────────────────────────────────


def _fla_chunk_gated_delta_rule_factory():
    """Return ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


KERNEL_REGISTRY.register(
    KernelSpec(
        name="fla",
        op_name="chunk_gated_delta_rule",
        variant="standard",
        factory=_fla_chunk_gated_delta_rule_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="flash-linear-attention chunk gated delta rule (Triton, varlen-aware)",
    )
)


def _flash_qla_chunk_gated_delta_rule_factory():
    """Return the FlashQLA implementation of chunk gated delta rule.

    Source: https://github.com/QwenLM/FlashQLA — opt-in via the ``flash-qla``
    pyproject extra. FlashQLA mirrors the FLA call signature
    (``query, key, value, g, beta, initial_state, output_final_state,
    use_qk_l2norm_in_kernel, cu_seqlens``) so the call site in
    ``Qwen3_5GatedDeltaNet.forward`` does not need to branch.
    """
    from flash_qla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


# FlashQLA only ships SM90 kernels today (Hopper); selecting it on Ampere or
# older raises early at OpSlot.bind() time with the standard registry message.
# Tracking upstream: https://github.com/QwenLM/FlashQLA/issues/2
KERNEL_REGISTRY.register(
    KernelSpec(
        name="flash_qla",
        op_name="chunk_gated_delta_rule",
        variant="standard",
        factory=_flash_qla_chunk_gated_delta_rule_factory,
        hardware=HardwareRequirement(device_type="gpu", min_compute_capability=90),
        description="QwenLM FlashQLA chunk gated delta rule (Hopper SM90, alternative Triton implementation)",
    )
)
