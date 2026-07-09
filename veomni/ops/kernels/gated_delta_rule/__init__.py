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
"""Gated delta-rule kernels: gated RMSNorm, causal conv1d, chunk gated delta rule.

These three ops back Qwen3.5's ``GatedDeltaNet`` linear-attention layer.
Linear attention has many variants; this sub-package is scoped to the
gated delta-rule family that Qwen3.5 uses. Unlike the kernel families in
sibling sub-packages, none of them have a torch eager fallback that
supports varlen training: HF's "eager" path here is essentially "raise at
the first packed-sequence step". The GPU non-eager backends come from the
``flash-linear-attention`` (``fla``) library, plus an alternative
``flash_qla`` implementation of ``chunk_gated_delta_rule`` from QwenLM. On
NPU, all three ops ship a ``npu`` backend backed by vendored Triton kernels
(``triton-ascend``, kept under ``_ascend/``).

Selection is driven by three fields on ``OpsImplementationConfig``:

- ``rms_norm_gated_implementation``    -> ``OpSlot("rms_norm_gated", "standard")``
- ``causal_conv1d_implementation``     -> ``OpSlot("causal_conv1d", "standard")``
- ``chunk_gated_delta_rule_implementation`` ->
  ``OpSlot("chunk_gated_delta_rule", "standard")``

Backends per op:

- ``rms_norm_gated``: ``fla`` (GPU), ``npu``.
- ``causal_conv1d``: ``fla`` (GPU), ``npu``.
- ``chunk_gated_delta_rule``: ``fla`` (GPU), ``flash_qla`` (GPU ``gpu`` extra,
  Hopper SM90), ``npu``.

The ``npu`` ``causal_conv1d`` backend is a thin adapter (``npu_causal_conv1d``)
over the vendored kernel; the ``npu`` ``chunk_gated_delta_rule`` binds the
vendored kernel directly (its signature already matches the call site).
"""

from __future__ import annotations

from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


# ── rms_norm_gated (Torch_npu FusedRMSNormGated) ───────────────────────────────────


def _npu_fused_rms_norm_gated_factory():
    """Return the ``NPUFusedRMSNormGated`` *class*.

    The kernel is consumed inside ``Qwen3_5GatedDeltaNet.__init__`` like a
    constructor — ``self.norm = veomni_rms_norm_gated(dim, eps=..., ...)``
    — so the slot stores the class itself, not an instance. Lazily imported
    via the factory so hosts without torch_npu can still load the module.
    """
    from .npu_rms_norm_gated import NPUFusedRMSNormGated

    return NPUFusedRMSNormGated


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="rms_norm_gated",
        variant="standard",
        factory=_npu_fused_rms_norm_gated_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="NPUFusedRMSNormGated (RMSNorm + SiLU gate fused)",
    )
)


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


# ── causal_conv1d (NPU vendored Triton) ──────────────────────────────────────


def _npu_causal_conv1d_factory():
    """Return the NPU causal conv1d adapter over the vendored Triton kernel.

    Lazily imported so hosts without ``triton-ascend`` can still load this
    module; the import (and its transitive Triton import) only fires when the
    ``npu`` backend is actually selected and bound at ``OpSlot.bind()`` time.
    """
    from .npu_causal_conv1d import causal_conv1d

    return causal_conv1d


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="causal_conv1d",
        variant="standard",
        factory=_npu_causal_conv1d_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="NPU vendored Triton causal conv1d (varlen-aware, MindSpeed-MM)",
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

    Source: https://github.com/QwenLM/FlashQLA — shipped under the ``gpu``
    extra (Hopper SM90 only). FlashQLA mirrors the FLA call signature
    (``query, key, value, g, beta, initial_state, output_final_state,
    use_qk_l2norm_in_kernel, cu_seqlens``) so the call site in
    ``Qwen3_5GatedDeltaNet.forward`` does not need to branch.
    """
    from flash_qla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


# FlashQLA today only ships SM90 kernels — neither older arches (Ampere, Ada)
# nor newer ones (Blackwell SM100/SM120) work; the SM10x wheels are WIP per
# https://github.com/QwenLM/FlashQLA/issues/2. Pin exactly SM90 so the
# registry rejects the kernel early at OpSlot.bind() time on every other arch
# (and we drop max_compute_capability once upstream adds support).
KERNEL_REGISTRY.register(
    KernelSpec(
        name="flash_qla",
        op_name="chunk_gated_delta_rule",
        variant="standard",
        factory=_flash_qla_chunk_gated_delta_rule_factory,
        hardware=HardwareRequirement(device_type="gpu", min_compute_capability=90, max_compute_capability=90),
        description="QwenLM FlashQLA chunk gated delta rule (Hopper SM90 only, alternative TileLang implementation)",
    )
)


# ── chunk_gated_delta_rule (NPU vendored Triton) ─────────────────────────────


def _npu_chunk_gated_delta_rule_factory():
    """Return the vendored NPU chunk gated delta rule kernel directly.

    Its public signature already matches the ``Qwen3_5GatedDeltaNet.forward``
    call site (``q, k, v, g, beta, ..., use_qk_l2norm_in_kernel, cu_seqlens``)
    and it returns ``(o, final_state)``, so no adapter is needed. Lazily
    imported so the transitive Triton import only fires under the ``npu``
    backend.
    """
    from ._ascend.chunk_gated_delta_rule_mm import chunk_gated_delta_rule

    return chunk_gated_delta_rule


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="chunk_gated_delta_rule",
        variant="standard",
        factory=_npu_chunk_gated_delta_rule_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="NPU vendored Triton chunk gated delta rule (varlen-aware, MindSpeed-MM)",
    )
)
