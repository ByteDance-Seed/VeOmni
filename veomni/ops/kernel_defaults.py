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
"""
Default OSS kernel registrations.

Imported at ``veomni.ops`` init time so that all registrations are
available before any model is built.
"""

from __future__ import annotations

from .kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


# ── Liger RMSNorm ─────────────────────────────────────────────────────────────


def _liger_rms_norm_factory():
    """Return a functional RMSNorm kernel (standard formulation, offset=0.0).

    Matches LigerRMSNorm in:
    https://github.com/linkedin/Liger-Kernel/blob/v0.7.0/src/liger_kernel/transformers/rms_norm.py
    """
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction

    def liger_rms_norm(hidden_states, weight, eps):
        return LigerRMSNormFunction.apply(
            hidden_states,
            weight,
            eps,
            0.0,  # offset — standard RMSNorm (no weight shift)
            "llama",  # casting_mode
            False,  # in_place
            None,  # row_mode
        )

    return liger_rms_norm


KERNEL_REGISTRY.register(
    KernelSpec(
        name="liger",
        op_name="rms_norm",
        variant="standard",
        factory=_liger_rms_norm_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="LigerKernel fused RMSNorm",
    )
)


# ── Liger RMSNorm (Qwen3.5 variant: offset=1.0, zeros init) ──────────────────


def _liger_rms_norm_qwen3_5_factory():
    """Return a functional RMSNorm kernel for Qwen3.5 (1+weight centered formulation).

    Uses LigerRMSNormFunction.apply directly with offset=1.0 and casting_mode="gemma".
    Matches LigerRMSNormForQwen3Next in:
    https://github.com/linkedin/Liger-Kernel/blob/v0.7.0/src/liger_kernel/transformers/rms_norm.py
    """
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction

    def liger_rms_norm_qwen3_5(hidden_states, weight, eps):
        return LigerRMSNormFunction.apply(
            hidden_states,
            weight,
            eps,
            1.0,  # offset — Qwen3.5 uses (1 + weight) formulation
            "gemma",  # casting_mode — full fp32
            False,  # in_place
            None,  # row_mode
        )

    return liger_rms_norm_qwen3_5


KERNEL_REGISTRY.register(
    KernelSpec(
        name="liger",
        op_name="rms_norm",
        variant="qwen3_5",
        factory=_liger_rms_norm_qwen3_5_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="LigerKernel fused RMSNorm for Qwen3.5 (1+weight, zeros init, gemma casting)",
    )
)


# ── Liger Rotary Positional Embedding ─────────────────────────────────────────

KERNEL_REGISTRY.register(
    KernelSpec(
        name="liger",
        op_name="apply_rotary_pos_emb",
        variant="full",
        factory=lambda: (
            __import__("liger_kernel.transformers.rope", fromlist=["liger_rotary_pos_emb"]).liger_rotary_pos_emb
        ),
        hardware=HardwareRequirement(device_type="gpu"),
        description="LigerKernel fused RoPE (full head_dim only)",
    )
)


# ── Liger SwiGLU MLP ─────────────────────────────────────────────────────────


def _liger_swiglu_factory():
    """Return a functional SwiGLU MLP kernel using LigerSiLUMulFunction.

    Matches LigerSwiGLUMLP.forward in:
    https://github.com/linkedin/Liger-Kernel/blob/v0.7.0/src/liger_kernel/transformers/swiglu.py
    """
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    def liger_swiglu_forward(self, x):
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))

    return liger_swiglu_forward


KERNEL_REGISTRY.register(
    KernelSpec(
        name="liger",
        op_name="swiglu_mlp",
        variant="standard",
        factory=_liger_swiglu_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="LigerKernel fused SwiGLU MLP",
    )
)


# ── MoE Experts: Triton group-gemm ────────────────────────────────────────────


def _make_moe_experts_adapter(raw_forward):
    """Adapt the raw fused MoE kernel to the OpSlot call signature.

    OpSlot invokes with ``(self, hidden_states, top_k_index, top_k_weights)``
    but the raw kernel uses ``(num_experts, routing_weights, ...)``.
    """

    def adapter(self, hidden_states, top_k_index, top_k_weights):
        return raw_forward(
            num_experts=self.num_experts,
            routing_weights=top_k_weights.to(hidden_states.dtype),
            selected_experts=top_k_index,
            hidden_states=hidden_states,
            fc1_1_weight=None,
            fc1_2_weight=None,
            fc2_weight=self.down_proj,
            fc1_1_2_weight=self.gate_up_proj,
        )

    return adapter


def _triton_group_gemm_factory():
    from .fused_moe.group_gemm import group_gemm_fused_moe_forward

    return _make_moe_experts_adapter(group_gemm_fused_moe_forward)


KERNEL_REGISTRY.register(
    KernelSpec(
        name="triton_group_gemm",
        op_name="moe_experts",
        variant="standard",
        factory=_triton_group_gemm_factory,
        hardware=HardwareRequirement(device_type="gpu", min_compute_capability=70),
        description="Triton group-gemm fused MoE forward",
    )
)


# ── MoE Experts: Quack CUTLASS ────────────────────────────────────────────────


def _quack_cutlass_factory():
    from .fused_moe.quack_gemm import quack_gemm_fused_moe_forward

    return _make_moe_experts_adapter(quack_gemm_fused_moe_forward)


KERNEL_REGISTRY.register(
    KernelSpec(
        name="quack_cutlass",
        op_name="moe_experts",
        variant="standard",
        factory=_quack_cutlass_factory,
        hardware=HardwareRequirement(device_type="gpu", min_compute_capability=90),
        description="Quack CUTLASS/CuTe fused MoE forward (SM90+)",
    )
)


# ── Cross-entropy loss: Liger fused ──────────────────────────────────────────


def _liger_fused_ce_factory():
    """Return ForCausalLMLoss with the Liger fused CE kernel bound via partial.

    This ensures the OpSlot path gets the full preprocessing (label shifting,
    flattening, SP reduction) that ForCausalLMLoss provides, not just the raw kernel.
    """
    from functools import partial

    from veomni.ops.fused_cross_entropy import ForCausalLMLoss
    from veomni.ops.fused_cross_entropy.liger_kernel import fused_liger_kernel_cross_entropy

    return partial(ForCausalLMLoss, cross_entropy_fn=fused_liger_kernel_cross_entropy)


KERNEL_REGISTRY.register(
    KernelSpec(
        name="liger_fused",
        op_name="cross_entropy_loss",
        variant="standard",
        factory=_liger_fused_ce_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="Liger fused linear cross-entropy loss (with label shifting and SP reduction)",
    )
)
