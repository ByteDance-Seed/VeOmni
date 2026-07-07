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

"""CPU-runnable tests for the eager reference implementations used by VeOmni.

These tests don't require any accelerator.  They are useful when:
  * Iterating on kernel math changes locally on a Mac (no NPU).
  * Catching regressions in the eager fallback that the NPU/GPU
    kernels are compared against.

The tests are deliberately small so they run in <1s on a laptop.

``load_balancing_loss_pytorch`` expects ``gate_logits`` to be a tuple
of per-layer tensors with shape ``[batch_size * seq_len, num_experts]``;
that convention is used throughout these tests.

Algebraic reference (Switch Transformer):
    loss = E * sum_e(f_e * P_e)
where f_e = (count of times e is in top-k) / N and P_e = mean probability
assigned to e.  With uniform probs (1/E), f_e = top_k / E, P_e = 1/E,
so loss = E * E * (top_k/E) * (1/E) = top_k.
"""

import torch
import torch.nn.functional as F

import veomni.ops  # noqa: F401 -- trigger KERNEL_REGISTRY registrations
from veomni.ops.dispatch import OpSlot
from veomni.ops.kernels.load_balancing_loss.eager import load_balancing_loss_pytorch


# ---------------------------------------------------------------------------
# Load balancing loss (pure-PyTorch) eager reference
# ---------------------------------------------------------------------------


class TestLoadBalancingLossEager:
    """Sanity tests for the load-balancing-loss pure-PyTorch implementation.

    Convention: ``gate_logits[i]`` has shape ``[N, E]`` where
    ``N = batch_size * seq_len``.
    """

    def test_uniform_distribution_well_conditioned(self):
        """With uniform random gate logits, the loss is in (0, top_k + 1)."""
        num_experts, top_k, num_layers, N = 8, 2, 3, 1024
        torch.manual_seed(0)
        gate_logits = tuple(torch.randn(N, num_experts, dtype=torch.float32) for _ in range(num_layers))
        loss = load_balancing_loss_pytorch(gate_logits, num_experts, top_k, None)
        assert loss.item() > 0
        assert loss.item() < 5.0

    def test_perfectly_balanced_gives_top_k(self):
        """All-equal gate logits -> uniform routing -> loss = top_k.

        With softmax(0) = 1/E, every expert is selected with equal
        probability.  Per the algebraic reference, the loss evaluates
        to exactly ``top_k``.
        """
        num_experts, top_k, N = 4, 2, 64
        gate_logits = (torch.zeros(N, num_experts, dtype=torch.float32),)
        loss = load_balancing_loss_pytorch(gate_logits, num_experts, top_k, None)
        assert torch.allclose(loss, torch.tensor(float(top_k)), atol=1e-5)

    def test_one_hot_gives_max_loss(self):
        """A perfectly imbalanced router (one expert always wins) gives loss = E.

        When f_0 = 1, f_{e!=0} = 0, and P_0 = 1 (others = 0),
        loss = E * (1 * 1) = E.
        """
        num_experts, top_k, N = 4, 2, 64
        # Make logit[..., 0] huge -> every token routes to expert 0.
        gate_logits = torch.full((N, num_experts), -1e9, dtype=torch.float32)
        gate_logits[:, 0] = 1e9
        gate_logits = (gate_logits,)
        loss = load_balancing_loss_pytorch(gate_logits, num_experts, top_k, None)
        # If top_k=1 routing is forced, loss = E.
        # With top_k=2 and a one-hot logit distribution, the second
        # selected expert is also forced to expert 0; loss still = E.
        assert torch.allclose(loss, torch.tensor(float(num_experts)), atol=1e-3)

    def test_handles_attention_mask(self):
        """When an attention_mask is provided, only unmasked tokens count.

        We mask out the second half of the sequence and confirm the
        loss matches computing it on only the unmasked subset.
        """
        num_experts, top_k = 4, 2
        N_full, N_keep = 64, 32
        torch.manual_seed(1)
        gate_logits_full = torch.randn(N_full, num_experts, dtype=torch.float32)
        # Mask shape: [batch_size, seq_len].  Here batch_size=1, seq_len=N_full.
        mask = torch.tensor([[1.0] * N_keep + [0.0] * (N_full - N_keep)], dtype=torch.float32)
        loss_masked = load_balancing_loss_pytorch((gate_logits_full,), num_experts, top_k, mask)
        # Reference: compute the loss on only the unmasked tokens.
        gate_logits_keep = (gate_logits_full[:N_keep, :].clone(),)
        loss_keep = load_balancing_loss_pytorch(gate_logits_keep, num_experts, top_k, None)
        assert torch.allclose(loss_masked, loss_keep, atol=1e-5)

    def test_multiple_layers_invariant(self):
        """With identical layer inputs, the loss is independent of the
        number of layers (both counts and weight double).
        """
        num_experts, top_k, N = 4, 2, 64
        torch.manual_seed(2)
        gl = torch.randn(N, num_experts, dtype=torch.float32)
        loss1 = load_balancing_loss_pytorch((gl,), num_experts, top_k, None)
        loss2 = load_balancing_loss_pytorch((gl, gl), num_experts, top_k, None)
        assert torch.allclose(loss1, loss2, atol=1e-4)

    def test_none_gate_logits_returns_zero(self):
        """Defensive: a None gate_logits returns 0 (HF compat)."""
        assert load_balancing_loss_pytorch(None, 4, 2, None) == 0

    def test_non_tuple_gate_logits_returns_zero(self):
        """Defensive: a non-tuple gate_logits returns 0 (HF compat)."""
        assert load_balancing_loss_pytorch(torch.randn(4, 8), 4, 2, None) == 0


# ---------------------------------------------------------------------------
# OpSlot CPU behavior (does not require the NPU or GPU)
# ---------------------------------------------------------------------------


class TestOpSlotEagerBinding:
    """When bound to ``eager``, the slot must not call any kernel."""

    def test_eager_bound_does_not_invoke_kernel(self):
        """``bind('eager')`` keeps ``use_non_eager_impl`` False."""
        slot = OpSlot("rms_norm", "standard")
        slot.bind("eager")
        assert slot.use_non_eager_impl is False
        assert slot.bound_kernel() is None

    def test_rebind_to_eager_keeps_flag_false(self):
        """Re-binding to eager after a previous eager bind is a no-op."""
        slot = OpSlot("rms_norm", "standard")
        slot.bind("eager")
        slot.bind("eager")
        assert slot.use_non_eager_impl is False


# ---------------------------------------------------------------------------
# Eager MoE numerical-ordering guardrail (CPU-only)
# ---------------------------------------------------------------------------


def _eager_moe_routing_before_fc2(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Reference eager MoE with routing weights applied before fc2."""
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden_states[token_idx]
        gate = F.linear(x, fc1_1_weight[idx])
        up = F.linear(x, fc1_2_weight[idx])
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        y = F.silu(gate) * up
        y = y * routing_weights[token_idx, top_k_pos, None]
        y = F.linear(y, fc2_weight[idx])
        output.index_add_(0, token_idx, y.to(output.dtype))

    return output


def _eager_moe_routing_after_fc2(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Alternative eager MoE ordering used by the broken parity oracle."""
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden_states[token_idx]
        gate = F.linear(x, fc1_1_weight[idx])
        up = F.linear(x, fc1_2_weight[idx])
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        y = F.linear(F.silu(gate) * up, fc2_weight[idx])
        y = y * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, y.to(output.dtype))

    return output


class TestEagerMoeOrdering:
    """CPU-only regression tests for the eager MoE parity oracle."""

    def test_routing_weight_order_matters_in_bf16(self):
        """The two mathematically equivalent orderings are not bf16-identical."""
        torch.manual_seed(0)

        num_tokens, num_experts, hidden_dim, ffn_dim, topk = 32, 4, 64, 32, 2
        dtype = torch.bfloat16

        hidden_states = 0.1 * torch.randn(num_tokens, hidden_dim, dtype=dtype)
        router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), topk, dim=-1)
        routing_weights = routing_weights.to(dtype)

        fc1_1_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, dtype=dtype)
        fc1_2_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, dtype=dtype)
        fc2_weight = 0.1 * torch.randn(num_experts, hidden_dim, ffn_dim, dtype=dtype)

        before = _eager_moe_routing_before_fc2(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            swiglu_limit=0.1,
        )
        after = _eager_moe_routing_after_fc2(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            swiglu_limit=0.1,
        )

        assert not torch.equal(before, after)
