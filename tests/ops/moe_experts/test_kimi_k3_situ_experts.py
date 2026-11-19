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

"""Kimi K3 situ MoE eager expert loop: single-device parity and shared-core wiring."""

import torch

import veomni.ops.kernels.moe_experts.kimi_k3.eager as moe_eager_module
from veomni.ops.kernels.situ_and_mul.kimi_k3.eager import situ_eager


def _run_wrapper(hidden_states, routing_weights, selected_experts, fc1_w, fc2_w, beta, linear_beta):
    return moe_eager_module.wrapper(
        hidden_states,
        routing_weights,
        selected_experts,
        fc1_w,
        fc2_w,
        num_experts=fc1_w.shape[0],
        beta=beta,
        linear_beta=linear_beta,
    )


def _manual_reference(hidden_states, routing_weights, selected_experts, fc1_w, fc2_w, beta, linear_beta):
    output = torch.zeros_like(hidden_states)
    for t in range(hidden_states.shape[0]):
        for k in range(selected_experts.shape[1]):
            e = int(selected_experts[t, k])
            gate, up = torch.nn.functional.linear(hidden_states[t], fc1_w[e]).chunk(2, dim=-1)
            situ = situ_eager(gate, up, beta=beta, linear_beta=linear_beta).to(hidden_states.dtype)
            y = torch.nn.functional.linear(situ, fc2_w[e]) * routing_weights[t, k]
            output[t] += y
    return output


def test_single_device_expert_loop_matches_manual_reference():
    torch.manual_seed(21)
    num_experts, d, tokens, top_k = 4, 16, 12, 2
    hidden_states = torch.randn(tokens, d)
    fc1_w = torch.randn(num_experts, 2 * d, d) * 0.1
    fc2_w = torch.randn(num_experts, d, d) * 0.1
    routing_weights = torch.rand(tokens, top_k)
    selected_experts = torch.randint(0, num_experts, (tokens, top_k))

    out = _run_wrapper(hidden_states, routing_weights, selected_experts, fc1_w, fc2_w, beta=1.0, linear_beta=None)
    ref = _manual_reference(hidden_states, routing_weights, selected_experts, fc1_w, fc2_w, beta=1.0, linear_beta=None)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    out = _run_wrapper(hidden_states, routing_weights, selected_experts, fc1_w, fc2_w, beta=0.7, linear_beta=1.3)
    ref = _manual_reference(hidden_states, routing_weights, selected_experts, fc1_w, fc2_w, beta=0.7, linear_beta=1.3)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_expert_loop_runs_shared_situ_eager_core(monkeypatch):
    """Both expert loops route through the single ``situ_eager`` definition.

    The single-device loop and the EP dispatch share the SituGLU math with
    the op registry's ``eager`` row by construction; a spy pins that the
    wrapper actually dispatches through it (no re-derived inline copy).
    """
    calls = []
    real = moe_eager_module.situ_eager

    def _spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(moe_eager_module, "situ_eager", _spy)

    torch.manual_seed(23)
    num_experts, d, tokens, top_k = 2, 8, 6, 1
    hidden_states = torch.randn(tokens, d)
    fc1_w = torch.randn(num_experts, 2 * d, d) * 0.1
    fc2_w = torch.randn(num_experts, d, d) * 0.1
    routing_weights = torch.rand(tokens, top_k)
    selected_experts = torch.randint(0, num_experts, (tokens, top_k))

    out = _run_wrapper(hidden_states, routing_weights, selected_experts, fc1_w, fc2_w, beta=1.0, linear_beta=None)
    assert calls, "the expert loop must call situ_eager"
    assert out.shape == (tokens, d)
