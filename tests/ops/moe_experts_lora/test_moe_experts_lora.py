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
# See the License for the specific language governing limitations
# under the License.

"""Registry, hand-written oracle, and fused parity tests for ``moe_experts_lora``."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from veomni.ops import OP_REGISTRY, VeomniOp, resolve_op
from veomni.ops.kernels.moe_experts.shared.dispatch import expert_histogram, moe_gather, moe_scatter
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type
from veomni.utils.import_utils import is_fused_moe_available


_FWD_L2REL_TOL = 0.02
_GRAD_L2REL_TOL = 0.02
_LORA_KEYS = ("lora_a_gate", "lora_b_gate", "lora_a_up", "lora_b_up", "lora_a_down", "lora_b_down")
_SCALES = dict(lora_scale_gate=0.5, lora_scale_up=0.5, lora_scale_down=0.5)


def _l2_rel(actual: torch.Tensor, ref: torch.Tensor) -> float:
    a = actual.float()
    r = ref.float()
    ref_norm = r.norm().item()
    if ref_norm == 0.0:
        return (a - r).norm().item()
    return ((a - r).norm() / ref_norm).item()


def _lora_tensors(variant: str, *, E: int, H: int, I: int, r: int, device, dtype):
    if variant == "shared":
        shapes = [(r, H), (I, r), (r, H), (I, r), (r, I), (H, r)]
    else:
        shapes = [(E, r, H), (E, I, r), (E, r, H), (E, I, r), (E, r, I), (E, H, r)]
    return [0.02 * torch.randn(*shape, device=device, dtype=dtype) for shape in shapes]


def _call(impl: str, variant: str, hidden, routing, selected, fc1, fc2, loras, *, num_experts: int):
    return VeomniOp("moe_experts_lora", variant, impl)(
        hidden,
        routing,
        selected,
        fc1,
        fc2,
        *loras,
        num_experts=num_experts,
        **_SCALES,
    )


def _run_fused_vs_eager(impl: str, variant: str):
    torch.manual_seed(0)
    device = torch.device("npu" if impl == "fused_npu" else "cuda")
    dtype = torch.bfloat16
    B, H, I, E, top_k, r = 32, 64, 96, 4, 2, 8
    hidden = 0.1 * torch.randn(B, H, device=device, dtype=dtype)
    routing = torch.softmax(torch.randn(B, top_k, device=device, dtype=torch.float32), dim=-1).to(dtype)
    selected = torch.randint(0, E, (B, top_k), device=device)
    fc1 = (0.05 * torch.randn(E, 2 * I, H, device=device, dtype=dtype)).detach()
    fc2 = (0.05 * torch.randn(E, H, I, device=device, dtype=dtype)).detach()
    loras = _lora_tensors(variant, E=E, H=H, I=I, r=r, device=device, dtype=dtype)
    train_base_weights = impl == "fused_triton"

    hidden_e = hidden.detach().requires_grad_(True)
    hidden_f = hidden.detach().requires_grad_(True)
    routing_e = routing.detach().clone().requires_grad_(train_base_weights)
    routing_f = routing.detach().clone().requires_grad_(train_base_weights)
    fc1_e = fc1.detach().clone().requires_grad_(train_base_weights)
    fc1_f = fc1.detach().clone().requires_grad_(train_base_weights)
    fc2_e = fc2.detach().clone().requires_grad_(train_base_weights)
    fc2_f = fc2.detach().clone().requires_grad_(train_base_weights)
    lora_e = [t.detach().clone().requires_grad_(True) for t in loras]
    lora_f = [t.detach().clone().requires_grad_(True) for t in loras]

    out_e = _call("eager", variant, hidden_e, routing_e, selected, fc1_e, fc2_e, lora_e, num_experts=E)
    out_f = _call(impl, variant, hidden_f, routing_f, selected, fc1_f, fc2_f, lora_f, num_experts=E)
    fwd_l2 = _l2_rel(out_f, out_e)
    assert fwd_l2 <= _FWD_L2REL_TOL, (
        f"[{impl}/{variant}] forward L2 rel {fwd_l2:.4%} > {_FWD_L2REL_TOL:.2%} "
        f"(eager_norm={out_e.float().norm().item():.3e})"
    )

    go = (0.1 * torch.randn_like(out_e)).detach()
    out_e.backward(go)
    out_f.backward(go)
    gradient_pairs = [("hidden_states", hidden_e, hidden_f)]
    if train_base_weights:
        gradient_pairs.extend(
            (
                ("routing_weights", routing_e, routing_f),
                ("fc1_1_2_weight", fc1_e, fc1_f),
                ("fc2_weight", fc2_e, fc2_f),
            )
        )
    gradient_pairs.extend(zip(_LORA_KEYS, lora_e, lora_f, strict=True))
    for name, eager_input, fused_input in gradient_pairs:
        l2 = _l2_rel(fused_input.grad, eager_input.grad)
        assert l2 <= _GRAD_L2REL_TOL, (
            f"[{impl}/{variant}] {name} grad L2 rel {l2:.4%} > {_GRAD_L2REL_TOL:.2%} "
            f"(eager_norm={eager_input.grad.float().norm().item():.3e})"
        )


def _manual_moe_lora_oracle(
    variant: str,
    hidden: torch.Tensor,
    routing: torch.Tensor,
    selected: torch.Tensor,
    fc1: torch.Tensor,
    fc2: torch.Tensor,
    loras: list[torch.Tensor],
) -> torch.Tensor:
    """Compute routed MoE-LoRA directly, without registry or scatter/grouped-GEMM helpers."""
    lora_a_gate, lora_b_gate, lora_a_up, lora_b_up, lora_a_down, lora_b_down = loras
    outputs = []
    for token_index, token in enumerate(hidden):
        token_output = torch.zeros_like(token)
        for slot_index in range(selected.shape[1]):
            expert_index = int(selected[token_index, slot_index])
            if variant == "shared":
                expert_loras = (
                    lora_a_gate,
                    lora_b_gate,
                    lora_a_up,
                    lora_b_up,
                    lora_a_down,
                    lora_b_down,
                )
            else:
                expert_loras = tuple(lora[expert_index] for lora in loras)
            a_gate, b_gate, a_up, b_up, a_down, b_down = expert_loras

            base_gate, base_up = F.linear(token, fc1[expert_index]).chunk(2, dim=-1)
            gate = base_gate + F.linear(F.linear(token, a_gate), b_gate) * _SCALES["lora_scale_gate"]
            up = base_up + F.linear(F.linear(token, a_up), b_up) * _SCALES["lora_scale_up"]
            intermediate = F.silu(gate) * up
            expert_output = F.linear(intermediate, fc2[expert_index])
            expert_output = expert_output + (
                F.linear(F.linear(intermediate, a_down), b_down) * _SCALES["lora_scale_down"]
            )
            token_output = token_output + routing[token_index, slot_index] * expert_output
        outputs.append(token_output)
    return torch.stack(outputs)


def _make_lora_leaf(*shape: int, dtype: torch.dtype, device: torch.device, scale: float = 0.02) -> torch.Tensor:
    return (torch.randn(*shape, dtype=dtype, device=device) * scale).detach().requires_grad_(True)


def _build_lora_leaves(variant: str, *, E: int, H: int, I: int, r: int, dtype: torch.dtype, device: torch.device):
    if variant == "shared":
        return {
            "lora_a_gate": _make_lora_leaf(r, H, dtype=dtype, device=device),
            "lora_b_gate": _make_lora_leaf(I, r, dtype=dtype, device=device),
            "lora_a_up": _make_lora_leaf(r, H, dtype=dtype, device=device),
            "lora_b_up": _make_lora_leaf(I, r, dtype=dtype, device=device),
            "lora_a_down": _make_lora_leaf(r, I, dtype=dtype, device=device),
            "lora_b_down": _make_lora_leaf(H, r, dtype=dtype, device=device),
        }
    return {
        "lora_a_gate": _make_lora_leaf(E, r, H, dtype=dtype, device=device),
        "lora_b_gate": _make_lora_leaf(E, I, r, dtype=dtype, device=device),
        "lora_a_up": _make_lora_leaf(E, r, H, dtype=dtype, device=device),
        "lora_b_up": _make_lora_leaf(E, I, r, dtype=dtype, device=device),
        "lora_a_down": _make_lora_leaf(E, r, I, dtype=dtype, device=device),
        "lora_b_down": _make_lora_leaf(E, H, r, dtype=dtype, device=device),
    }


@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_moe_experts_lora_registered_impls(variant):
    registered = OP_REGISTRY.list_registered("moe_experts_lora", variant)
    assert "eager" in registered
    assert "fused_triton" in registered
    assert "fused_npu" in registered
    assert "fused_quack" not in registered
    assert "fused_mlu" not in registered


@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_moe_experts_lora_eager_resolves(variant):
    entry = resolve_op("moe_experts_lora", variant, "eager")
    assert entry.wrapper is not None
    handle = VeomniOp("moe_experts_lora", variant, "eager")
    assert handle.impl == "eager"


@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_moe_experts_lora_eager_forward_smoke(variant):
    torch.manual_seed(0)
    B, H, I, E, top_k, r = 8, 16, 24, 4, 2, 4
    hidden = torch.randn(B, H)
    routing = torch.softmax(torch.randn(B, top_k), dim=-1)
    selected = torch.randint(0, E, (B, top_k))
    fc1 = torch.randn(E, 2 * I, H) * 0.05
    fc2 = torch.randn(E, H, I) * 0.05
    loras = _lora_tensors(variant, E=E, H=H, I=I, r=r, device=hidden.device, dtype=hidden.dtype)
    out = _call("eager", variant, hidden, routing, selected, fc1, fc2, loras, num_experts=E)
    assert out.shape == hidden.shape


@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_moe_experts_lora_eager_matches_manual_oracle_forward_and_all_gradients(variant):
    torch.manual_seed(11)
    B, H, I, E, top_k, r = 4, 5, 7, 3, 2, 3
    dtype = torch.float64
    hidden = torch.randn(B, H, dtype=dtype)
    routing = torch.softmax(torch.randn(B, top_k, dtype=dtype), dim=-1)
    selected = torch.tensor([[0, 1], [2, 0], [1, 2], [2, 1]])
    fc1 = torch.randn(E, 2 * I, H, dtype=dtype) * 0.2
    fc2 = torch.randn(E, H, I, dtype=dtype) * 0.2
    loras = _lora_tensors(variant, E=E, H=H, I=I, r=r, device=hidden.device, dtype=dtype)

    eager_inputs = [value.detach().clone().requires_grad_(True) for value in (hidden, routing, fc1, fc2, *loras)]
    oracle_inputs = [value.detach().clone().requires_grad_(True) for value in (hidden, routing, fc1, fc2, *loras)]
    hidden_e, routing_e, fc1_e, fc2_e, *lora_e = eager_inputs
    hidden_o, routing_o, fc1_o, fc2_o, *lora_o = oracle_inputs

    output_e = _call("eager", variant, hidden_e, routing_e, selected, fc1_e, fc2_e, lora_e, num_experts=E)
    output_o = _manual_moe_lora_oracle(variant, hidden_o, routing_o, selected, fc1_o, fc2_o, lora_o)
    torch.testing.assert_close(output_e, output_o, atol=1e-10, rtol=1e-10)

    grad_output = torch.randn_like(output_e)
    eager_grads = torch.autograd.grad(output_e, eager_inputs, grad_outputs=grad_output)
    oracle_grads = torch.autograd.grad(output_o, oracle_inputs, grad_outputs=grad_output)
    gradient_names = ("hidden_states", "routing_weights", "fc1_1_2_weight", "fc2_weight", *_LORA_KEYS)
    for name, actual, expected in zip(gradient_names, eager_grads, oracle_grads, strict=True):
        torch.testing.assert_close(
            actual,
            expected,
            atol=1e-10,
            rtol=1e-10,
            msg=lambda message, name=name: f"{name}: {message}",
        )


@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_moe_experts_lora_triton_available_on_cuda(variant):
    if not IS_CUDA_AVAILABLE:
        pytest.skip("triton LoRA row requires a GPU")
    assert "fused_triton" in OP_REGISTRY.list_available("moe_experts_lora", variant)
    resolve_op("moe_experts_lora", variant, "fused_triton")


@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_moe_experts_lora_npu_available_on_npu(variant):
    if not IS_NPU_AVAILABLE:
        pytest.skip("npu LoRA row is NPU-gated")
    assert "fused_npu" in OP_REGISTRY.list_available("moe_experts_lora", variant)
    resolve_op("moe_experts_lora", variant, "fused_npu")


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton moe_experts_lora needs a GPU + triton",
)
@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_triton_matches_eager(variant):
    _run_fused_vs_eager("fused_triton", variant)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="npu moe_experts_lora needs torch_npu")
@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_npu_matches_eager(variant):
    _run_fused_vs_eager("fused_npu", variant)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton moe_experts_lora needs a GPU + triton",
)
@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_triton_ep_class_matches_nonep_single_rank(variant):
    """EP autograd class output and LoRA grads match the non-EP class on one rank."""
    from veomni.ops.kernels.moe_experts_lora.independent.triton import (
        EPMergedFc1IndependentLoRAGroupGemm,
        MergedFc1IndependentTritonFusedLoRAMoeExpertFunction,
    )
    from veomni.ops.kernels.moe_experts_lora.shared.triton import (
        EPMergedFc1SharedLoRAGroupGemm,
        MergedFc1TritonFusedLoRAMoeExpertFunction,
    )

    classes = {
        "shared": (EPMergedFc1SharedLoRAGroupGemm, MergedFc1TritonFusedLoRAMoeExpertFunction),
        "independent": (EPMergedFc1IndependentLoRAGroupGemm, MergedFc1IndependentTritonFusedLoRAMoeExpertFunction),
    }
    ep_cls, nonep_cls = classes[variant]

    dev = torch.device(get_device_type())
    dtype = torch.bfloat16
    B, H, I, E, top_k, r = 32, 64, 96, 4, 2, 8
    scale_gate, scale_up, scale_down = 0.5, 0.5, 0.5

    torch.manual_seed(0)
    hidden_states = torch.randn(B, H, dtype=dtype, device=dev)
    top_k_index = torch.randint(0, E, (B, top_k), device=dev)
    top_k_weights = torch.softmax(torch.randn(B, top_k, dtype=torch.float32, device=dev), dim=-1).to(dtype)

    splits = expert_histogram(top_k_index, E)
    scatter_index = top_k_index.flatten().argsort(stable=True).argsort().int().view(top_k_index.shape)
    permute_tokens = moe_scatter(hidden_states, scatter_index)
    cumsum = torch.cumsum(splits, dim=0)
    T = permute_tokens.shape[0]
    scattered_gate_weights = torch.empty(T, 1, dtype=dtype, device=dev)
    scattered_gate_weights[scatter_index.flatten()] = top_k_weights.reshape(-1, 1)

    gate_up_proj = (torch.randn(E, 2 * I, H, dtype=dtype, device=dev) * 0.05).detach()
    down_proj = (torch.randn(E, H, I, dtype=dtype, device=dev) * 0.05).detach()

    def _build_branch(*, ep: bool):
        torch.manual_seed(123)
        lora = _build_lora_leaves(variant, E=E, H=H, I=I, r=r, dtype=dtype, device=dev)
        if ep:
            out = ep_cls.apply(
                permute_tokens,
                cumsum,
                gate_up_proj,
                down_proj,
                *(lora[k] for k in _LORA_KEYS),
                scale_gate,
                scale_up,
                scale_down,
            )
        else:
            out = nonep_cls.apply(
                E,
                top_k_weights,
                top_k_index,
                hidden_states,
                gate_up_proj,
                down_proj,
                *(lora[k] for k in _LORA_KEYS),
                scale_gate,
                scale_up,
                scale_down,
            )
        return out, lora

    nonep_out, nonep_lora = _build_branch(ep=False)
    ep_permuted, ep_lora = _build_branch(ep=True)

    with torch.no_grad():
        ep_out = moe_gather(ep_permuted.detach() * scattered_gate_weights, scatter_index).reshape(hidden_states.shape)
    fwd_l2 = _l2_rel(ep_out, nonep_out.detach())
    assert fwd_l2 <= _FWD_L2REL_TOL, f"[{variant}] EP-vs-non-EP forward L2 rel {fwd_l2:.4%} > {_FWD_L2REL_TOL:.2%}"

    torch.manual_seed(456)
    grad_out = (torch.randn(B, H, dtype=dtype, device=dev) * 0.1).detach()
    grad_permuted = (moe_scatter(grad_out, scatter_index) * scattered_gate_weights).detach()
    nonep_grads = dict(
        zip(
            _LORA_KEYS,
            torch.autograd.grad(nonep_out, [nonep_lora[k] for k in _LORA_KEYS], grad_outputs=grad_out),
            strict=True,
        )
    )
    ep_grads = dict(
        zip(
            _LORA_KEYS,
            torch.autograd.grad(ep_permuted, [ep_lora[k] for k in _LORA_KEYS], grad_outputs=grad_permuted),
            strict=True,
        )
    )
    for name in _LORA_KEYS:
        l2 = _l2_rel(ep_grads[name], nonep_grads[name])
        assert l2 <= _GRAD_L2REL_TOL, f"[{variant}] {name}: EP-vs-non-EP grad L2 rel {l2:.4%} > {_GRAD_L2REL_TOL:.2%}"


@pytest.fixture
def _single_rank_dist():
    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29611")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created = True
    try:
        yield
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="npu moe_experts_lora needs torch_npu")
@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_npu_ep_matches_nonep_single_rank(variant, _single_rank_dist):
    if variant == "shared":
        from veomni.ops.kernels.moe_experts_lora.shared.npu import (
            _npu_ep_fused_lora_moe_forward,
            _npu_fused_lora_moe_forward,
        )
    else:
        from veomni.ops.kernels.moe_experts_lora.independent.npu import (
            _npu_ep_fused_lora_moe_forward,
            _npu_fused_lora_moe_forward,
        )

    dev = torch.device(get_device_type())
    dtype = torch.bfloat16
    B, H, I, E, top_k, r = 32, 64, 96, 4, 2, 8
    grad_keys = ("hidden_states",) + _LORA_KEYS

    torch.manual_seed(0)
    selected_experts = torch.randint(0, E, (B, top_k), device=dev)
    routing_weights = torch.softmax(torch.randn(B, top_k, dtype=torch.float32, device=dev), dim=-1).to(dtype)
    gate_up_proj = (torch.randn(E, 2 * I, H, dtype=dtype, device=dev) * 0.05).detach()
    down_proj = (torch.randn(E, H, I, dtype=dtype, device=dev) * 0.05).detach()
    torch.manual_seed(1)
    hidden_states_base = torch.randn(B, H, dtype=dtype, device=dev)

    def _run(*, ep: bool):
        torch.manual_seed(123)
        lora = _build_lora_leaves(variant, E=E, H=H, I=I, r=r, dtype=dtype, device=dev)
        h = hidden_states_base.detach().clone().requires_grad_(True)
        kwargs = dict(
            num_experts=E,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=h,
            fc1_1_2_weight=gate_up_proj,
            fc2_weight=down_proj,
            lora_a_gate=lora["lora_a_gate"],
            lora_b_gate=lora["lora_b_gate"],
            lora_a_up=lora["lora_a_up"],
            lora_b_up=lora["lora_b_up"],
            lora_a_down=lora["lora_a_down"],
            lora_b_down=lora["lora_b_down"],
            **_SCALES,
        )
        if ep:
            out = _npu_ep_fused_lora_moe_forward(ep_group=None, **kwargs)
        else:
            out = _npu_fused_lora_moe_forward(**kwargs)
        return out, h, lora

    nonep_out, nonep_h, nonep_lora = _run(ep=False)
    ep_out, ep_h, ep_lora = _run(ep=True)
    fwd_l2 = _l2_rel(ep_out.detach(), nonep_out.detach())
    assert fwd_l2 <= _FWD_L2REL_TOL, f"[{variant}] NPU EP-vs-non-EP forward L2 rel {fwd_l2:.4%} > {_FWD_L2REL_TOL:.2%}"

    torch.manual_seed(456)
    grad_out = (torch.randn(B, H, dtype=dtype, device=dev) * 0.1).detach()
    nonep_grads = dict(
        zip(
            grad_keys,
            torch.autograd.grad(nonep_out, [nonep_h] + [nonep_lora[k] for k in _LORA_KEYS], grad_outputs=grad_out),
            strict=True,
        )
    )
    ep_grads = dict(
        zip(
            grad_keys,
            torch.autograd.grad(ep_out, [ep_h] + [ep_lora[k] for k in _LORA_KEYS], grad_outputs=grad_out),
            strict=True,
        )
    )
    for name in grad_keys:
        l2 = _l2_rel(ep_grads[name], nonep_grads[name])
        assert l2 <= _GRAD_L2REL_TOL, (
            f"[{variant}] {name}: NPU EP-vs-non-EP grad L2 rel {l2:.4%} > {_GRAD_L2REL_TOL:.2%}"
        )
