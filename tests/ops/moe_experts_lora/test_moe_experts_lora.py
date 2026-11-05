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

"""Registry, hand-written oracle, and fused parity tests for ``moe_experts_lora``."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.ops import VeomniOp
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type
from veomni.utils.import_utils import is_fused_moe_available


_CI_ACCELERATOR_2 = LaunchSpec(nproc=2, device="accelerator")
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


_SHAPES = {
    "small": dict(B=32, H=64, I=96, E=4, top_k=2, r=8),
    # Explicit larger dimensions retain kernel-tile coverage independently of model fixtures.
    "qwen3_moe": dict(B=64, H=2048, I=768, E=16, top_k=2, r=16),
}


def _run_fused_vs_eager(
    impl: str,
    variant: str,
    *,
    shape: str = "small",
    base_trainable: bool = True,
    concentrated_routing: bool = False,
    routing_dtype: torch.dtype = torch.bfloat16,
):
    """Compare fused MoE-LoRA forward and gradients against the eager reference."""
    torch.manual_seed(0)
    device = torch.device("npu" if impl == "fused_npu" else "cuda")
    dtype = torch.bfloat16
    B, H, I, E, top_k, r = (_SHAPES[shape][key] for key in ("B", "H", "I", "E", "top_k", "r"))
    if concentrated_routing:
        B = 256
    hidden = 0.1 * torch.randn(B, H, device=device, dtype=dtype)
    routing = torch.softmax(torch.randn(B, top_k, device=device, dtype=torch.float32), dim=-1).to(routing_dtype)
    selected = (
        torch.zeros(B, top_k, device=device, dtype=torch.long)
        if concentrated_routing
        else torch.randint(0, E, (B, top_k), device=device)
    )
    fc1 = (0.05 * torch.randn(E, 2 * I, H, device=device, dtype=dtype)).detach()
    fc2 = (0.05 * torch.randn(E, H, I, device=device, dtype=dtype)).detach()
    loras = _lora_tensors(variant, E=E, H=H, I=I, r=r, device=device, dtype=dtype)
    hidden_e = hidden.detach().requires_grad_(True)
    hidden_f = hidden.detach().requires_grad_(True)
    routing_e = routing.detach().clone().requires_grad_(True)
    routing_f = routing.detach().clone().requires_grad_(True)
    fc1_e = fc1.detach().clone().requires_grad_(base_trainable)
    fc1_f = fc1.detach().clone().requires_grad_(base_trainable)
    fc2_e = fc2.detach().clone().requires_grad_(base_trainable)
    fc2_f = fc2.detach().clone().requires_grad_(base_trainable)
    lora_e = [t.detach().clone().requires_grad_(True) for t in loras]
    lora_f = [t.detach().clone().requires_grad_(True) for t in loras]
    case = f"[{impl}/{variant}/{shape}/base_trainable={base_trainable}]"

    out_e = _call("eager", variant, hidden_e, routing_e, selected, fc1_e, fc2_e, lora_e, num_experts=E)
    out_f = _call(impl, variant, hidden_f, routing_f, selected, fc1_f, fc2_f, lora_f, num_experts=E)
    fwd_l2 = _l2_rel(out_f, out_e)
    assert fwd_l2 <= _FWD_L2REL_TOL, (
        f"{case} forward L2 rel {fwd_l2:.4%} > {_FWD_L2REL_TOL:.2%} (eager_norm={out_e.float().norm().item():.3e})"
    )

    go = (0.1 * torch.randn_like(out_e)).detach()
    out_e.backward(go)
    out_f.backward(go)
    gradient_pairs = [
        ("hidden_states", hidden_e, hidden_f),
        ("routing_weights", routing_e, routing_f),
    ]
    if base_trainable:
        gradient_pairs += [("fc1_1_2_weight", fc1_e, fc1_f), ("fc2_weight", fc2_e, fc2_f)]
    else:
        for name, eager_input, fused_input in (("fc1_1_2_weight", fc1_e, fc1_f), ("fc2_weight", fc2_e, fc2_f)):
            assert eager_input.grad is None and fused_input.grad is None, f"{case} frozen {name} received a gradient"
    gradient_pairs.extend(zip(_LORA_KEYS, lora_e, lora_f, strict=True))
    for name, eager_input, fused_input in gradient_pairs:
        l2 = _l2_rel(fused_input.grad, eager_input.grad)
        assert l2 <= _GRAD_L2REL_TOL, (
            f"{case} {name} grad L2 rel {l2:.4%} > {_GRAD_L2REL_TOL:.2%} "
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
            intermediate = intermediate * routing[token_index, slot_index]
            expert_output = F.linear(intermediate, fc2[expert_index])
            expert_output = expert_output + (
                F.linear(F.linear(intermediate, a_down), b_down) * _SCALES["lora_scale_down"]
            )
            token_output = token_output + expert_output
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


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton moe_experts_lora needs a GPU + triton",
)
@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("base_trainable", [True, False], ids=["base_trainable", "base_frozen"])
@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_triton_matches_eager(variant, base_trainable, shape):
    _run_fused_vs_eager("fused_triton", variant, shape=shape, base_trainable=base_trainable)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton moe_experts_lora needs a GPU + triton",
)
@pytest.mark.parametrize("routing_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_triton_matches_eager_with_concentrated_routes(variant, routing_dtype):
    """Triton MoE-LoRA matches eager under concentrated duplicate routing and FP32/BF16 weights."""
    _run_fused_vs_eager("fused_triton", variant, concentrated_routing=True, routing_dtype=routing_dtype)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="npu moe_experts_lora needs torch_npu")
@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("base_trainable", [True, False], ids=["base_trainable", "base_frozen"])
@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_npu_matches_eager(variant, base_trainable, shape):
    _run_fused_vs_eager("fused_npu", variant, shape=shape, base_trainable=base_trainable)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton moe_experts_lora needs a GPU + triton",
)
@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_triton_ep_class_matches_nonep_single_rank(variant):
    """EP autograd class output and LoRA grads match the non-EP class on one rank."""
    from veomni.ops.kernels.moe_experts.shared.dispatch import expert_histogram, moe_gather, moe_scatter
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
    grad_keys = ("hidden_states", "routing_weights", "fc1_1_2_weight", "fc2_weight", *_LORA_KEYS)

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
        routing = routing_weights.detach().clone().requires_grad_(True)
        fc1 = gate_up_proj.detach().clone().requires_grad_(True)
        fc2 = down_proj.detach().clone().requires_grad_(True)
        kwargs = dict(
            num_experts=E,
            routing_weights=routing,
            selected_experts=selected_experts,
            hidden_states=h,
            fc1_1_2_weight=fc1,
            fc2_weight=fc2,
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
        return out, (h, routing, fc1, fc2, *(lora[key] for key in _LORA_KEYS))

    nonep_out, nonep_inputs = _run(ep=False)
    ep_out, ep_inputs = _run(ep=True)
    fwd_l2 = _l2_rel(ep_out.detach(), nonep_out.detach())
    assert fwd_l2 <= _FWD_L2REL_TOL, f"[{variant}] NPU EP-vs-non-EP forward L2 rel {fwd_l2:.4%} > {_FWD_L2REL_TOL:.2%}"

    torch.manual_seed(456)
    grad_out = (torch.randn(B, H, dtype=dtype, device=dev) * 0.1).detach()
    nonep_grads = dict(
        zip(
            grad_keys,
            torch.autograd.grad(nonep_out, nonep_inputs, grad_outputs=grad_out),
            strict=True,
        )
    )
    ep_grads = dict(
        zip(
            grad_keys,
            torch.autograd.grad(ep_out, ep_inputs, grad_outputs=grad_out),
            strict=True,
        )
    )
    for name in grad_keys:
        l2 = _l2_rel(ep_grads[name], nonep_grads[name])
        assert l2 <= _GRAD_L2REL_TOL, (
            f"[{variant}] {name}: NPU EP-vs-non-EP grad L2 rel {l2:.4%} > {_GRAD_L2REL_TOL:.2%}"
        )


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton moe_experts_lora needs a GPU + triton",
)
@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_zero_b_preserves_fused_base_output_and_vjp(variant):
    """Zero-initialized LoRA B matrices must preserve the base fused MoE path."""
    from veomni.ops.kernels.moe_experts.standard.triton import MergedFc1TritonFusedMoeExpertFunction
    from veomni.ops.kernels.moe_experts_lora.independent.triton import (
        MergedFc1IndependentTritonFusedLoRAMoeExpertFunction,
    )
    from veomni.ops.kernels.moe_experts_lora.shared.triton import (
        MergedFc1TritonFusedLoRAMoeExpertFunction,
    )

    nonep_cls = (
        MergedFc1TritonFusedLoRAMoeExpertFunction
        if variant == "shared"
        else MergedFc1IndependentTritonFusedLoRAMoeExpertFunction
    )

    dev = torch.device(get_device_type())
    dtype = torch.bfloat16
    B, H, I, E, top_k, r = 32, 64, 96, 4, 2, 8
    scale_gate, scale_up, scale_down = 0.5, 0.5, 0.5

    torch.manual_seed(0)
    hidden_states = torch.randn(B, H, dtype=dtype, device=dev)
    top_k_index = torch.randint(0, E, (B, top_k), device=dev)
    top_k_weights = torch.softmax(torch.randn(B, top_k, dtype=torch.float32, device=dev), dim=-1).to(dtype)
    gate_up_proj = (torch.randn(E, 2 * I, H, dtype=dtype, device=dev) * 0.05).detach()
    down_proj = (torch.randn(E, H, I, dtype=dtype, device=dev) * 0.05).detach()
    grad_out = torch.randn(B, H, dtype=dtype, device=dev)

    hs_base = hidden_states.clone().detach().requires_grad_(True)
    rw_base = top_k_weights.clone().detach().requires_grad_(True)
    fc1_base = gate_up_proj.clone().detach().requires_grad_(True)
    fc2_base = down_proj.clone().detach().requires_grad_(True)
    out_base = MergedFc1TritonFusedMoeExpertFunction.apply(E, rw_base, top_k_index, hs_base, fc1_base, fc2_base, None)
    out_base.backward(grad_out)

    torch.manual_seed(123)
    lora = _build_lora_leaves(variant, E=E, H=H, I=I, r=r, dtype=dtype, device=dev)
    for key in ("lora_b_gate", "lora_b_up", "lora_b_down"):
        lora[key] = torch.zeros_like(lora[key], requires_grad=True)

    hs_lora = hidden_states.clone().detach().requires_grad_(True)
    rw_lora = top_k_weights.clone().detach().requires_grad_(True)
    fc1_lora = gate_up_proj.clone().detach().requires_grad_(True)
    fc2_lora = down_proj.clone().detach().requires_grad_(True)
    out_lora = nonep_cls.apply(
        E,
        rw_lora,
        top_k_index,
        hs_lora,
        fc1_lora,
        fc2_lora,
        *(lora[k] for k in _LORA_KEYS),
        scale_gate,
        scale_up,
        scale_down,
    )
    out_lora.backward(grad_out)

    torch.testing.assert_close(out_lora, out_base, rtol=0, atol=0)
    torch.testing.assert_close(hs_lora.grad, hs_base.grad, rtol=0, atol=0)
    torch.testing.assert_close(rw_lora.grad, rw_base.grad, rtol=0, atol=0)
    torch.testing.assert_close(fc1_lora.grad, fc1_base.grad, rtol=0, atol=0)
    torch.testing.assert_close(fc2_lora.grad, fc2_base.grad, rtol=0, atol=0)
    for key in ("lora_a_gate", "lora_a_up", "lora_a_down"):
        assert torch.count_nonzero(lora[key].grad) == 0
    for key in ("lora_b_gate", "lora_b_up", "lora_b_down"):
        assert torch.isfinite(lora[key].grad).all() and torch.count_nonzero(lora[key].grad) > 0


def _distributed_ep_parity_worker(rank: int, variant: str, endpoint) -> None:
    """Compare the full EP dispatch against non-EP with identical inputs and upstream gradients."""
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", store=endpoint.connect(), rank=rank, world_size=_CI_ACCELERATOR_2.nproc)
    try:
        device = torch.device("cuda", rank)
        dtype = torch.bfloat16
        experts, hidden_size, intermediate_size, lora_rank, batch = 4, 64, 96, 8, 16
        generator = torch.Generator().manual_seed(7123)

        def random_tensor(shape: tuple[int, ...], scale: float) -> torch.Tensor:
            """Draw a deterministic scaled random tensor on the worker device."""
            return (torch.randn(shape, generator=generator) * scale).to(device=device, dtype=dtype)

        hidden = random_tensor((2, batch, hidden_size), 0.5)[rank]
        upstream = random_tensor((2, batch, hidden_size), 0.1)[rank]
        gate_up = random_tensor((experts, 2 * intermediate_size, hidden_size), 0.05)
        down = random_tensor((experts, hidden_size, intermediate_size), 0.05)
        ids = torch.arange(batch, device=device)
        selected = torch.stack(((ids + rank) % experts, (ids + rank + 1) % experts), dim=1)
        routing = torch.tensor([0.75, 0.25], dtype=torch.float32, device=device).expand(batch, 2).contiguous()
        if variant == "shared":
            shapes = [
                (lora_rank, hidden_size),
                (intermediate_size, lora_rank),
                (lora_rank, hidden_size),
                (intermediate_size, lora_rank),
                (lora_rank, intermediate_size),
                (hidden_size, lora_rank),
            ]
        else:
            shapes = [
                (experts, lora_rank, hidden_size),
                (experts, intermediate_size, lora_rank),
                (experts, lora_rank, hidden_size),
                (experts, intermediate_size, lora_rank),
                (experts, lora_rank, intermediate_size),
                (experts, hidden_size, lora_rank),
            ]
        full_lora = [random_tensor(shape, 0.02).detach().requires_grad_(True) for shape in shapes]
        hidden_reference = hidden.detach().requires_grad_(True)
        routing_reference = routing.detach().clone().requires_grad_(True)
        non_ep_state = _init_parallel_state(dp_size=2, device_type="cuda", name="non_ep")
        ep_state = _init_parallel_state(dp_size=2, device_type="cuda", extra_parallel_sizes=(2,), name="ep")

        with use_parallel_state(non_ep_state):
            reference = _call(
                "fused_triton",
                variant,
                hidden_reference,
                routing_reference,
                selected,
                gate_up,
                down,
                full_lora,
                num_experts=experts,
            )
            reference_grads = torch.autograd.grad(
                reference, [*full_lora, hidden_reference, routing_reference], grad_outputs=upstream
            )

        local_lora = []
        for value in full_lora:
            if variant == "independent":
                value = value[rank * 2 : (rank + 1) * 2]
            local_lora.append(value.detach().clone().requires_grad_(True))
        hidden_ep = hidden.detach().requires_grad_(True)
        routing_ep = routing.detach().clone().requires_grad_(True)
        with use_parallel_state(ep_state):
            actual = _call(
                "fused_triton",
                variant,
                hidden_ep,
                routing_ep,
                selected,
                gate_up[rank * 2 : (rank + 1) * 2].contiguous(),
                down[rank * 2 : (rank + 1) * 2].contiguous(),
                local_lora,
                num_experts=experts,
            )
            actual_grads = torch.autograd.grad(actual, [*local_lora, hidden_ep, routing_ep], grad_outputs=upstream)

        # Non-EP computes a full gradient from each rank's tokens. EP computes
        # one local expert shard from both ranks' tokens, or a shared-LoRA
        # partial from its local experts. Compare the logical full gradients.
        gradient_checks = []
        for index, (local, full) in enumerate(zip(actual_grads[:-2], reference_grads[:-2], strict=True)):
            expected = full.float().clone()
            dist.all_reduce(expected, op=dist.ReduceOp.SUM)
            if variant == "independent":
                parts = [torch.empty_like(local) for _ in range(2)]
                dist.all_gather(parts, local.contiguous())
                observed = torch.cat(parts, dim=0).float()
            else:
                observed = local.float().clone()
                dist.all_reduce(observed, op=dist.ReduceOp.SUM)
            gradient_checks.append((index, observed, expected))

        assert actual.dtype == dtype, f"{variant}: distributed EP output dtype {actual.dtype} != {dtype}"
        assert _l2_rel(actual, reference) <= _FWD_L2REL_TOL, f"{variant}: distributed EP forward differs from non-EP"
        assert _l2_rel(actual_grads[-2], reference_grads[-2]) <= _GRAD_L2REL_TOL, (
            f"{variant}: distributed EP input gradient differs from non-EP"
        )
        assert actual_grads[-1].dtype == torch.float32, (
            f"{variant}: distributed EP routing gradient dtype {actual_grads[-1].dtype} != float32"
        )
        assert _l2_rel(actual_grads[-1], reference_grads[-1]) <= _GRAD_L2REL_TOL, (
            f"{variant}: distributed EP routing gradient differs from non-EP"
        )
        for index, observed, expected in gradient_checks:
            assert expected.count_nonzero().item() > 0, f"{variant}: LoRA gradient {index} was zero"
            relative_error = _l2_rel(observed, expected)
            assert relative_error <= _GRAD_L2REL_TOL, (
                f"{variant}: distributed EP LoRA gradient {index} differs from non-EP "
                f"(relative L2 error {relative_error:.4%})"
            )
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="distributed triton moe_experts_lora needs GPUs + triton",
)
@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_distributed_ep_dispatch_matches_nonep_forward_and_gradients(variant):
    """Two GPUs exercise real all-to-all dispatch and both LoRA gradient layouts."""
    if runner_device_count() < _CI_ACCELERATOR_2.nproc:
        pytest.skip("Distributed EP parity requires two CUDA devices.")
    spawn_with_rendezvous(
        _distributed_ep_parity_worker,
        args=(variant,),
        nprocs=_CI_ACCELERATOR_2.nproc,
    )
