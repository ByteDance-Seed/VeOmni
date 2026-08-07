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
"""NPU fused MoE-LoRA forward/backward parity vs eager (shared + independent).

Counterpart of ``test_moe_lora_fused.py`` for Ascend. The NPU kernels compose
``npu_group_gemm`` + seed-style LoRA deltas under autograd (no hand-written
backward), so this suite is the correctness gate for that composition:

1. Wrapper fused_npu vs eager — forward outputs + ``h.grad`` + A/B grads.
2. EP vs non-EP single-rank — ``_npu_ep_fused_lora_moe_forward(ep_group=None)``
   must match ``_npu_fused_lora_moe_forward`` for outputs, ``h.grad``, and LoRA grads.

Skipped on non-NPU hosts. Wired into ``npu_unit_tests.yml``.

Run:
    pytest -v tests/lora/test_moe_lora_fused_npu.py
"""

from __future__ import annotations

import os
import warnings

import pytest
import torch
import torch.distributed as dist

from veomni.lora import resolve_fused_moe_lora_targets
from veomni.lora.moe_layers import (
    LoraIndependentExperts,
    LoraSharedExperts,
    apply_independent_moe_lora,
    apply_shared_moe_lora,
)
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type

from .utils import (
    build_toy,
    experts_module_globs,
    find_first_matching_module,
    fused_npu_moe_ops,
    load_lora_config,
)


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU fused MoE-LoRA parity requires torch_npu")

_TOY = "qwen3_moe_toy"
# Same L2-relative rationale as ``test_moe_lora_fused.py`` (bf16 reduction order).
_FWD_L2REL_TOL = 0.02
_GRAD_L2REL_TOL = 0.02

_MODES = {
    "shared": (LoraSharedExperts, apply_shared_moe_lora, "_fused_lora_moe_forward"),
    "independent": (LoraIndependentExperts, apply_independent_moe_lora, "_fused_independent_lora_moe_forward"),
}


def _l2_rel(actual: torch.Tensor, ref: torch.Tensor) -> float:
    a = actual.float()
    r = ref.float()
    ref_norm = r.norm().item()
    if ref_norm == 0.0:
        return (a - r).norm().item()
    return ((a - r).norm() / ref_norm).item()


@pytest.fixture(autouse=True)
def _restore_moe_pointers():
    """Save / restore fused MoE + LoRA pointers mutated by ``fused_npu`` builds."""
    from veomni.lora import ops as _lora_ops
    from veomni.ops.kernels import moe as _moe_ops

    saved_base = _moe_ops._fused_moe_forward
    saved_lora = _lora_ops._fused_lora_moe_forward
    saved_indep = _lora_ops._fused_independent_lora_moe_forward
    try:
        yield
    finally:
        _moe_ops._fused_moe_forward = saved_base
        _lora_ops._fused_lora_moe_forward = saved_lora
        _lora_ops._fused_independent_lora_moe_forward = saved_indep


def _wrap_with_lora(model, lora_cfg, *, apply_fn, lora_b_perturb_std: float = 0.0):
    apply_fn(
        model,
        target_parameter_patterns=lora_cfg["target_parameters"],
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        freeze_base_model=True,
    )
    if lora_b_perturb_std > 0:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if ".lora_B." in n:
                    p.add_(torch.randn_like(p) * lora_b_perturb_std)


def _make_inputs(experts_module, batch: int = 64, top_k: int = 2):
    H, E = experts_module.hidden_dim, experts_module.num_experts
    p0 = next(experts_module.parameters())
    dtype, dev = p0.dtype, p0.device
    h = torch.randn(batch, H, dtype=dtype, device=dev)
    top_k_index = torch.randint(0, E, (batch, top_k), device=dev)
    top_k_weights = torch.softmax(torch.randn(batch, top_k, dtype=torch.float32, device=dev), dim=-1).to(dtype)
    return h, top_k_index, top_k_weights


def _build_wrapped(*, mode: str, fused: bool, lora_b_perturb_std: float = 0.02):
    _, apply_fn, _ = _MODES[mode]
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_toy(_TOY, ops=fused_npu_moe_ops() if fused else None)
    lora_cfg = resolve_fused_moe_lora_targets(model, load_lora_config(_TOY))
    _wrap_with_lora(model, lora_cfg, apply_fn=apply_fn, lora_b_perturb_std=lora_b_perturb_std)
    sample_fqn, exp = find_first_matching_module(model, experts_module_globs(lora_cfg["target_parameters"]))
    return model, sample_fqn, exp, lora_cfg


def test_npu_fused_pointer_bound_after_fused_npu_build():
    """Building with ``moe_implementation=fused_npu`` binds both LoRA pointers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_toy(_TOY, ops=fused_npu_moe_ops())
    from veomni.lora import ops as _lora_ops
    from veomni.ops.kernels import moe as _moe_ops

    assert _lora_ops._fused_lora_moe_forward is not None
    assert _lora_ops._fused_independent_lora_moe_forward is not None
    assert _moe_ops._fused_moe_forward is not None


@pytest.mark.parametrize("mode", list(_MODES.keys()))
def test_npu_fused_vs_eager_forward_parity(mode):
    """Forward output of the NPU fused path matches the eager wrapper (L2-rel)."""
    model_e, fqn_e, exp_e, _ = _build_wrapped(mode=mode, fused=False, lora_b_perturb_std=0.02)
    h, idx, w = _make_inputs(exp_e)
    wrapper_e = model_e.get_submodule(fqn_e)
    with torch.no_grad():
        out_eager = wrapper_e(h, idx, w).clone()

    model_f, fqn_f, _, _ = _build_wrapped(mode=mode, fused=True, lora_b_perturb_std=0.02)
    wrapper_f = model_f.get_submodule(fqn_f)
    for pname in wrapper_e._lora_specs:
        assert torch.equal(wrapper_e.get_lora_A_weight(pname), wrapper_f.get_lora_A_weight(pname))
        assert torch.equal(wrapper_e.get_lora_B_weight(pname), wrapper_f.get_lora_B_weight(pname))
    with torch.no_grad():
        out_fused = wrapper_f(h, idx, w)

    l2 = _l2_rel(out_fused, out_eager)
    assert l2 <= _FWD_L2REL_TOL, (
        f"[{mode}] NPU forward parity broken: L2 relative error {l2:.4%} > {_FWD_L2REL_TOL:.2%} "
        f"(eager_norm={out_eager.float().norm().item():.3e})"
    )


@pytest.mark.parametrize("mode", list(_MODES.keys()))
def test_npu_fused_vs_eager_backward_parity(mode):
    """Gradients on hidden states + <spec>.lora_A / <spec>.lora_B match fused vs eager.

    Hidden-state grad is required: A/B-only checks would miss bugs in
    ``GmmFunction.backward``'s ``grad_input`` (or the all-to-all backward on the
    EP path) while every adapter assertion stayed green.
    """

    def _grads(*, fused: bool):
        model, fqn, exp, _ = _build_wrapped(mode=mode, fused=fused, lora_b_perturb_std=0.02)
        wrapper = model.get_submodule(fqn)
        wrapper.train()
        h, idx, w = _make_inputs(exp)
        h = h.detach().requires_grad_(True)
        loss = wrapper(h, idx, w).float().pow(2).sum()
        loss.backward()
        assert h.grad is not None, f"[{mode}] expected grad on hidden-state leaf"
        param_grads = {n: p.grad.detach().clone() for n, p in wrapper.named_parameters() if p.grad is not None}
        return param_grads, h.grad.detach().clone()

    grads_eager, hgrad_eager = _grads(fused=False)
    grads_fused, hgrad_fused = _grads(fused=True)

    assert set(grads_eager) == set(grads_fused), (
        f"[{mode}] different param sets received grad: only-eager={set(grads_eager) - set(grads_fused)}, "
        f"only-fused={set(grads_fused) - set(grads_eager)}"
    )
    h_l2 = _l2_rel(hgrad_fused, hgrad_eager)
    assert h_l2 <= _GRAD_L2REL_TOL, (
        f"[{mode}] hidden-state grad parity broken — L2 relative error {h_l2:.4%} > {_GRAD_L2REL_TOL:.2%} "
        f"(eager_norm={hgrad_eager.float().norm().item():.3e}, "
        f"max|fused-eager|={(hgrad_eager - hgrad_fused).abs().max().item():.3e})"
    )
    lora_param_names = sorted(n for n in grads_eager if "lora_A" in n.split(".") or "lora_B" in n.split("."))
    assert lora_param_names, f"[{mode}] expected LoRA A/B params to receive gradients"
    for n in lora_param_names:
        ge, gf = grads_eager[n], grads_fused[n]
        assert ge.shape == gf.shape, f"[{mode}] {n}: shape mismatch eager={ge.shape} fused={gf.shape}"
        l2 = _l2_rel(gf, ge)
        assert l2 <= _GRAD_L2REL_TOL, (
            f"[{mode}] {n}: NPU grad parity broken — L2 relative error {l2:.4%} > {_GRAD_L2REL_TOL:.2%} "
            f"(eager_norm={ge.float().norm().item():.3e}, max|fused-eager|={(ge - gf).abs().max().item():.3e})"
        )


def _make_lora_leaf(*shape: int, dtype: torch.dtype, device: torch.device, scale: float = 0.02) -> torch.Tensor:
    return (torch.randn(*shape, dtype=dtype, device=device) * scale).detach().requires_grad_(True)


def _build_lora_leaves(mode: str, *, E: int, H: int, I: int, r: int, dtype: torch.dtype, device: torch.device):
    if mode == "shared":
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


@pytest.fixture
def _single_rank_dist():
    """EP=1 identity path still calls ``all_to_all``; need a world_size=1 process group."""
    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29611")
        # gloo is fine for world_size=1 identity; avoids requiring HCCL for this unit check.
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created = True
    try:
        yield
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_npu_ep_vs_nonep_single_rank_parity(mode, _single_rank_dist):
    """EP path with ``ep_group=None`` (EP=1) matches non-EP on outputs + h/LoRA grads."""
    from veomni.lora.ops.npu_moe_group_gemm import (
        _npu_ep_fused_lora_moe_forward,
        _npu_fused_lora_moe_forward,
    )

    dev = torch.device(get_device_type())
    dtype = torch.bfloat16
    B, H, I, E, top_k, r = 32, 64, 96, 4, 2, 8
    scale_gate = scale_up = scale_down = 0.5
    _LORA_KEYS = ("lora_a_gate", "lora_b_gate", "lora_a_up", "lora_b_up", "lora_a_down", "lora_b_down")
    _GRAD_KEYS = ("hidden_states",) + _LORA_KEYS

    torch.manual_seed(0)
    selected_experts = torch.randint(0, E, (B, top_k), device=dev)
    routing_weights = torch.softmax(torch.randn(B, top_k, dtype=torch.float32, device=dev), dim=-1).to(dtype)
    gate_up_proj = (torch.randn(E, 2 * I, H, dtype=dtype, device=dev) * 0.05).detach()
    down_proj = (torch.randn(E, H, I, dtype=dtype, device=dev) * 0.05).detach()
    # Shared base activations so both branches start from the same h values;
    # each branch then clones its own leaf so autograd graphs stay isolated.
    torch.manual_seed(1)
    hidden_states_base = torch.randn(B, H, dtype=dtype, device=dev)

    def _run(*, ep: bool):
        torch.manual_seed(123)
        lora = _build_lora_leaves(mode, E=E, H=H, I=I, r=r, dtype=dtype, device=dev)
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
            lora_scale_gate=scale_gate,
            lora_scale_up=scale_up,
            lora_scale_down=scale_down,
            independent=(mode == "independent"),
        )
        if ep:
            out = _npu_ep_fused_lora_moe_forward(ep_group=None, **kwargs)
        else:
            out = _npu_fused_lora_moe_forward(**kwargs)
        return out, h, lora

    nonep_out, nonep_h, nonep_lora = _run(ep=False)
    ep_out, ep_h, ep_lora = _run(ep=True)

    fwd_l2 = _l2_rel(ep_out.detach(), nonep_out.detach())
    assert fwd_l2 <= _FWD_L2REL_TOL, (
        f"[{mode}] NPU EP-vs-non-EP forward parity broken: L2 rel {fwd_l2:.4%} > {_FWD_L2REL_TOL:.2%} "
        f"(ref_norm={nonep_out.float().norm().item():.3e})"
    )

    torch.manual_seed(456)
    grad_out = (torch.randn(B, H, dtype=dtype, device=dev) * 0.1).detach()
    nonep_grads = dict(
        zip(
            _GRAD_KEYS,
            torch.autograd.grad(
                nonep_out,
                [nonep_h] + [nonep_lora[k] for k in _LORA_KEYS],
                grad_outputs=grad_out,
            ),
        )
    )
    ep_grads = dict(
        zip(
            _GRAD_KEYS,
            torch.autograd.grad(
                ep_out,
                [ep_h] + [ep_lora[k] for k in _LORA_KEYS],
                grad_outputs=grad_out,
            ),
        )
    )
    for name in _GRAD_KEYS:
        g_nonep, g_ep = nonep_grads[name], ep_grads[name]
        assert g_nonep is not None and g_ep is not None, f"[{mode}] {name}: missing grad"
        assert g_nonep.shape == g_ep.shape, f"[{mode}] {name}: shape mismatch"
        l2 = _l2_rel(g_ep, g_nonep)
        assert l2 <= _GRAD_L2REL_TOL, (
            f"[{mode}] {name}: NPU EP-vs-non-EP backward parity broken — L2 rel {l2:.4%} > {_GRAD_L2REL_TOL:.2%} "
            f"(nonep_norm={g_nonep.float().norm().item():.3e}, max|Δ|={(g_nonep - g_ep).abs().max().item():.3e})"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
