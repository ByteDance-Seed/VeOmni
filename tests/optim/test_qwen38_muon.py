# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Logical-slice, precision, and cold-resume tests independent of model kernels."""

import copy
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.optim import qwen38_muon as module


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
_CI_SHARD_LAUNCH = LaunchSpec(nproc=2, device="cpu")


def make_optimizer(p, plan, **kwargs):
    return module.Qwen38MuonAdamW(
        [
            {
                "params": [p],
                "cpu_offload": True,
                "qwen38_plan": json.dumps(plan),
                "qwen38_recipe": module.RECIPE,
            }
        ],
        lr=1e-4,
        weight_decay=0.1,
        **kwargs,
    )


@pytest.mark.parametrize("kind", ["muon", "adamw"])
def test_qwen_step_stays_fp32_with_bf16_default_dtype(kind):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        param = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
        optimizer = make_optimizer(param, [(kind, 0, 2)])
        param.grad = torch.ones_like(param)
        for _ in range(257):
            optimizer.step()
        assert optimizer.state[param]["step"].dtype == torch.float32
        assert optimizer.state[param]["step"].device.type == "cpu"
        assert optimizer.state[param]["step"].item() == 257
    finally:
        torch.set_default_dtype(previous)


def test_qwen_multioptimizer_raw_snapshot_is_independent():
    from veomni.optim.optimizer import MultiOptimizer

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2, 2))
    leaf = module.Qwen38MuonAdamW(
        [{"params": [model.weight], "qwen38_recipe": module.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}]
    )
    optimizer = MultiOptimizer(model, {"muon": leaf}, ["muon"])
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    snapshot = optimizer.state_dict()
    master = leaf.state[model.weight]["master_weight"]
    before = master.clone()
    snapshot["state.weight.master_weight"].add_(1)
    torch.testing.assert_close(master, before, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="Unverified MultiOptimizer state"):
        optimizer.load_state_dict(snapshot)


def reference_polar(matrix):
    # FP64 reference with the unexpanded X, X^3, X^5 expression.
    x = matrix.double()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (torch.linalg.vector_norm(x) * 1.01 + 1e-14)
    for a, b, c in module.POLAR_EXPRESS_8:
        gram = x @ x.T
        x = a * x + b * (gram @ x) + c * (gram @ gram @ x)
    return x.T if transpose else x


@pytest.mark.parametrize("shape", [(5, 17), (17, 5), (8, 8)])
def test_polar_against_high_precision_reference(shape):
    torch.manual_seed(21)
    matrix = torch.randn(shape)
    result = module.polar_express_fp32(matrix)
    torch.testing.assert_close(result.double(), reference_polar(matrix), atol=2e-5, rtol=2e-5)
    assert torch.equal(module.polar_express_fp32(torch.zeros(shape)), torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_interleaved_q_gate_fp32_master_and_resume(dtype):
    torch.manual_seed(10)
    p = torch.nn.Parameter(torch.randn(12, 9).to(dtype))
    plan = [("muon" if i % 2 == 0 else "adamw", 3 * i, 3 * (i + 1)) for i in range(4)]
    opt = make_optimizer(p, plan)
    master = p.detach().double().clone()
    first, second = torch.zeros_like(master), torch.zeros_like(master)
    for step in range(1, 11):
        p.grad = torch.randn_like(p) * 0.01
        grad = p.grad.double()
        update = torch.empty_like(master)
        for kind, start, stop in plan:
            g, m, v = grad[start:stop], first[start:stop], second[start:stop]
            if kind == "muon":
                m.mul_(0.95).add_(g, alpha=0.05)
                update[start:stop] = reference_polar(0.05 * g + 0.95 * m) * (0.2 * max(g.shape) ** 0.5)
            else:
                m.mul_(0.9).add_(g, alpha=0.1)
                v.mul_(0.95).addcmul_(g, g, value=0.05)
                update[start:stop] = m / (1 - 0.9**step) / (v.sqrt() / (1 - 0.95**step) ** 0.5 + 1e-8)
        master.mul_(1 - 1e-5).add_(update, alpha=-1e-4)
        opt.step()
        actual = opt.state[p]["master_weight"]
        torch.testing.assert_close(actual.double(), master, atol=3e-6, rtol=1e-6)
        assert torch.equal(p, actual.to(dtype))
        assert all(opt.state[p][key].dtype == torch.float32 for key in ("master_weight", "exp_avg", "exp_avg_sq"))
        if step == 5:
            resumed = make_optimizer(p, plan)
            resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
            for key in ("master_weight", "exp_avg", "exp_avg_sq"):
                assert torch.equal(resumed.state[p][key], opt.state[p][key])
            opt = resumed


def test_adam_fallback_matches_existing_fp32_optimizer():
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(11, 9).bfloat16())
    r = torch.nn.Parameter(p.detach().float())
    opt = make_optimizer(p, [("adamw", 0, 11)])
    reference = torch.optim.AdamW([r], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, foreach=False)
    for _ in range(8):
        p.grad = torch.randn_like(p)
        r.grad = p.grad.float()
        opt.step()
        reference.step()
        torch.testing.assert_close(opt.state[p]["master_weight"], r, atol=1e-7, rtol=1e-6)


def test_resume_hooks_preserve_moments_and_next_mixed_update():
    param = torch.nn.Parameter(torch.arange(12).reshape(4, 3).bfloat16() / 16)
    plan = [("muon", 0, 2), ("adamw", 2, 4)]
    source = make_optimizer(param, plan)
    param.grad = torch.arange(12).reshape(4, 3).bfloat16() / 32
    source.step()
    saved = copy.deepcopy(source.state_dict())
    saved["state"]["global_metadata"] = {"version": 7}
    key = saved["param_groups"][0]["params"][0]
    target = torch.nn.Parameter(param.detach().clone())
    resumed = make_optimizer(target, plan)
    events = []

    def before_load(optimizer, incoming):
        events.append("before")
        assert optimizer is resumed
        assert incoming["state"][key]["exp_avg"].dtype == torch.float32
        replacement = copy.deepcopy(incoming)
        replacement["state"][key]["exp_avg"].add_(0.125)
        return replacement

    def after_load(optimizer):
        events.append("after")
        assert optimizer.state["global_metadata"] == {"version": 7}
        state = optimizer.state[target]
        assert set(state) == {"step", "master_weight", "exp_avg", "exp_avg_sq"}
        state["exp_avg"].add_(0.25)

    resumed.register_load_state_dict_pre_hook(before_load)
    resumed.register_load_state_dict_post_hook(after_load)
    resumed.load_state_dict(saved)
    assert events == ["before", "after"]

    expected = copy.deepcopy(saved)
    expected["state"][key]["exp_avg"].add_(0.125).add_(0.25)
    reference_param = torch.nn.Parameter(param.detach().clone())
    reference = make_optimizer(reference_param, plan)
    reference.load_state_dict(expected)
    for parameter, optimizer in ((target, resumed), (reference_param, reference)):
        parameter.grad = torch.arange(12).reshape(4, 3).flip(0).bfloat16() / 32
        optimizer.step()
    torch.testing.assert_close(target, reference_param, rtol=0, atol=0)
    for name in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(resumed.state[target][name], reference.state[reference_param][name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_trainable_ple_lookup_matches_fp32_adam_and_resumes(dtype):
    torch.manual_seed(14)
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.ModuleDict({"shard_0": torch.nn.Embedding(12, 8, dtype=dtype)})
    p = model.ngram_embedding["shard_0"].weight
    reference_p = torch.nn.Parameter(p.detach().float().clone())
    initial = p.detach().clone()

    def build(param):
        groups = module.add_qwen38_plans(model, [{"params": [param], "weight_decay": 0.0, "cpu_offload": True}])
        assert json.loads(groups[0]["qwen38_plan"]) == [["adamw", 0, 12]]
        return module.Qwen38MuonAdamW(groups, lr=1e-4)

    opt = build(p)
    reference = torch.optim.Adam([reference_p], lr=1e-4, betas=(0.9, 0.95), foreach=False)
    for step in range(6):
        opt.zero_grad()
        model.ngram_embedding["shard_0"](torch.tensor([0, 2, 2, 5])).square().mean().backward()
        reference_p.grad = p.grad.float().clone()
        opt.step()
        reference.step()
        for key in ("master_weight", "exp_avg", "exp_avg_sq"):
            assert opt.state[p][key].dtype == torch.float32
            assert opt.state[p][key].device.type == "cpu"
        torch.testing.assert_close(opt.state[p]["master_weight"], reference_p, atol=1e-7, rtol=1e-6)
        assert torch.equal(p, opt.state[p]["master_weight"].to(dtype))
        assert torch.equal(p.detach()[6:], initial[6:])
        if step == 2:
            saved = copy.deepcopy(opt.state_dict())
            fresh_p = torch.nn.Parameter(p.detach().clone())
            model.ngram_embedding["shard_0"].weight = fresh_p
            fresh = build(fresh_p)
            fresh.load_state_dict(saved)
            p, opt = fresh_p, fresh

    for group in ({"params": [p], "weight_decay": 0.1}, {"params": [p]}):
        with pytest.raises(ValueError, match="zero weight decay"):
            module.add_qwen38_plans(model, [group])
    groups = module.add_qwen38_plans(model, [{"params": [p], "weight_decay": 0.0}])
    groups[0]["weight_decay"] = 0.1
    with pytest.raises(ValueError, match="final optimizer groups"):
        module.Qwen38MuonAdamW(groups)
    saved = copy.deepcopy(opt.state_dict())
    saved["param_groups"][0]["weight_decay"] = 0.1
    with pytest.raises(RuntimeError, match="PLE lookup marker and zero weight decay"):
        opt.load_state_dict(saved)


def test_resume_rejects_old_adam_and_changed_splits():
    p = torch.nn.Parameter(torch.ones(8, 5, dtype=torch.bfloat16))
    opt = make_optimizer(p, [("muon", 0, 8)])
    p.grad = torch.ones_like(p)
    opt.step()
    state = opt.state_dict()
    del state["param_groups"][0]["qwen38_recipe"]
    with pytest.raises(RuntimeError, match="recipe mismatch"):
        opt.load_state_dict(state)
    with pytest.raises(ValueError, match="FP32"):
        make_optimizer(p, [("muon", 0, 8)], cpu_moment_dtype="bfloat16")


def test_logical_model_layout_and_frozen_rules():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.head_dim = 3
            self.attn.q_proj = torch.nn.Linear(9, 12, bias=False)
            self.attn.k_proj = torch.nn.Linear(9, 6, bias=False)
            self.gdn = torch.nn.Module()
            self.gdn.head_k_dim, self.gdn.num_k_heads = 2, 2
            self.gdn.head_v_dim, self.gdn.num_v_heads = 3, 4
            self.gdn.in_proj_qkv = torch.nn.Linear(9, 20, bias=False)
            self.gdn.in_proj_z = torch.nn.Linear(9, 12, bias=False)
            self.experts = torch.nn.Module()
            self.experts.gate_up_proj = torch.nn.Parameter(torch.ones(4, 8, 9))
            self.router = torch.nn.Linear(9, 4, bias=False)
            self.unknown = torch.nn.Linear(9, 4, bias=False)

    model = Model()
    plans = {n: module.logical_plan(model, n, p) for n, p in model.named_parameters() if "unknown" not in n}
    assert plans["attn.q_proj.weight"] == [("muon", 0, 3), ("adamw", 3, 6), ("muon", 6, 9), ("adamw", 9, 12)]
    assert plans["gdn.in_proj_qkv.weight"] == [("muon", i, i + 2) for i in range(0, 8, 2)] + [
        ("muon", i, i + 3) for i in range(8, 20, 3)
    ]
    assert plans["experts.gate_up_proj"] == [("muon", 0, 4), ("muon", 4, 8)]
    assert plans["gdn.in_proj_z.weight"] == [("adamw", 0, 12)]
    with pytest.raises(ValueError, match="Unclassified"):
        module.logical_plan(model, "unknown.weight", model.unknown.weight)


def test_workspace_limits_fail_before_allocation():
    p = torch.nn.Parameter(torch.zeros(8, 9))
    with pytest.raises(ValueError, match="workspace"):
        make_optimizer(p, [("muon", 0, 8)], max_matrix_parameter_numel=10)


def _distributed_worker(rank, world, rendezvous, checkpoint_dir):
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    try:
        aligned_2d = [("muon", 0, 3), ("adamw", 3, 6), ("muon", 6, 9), ("adamw", 9, 12)]
        crossing_2d = [("adamw", 0, 4), ("muon", 4, 8), ("adamw", 8, 12)]
        aligned_3d = [("muon", 0, 4), ("muon", 4, 8)]
        crossing_3d = [("muon", 0, 3), ("muon", 3, 8)]
        # Keep aligned blocks, column shards, and whole-expert ownership covered.
        # The added row-axis cases cross a logical Muon matrix: row 6 cuts
        # [4, 8) in 2D; row 4 cuts [3, 8) independently within each expert.
        cases = [
            ((12, 8), 0, [("adamw", 0, 12)]),
            ((12, 8), 1, [("adamw", 0, 12)]),
            ((12, 9), 0, aligned_2d),
            ((12, 9), 1, aligned_2d),
            ((4, 8, 9), 0, aligned_3d),
            ((4, 8, 9), 1, aligned_3d),
            ((12, 9), 0, crossing_2d),
            ((4, 8, 9), 1, crossing_3d),
        ]
        for case_id, (shape, axis, plan) in enumerate(cases):
            torch.manual_seed(123)
            full = torch.randn(shape).bfloat16()
            model = torch.nn.Module()
            model.register_parameter("weight", torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(axis)])))
            p = model.weight
            reference_p = torch.nn.Parameter(full.clone())
            opt, reference = make_optimizer(p, plan), make_optimizer(reference_p, plan)
            for step in range(3):
                grad = torch.randn_like(full) * 0.01
                p.grad = distribute_tensor(grad, mesh, [Shard(axis)])
                reference_p.grad = grad
                opt.step()
                reference.step()
                for key in ("master_weight", "exp_avg", "exp_avg_sq"):
                    wanted = distribute_tensor(reference.state[reference_p][key], mesh, [Shard(axis)]).to_local()
                    torch.testing.assert_close(opt.state[p][key].to_local(), wanted, atol=1e-7, rtol=1e-6)
                if step == 1:
                    path = str(Path(checkpoint_dir) / f"case{case_id}-ndim{len(shape)}-axis{axis}")
                    msd, osd = get_state_dict(model, opt)
                    dcp.save({"model": msd, "optimizer": osd}, checkpoint_id=path)
                    fresh_p = torch.nn.Parameter(distribute_tensor(torch.zeros_like(full), mesh, [Shard(axis)]))
                    model.weight = fresh_p
                    fresh = make_optimizer(fresh_p, plan)
                    fresh_p.grad = torch.zeros_like(fresh_p)
                    fresh.step()
                    fresh.zero_grad()
                    msd, osd = get_state_dict(model, fresh)
                    dcp.load({"model": msd, "optimizer": osd}, checkpoint_id=path)
                    set_state_dict(model, fresh, model_state_dict=msd, optim_state_dict=osd)
                    p, opt = fresh_p, fresh
                    assert opt.state[p]["master_weight"].dtype == torch.float32
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_SHARD_LAUNCH)
def test_distributed_updates_and_dcp_resume(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _distributed_worker,
        args=(_CI_SHARD_LAUNCH.nproc, str(tmp_path / "rdzv"), str(tmp_path / "dcp")),
        nprocs=_CI_SHARD_LAUNCH.nproc,
        join=True,
        start_method="fork",
    )


def test_frozen_policy_covers_vectors_and_embeddings():
    model = torch.nn.Module()
    model.indexer = torch.nn.LayerNorm(4)
    model.visual = torch.nn.Module()
    model.visual.patch = torch.nn.Conv1d(4, 4, 1)
    model.visual.embedding = torch.nn.Embedding(4, 4)
    for name, p in model.named_parameters():
        with pytest.raises(ValueError, match="remain frozen"):
            module.logical_plan(model, "model." + name, p)


def test_step_keeps_fp32_muon_states_without_diagnostic_output(capsys):
    p = torch.nn.Parameter(torch.ones(8, 5))
    opt = make_optimizer(p, [("muon", 0, 8)])
    p.grad = torch.ones_like(p)
    opt.step()
    assert not capsys.readouterr().out
    state = opt.state[p]
    assert all(state[key].dtype == torch.float32 for key in ("master_weight", "exp_avg", "exp_avg_sq"))
    assert sum(
        state[key].numel() * state[key].element_size() for key in ("master_weight", "exp_avg", "exp_avg_sq")
    ) == (8 * 5 * 4 * 3)


@pytest.mark.parametrize("parent", ["none", "ieee"])
def test_strict_cuda_precision_scope_preserves_inheritance(parent):
    code = """
import torch
from veomni.optim.qwen38_muon import _strict_matmul
torch.backends.fp32_precision = PARENT
before = torch.backends.cuda.matmul.fp32_precision
with _strict_matmul('cuda'):
    assert torch.backends.cuda.matmul.fp32_precision == before
torch.backends.fp32_precision = 'tf32'
assert torch.backends.cuda.matmul.fp32_precision == 'tf32'
""".replace("PARENT", repr(parent))
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_strict_cuda_precision_scope_rejects_legacy_tf32_without_mutation(monkeypatch):
    backend = types.SimpleNamespace(allow_tf32=True)
    monkeypatch.setattr(torch.backends.cuda, "matmul", backend)
    with pytest.raises(RuntimeError, match="IEEE FP32"):
        with module._strict_matmul("cuda"):
            pass
    assert backend.allow_tf32 is True


@pytest.mark.parametrize(
    "setup",
    [
        "torch.backends.fp32_precision = 'tf32'",
        "torch.backends.cuda.matmul.fp32_precision = 'tf32'",
        "torch.set_float32_matmul_precision('high')",
        "torch.set_float32_matmul_precision('medium')",
        "torch.set_float32_matmul_precision('medium'); torch.backends.cuda.matmul.fp32_precision = 'none'",
    ],
)
def test_cuda_precision_preflight_rejects_unsafe_policy_without_mutating_it(setup):
    code = """
import torch
from veomni.optim.qwen38_muon import _check_cuda_fp32_policy
SETUP
before = torch.backends.cuda.matmul.fp32_precision
try:
    _check_cuda_fp32_policy()
except RuntimeError:
    pass
else:
    raise AssertionError('unsupported policy was accepted')
assert torch.backends.cuda.matmul.fp32_precision == before
""".replace("SETUP", setup)
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_cuda_precision_preflight_rejects_cublas_override(monkeypatch):
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    with pytest.raises(RuntimeError, match="forced cuBLAS TF32"):
        module._check_cuda_fp32_policy()


def test_cuda_precision_preflight_fails_before_adam_or_muon_updates(monkeypatch):
    from veomni.optim import optimizer as factory
    from veomni.optim import qwen38_muon as implementation

    model = torch.nn.Module()
    model.first = torch.nn.Parameter(torch.tensor([1.0]))
    model.second = torch.nn.Parameter(torch.ones(2, 2))
    model.first.grad = torch.ones_like(model.first)
    model.second.grad = torch.ones_like(model.second)
    adam = torch.optim.AdamW([model.first], lr=0.01)
    muon = implementation.Qwen38MuonAdamW(
        [{"params": [model.second], "qwen38_recipe": implementation.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}]
    )
    combined = factory.MultiOptimizer(model, {"adam": adam, "muon": muon}, ["adam", "muon"])
    monkeypatch.setattr(factory, "_optimizer_uses_cuda", lambda _: True)

    def reject():
        raise RuntimeError("unsafe test precision")

    monkeypatch.setattr(implementation, "_check_cuda_fp32_policy", reject)
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(RuntimeError, match="unsafe test precision"):
        combined.step()
    assert all(torch.equal(p, saved) for p, saved in zip(model.parameters(), before, strict=True))
    assert not adam.state and not muon.state


def test_cuda_precision_preflight_rejects_child_step_hook_before_any_update(monkeypatch):
    from veomni.optim import optimizer as factory
    from veomni.optim import qwen38_muon as implementation

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([1.0]))
    model.weight.grad = torch.ones_like(model.weight)
    adam = torch.optim.AdamW([model.weight], lr=0.01)
    calls = []
    adam.register_step_pre_hook(lambda *_: calls.append("executed"))
    combined = factory.MultiOptimizer(model, {"adam": adam}, ["adam"], qwen38_cuda_guard=True)
    monkeypatch.setattr(factory, "_optimizer_uses_cuda", lambda _: True)
    monkeypatch.setattr(implementation, "_check_cuda_fp32_policy", lambda: None)
    with pytest.raises(RuntimeError, match="step hooks"):
        combined.step()
    assert not calls and not adam.state
    assert torch.equal(model.weight, torch.tensor([1.0]))


def test_direct_cuda_preflight_runs_before_pytorch_step_hooks(monkeypatch):
    param = torch.nn.Parameter(torch.ones(2, 2))
    param.grad = torch.ones_like(param)
    opt = make_optimizer(param, [("muon", 0, 2)])
    calls = []

    def mutate(*_):
        calls.append(True)
        with torch.no_grad():
            param.add_(10)

    opt.register_step_pre_hook(mutate)
    monkeypatch.setattr(module, "_optimizer_uses_cuda", lambda _: True)
    monkeypatch.setattr(module, "_check_cuda_fp32_policy", lambda: None)
    with pytest.raises(RuntimeError, match="step hooks"):
        opt.step()
    assert not calls and not opt.state
    assert torch.equal(param, torch.ones_like(param))


def test_direct_cuda_preflight_runs_before_global_step_hook(monkeypatch):
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    param = torch.nn.Parameter(torch.ones(2, 2))
    param.grad = torch.ones_like(param)
    opt = make_optimizer(param, [("muon", 0, 2)])
    calls = []
    handle = register_optimizer_step_pre_hook(lambda *_: calls.append(True))
    try:
        monkeypatch.setattr(module, "_optimizer_uses_cuda", lambda _: True)
        monkeypatch.setattr(module, "_check_cuda_fp32_policy", lambda: None)
        with pytest.raises(RuntimeError, match="global optimizer step hooks"):
            opt.step()
    finally:
        handle.remove()
    assert not calls and not opt.state
    assert torch.equal(param, torch.ones_like(param))


def test_cuda_preflight_rejects_custom_child_before_its_step(monkeypatch):
    from veomni.optim import optimizer as factory
    from veomni.optim import qwen38_muon as implementation

    class CustomAdam(torch.optim.AdamW):
        def step(self, closure=None):
            raise AssertionError("custom step must not execute")

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(1))
    model.weight.grad = torch.ones_like(model.weight)
    adam = CustomAdam([model.weight])
    combined = factory.MultiOptimizer(model, {"adam": adam}, ["adam"], qwen38_cuda_guard=True)
    monkeypatch.setattr(factory, "_optimizer_uses_cuda", lambda _: True)
    monkeypatch.setattr(implementation, "_check_cuda_fp32_policy", lambda: None)
    with pytest.raises(RuntimeError, match="custom optimizer children"):
        combined.step()
    assert not adam.state and torch.equal(model.weight, torch.ones(1))


def test_npu_precision_scope_still_restores_hf32(monkeypatch):
    from contextlib import nullcontext

    backend = types.SimpleNamespace(allow_hf32=True)
    monkeypatch.setattr(torch, "npu", types.SimpleNamespace(matmul=backend), raising=False)
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    with module._strict_matmul("npu"):
        assert backend.allow_hf32 is False
    assert backend.allow_hf32 is True


@pytest.mark.parametrize("compiled", [False, True])
def test_multioptimizer_direct_load_does_not_run_a_synthetic_cuda_step(monkeypatch, compiled, tmp_path):
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.optim import optimizer as factory
    from veomni.optim import qwen38_muon as implementation

    def make():
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(2, 2))
        if compiled:
            model = torch.compile(model, backend="eager")
        opt = implementation.Qwen38MuonAdamW(
            [{"params": [model.weight], "qwen38_recipe": implementation.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}]
        )
        combined = factory.MultiOptimizer(model, {"muon": opt}, ["muon"])
        return model, opt, combined

    source, source_opt, source_combined = make()
    source.weight.grad = torch.full_like(source.weight, 0.5)
    source_combined.step()
    source_combined.zero_grad()
    parallel = types.SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path), {"model": source, "optimizer": source_combined}, parallel_state=parallel
    )
    target, target_opt, target_combined = make()
    monkeypatch.setattr(implementation, "_optimizer_uses_cuda", lambda _: True)
    monkeypatch.setattr(factory, "_optimizer_uses_cuda", lambda _: True)

    def unexpected_step(*args, **kwargs):
        raise AssertionError("checkpoint load must not synthesize an optimizer step")

    monkeypatch.setattr(implementation, "_cuda_step_preflight", unexpected_step)
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target, "optimizer": target_combined}, parallel_state=parallel
    )
    assert torch.equal(target.weight, source.weight)
    for key in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            target_opt.state[target.weight][key], source_opt.state[source.weight][key], rtol=0, atol=0
        )
    monkeypatch.undo()
    source.weight.grad = torch.full_like(source.weight, 0.25)
    target.weight.grad = source.weight.grad.clone()
    source_combined.step()
    target_combined.step()
    torch.testing.assert_close(target.weight, source.weight, rtol=0, atol=0)


def test_cuda_subclass_override_cannot_run_before_outer_preflight(monkeypatch):
    from functools import wraps

    make_optimizer(torch.nn.Parameter(torch.ones(2, 2)), [("muon", 0, 2)])

    class Child(module.Qwen38MuonAdamW):
        @wraps(module.Qwen38MuonAdamW.step)
        def step(self, *args, **kwargs):
            with torch.no_grad():
                self.param_groups[0]["params"][0].add_(10)
            return super().step(*args, **kwargs)

    param = torch.nn.Parameter(torch.ones(2, 2))
    param.grad = torch.ones_like(param)
    opt = Child([{"params": [param], "qwen38_recipe": module.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}])
    monkeypatch.setattr(module, "_optimizer_uses_cuda", lambda _: True)
    monkeypatch.setattr(module, "_check_cuda_fp32_policy", lambda: None)
    with pytest.raises(RuntimeError, match="custom optimizer children"):
        opt.step()
    assert not opt.state and torch.equal(param, torch.ones_like(param))


@pytest.mark.parametrize("positional", [False, True])
def test_cuda_closure_is_rejected_before_execution_or_update(monkeypatch, positional):
    param = torch.nn.Parameter(torch.ones(2, 2))
    param.grad = torch.ones_like(param)
    opt = make_optimizer(param, [("muon", 0, 2)])
    called = []

    def closure():
        called.append(True)

    monkeypatch.setattr(module, "_optimizer_uses_cuda", lambda _: True)
    monkeypatch.setattr(module, "_check_cuda_fp32_policy", lambda: None)
    with pytest.raises(RuntimeError, match="closures"):
        if positional:
            opt.step(closure)
        else:
            opt.step(closure=closure)
    assert not called and not opt.state and torch.equal(param, torch.ones_like(param))


@pytest.mark.parametrize("failure", [None, "partial", "missing_group", "pre_hook", "post_hook"])
def test_multioptimizer_load_preserves_lazy_parameters_and_failed_state(failure, monkeypatch, tmp_path):
    import pickle

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.checkpoint.layout import optimizer_dir
    from veomni.optim.optimizer import MultiOptimizer

    def make():
        model = torch.nn.Module()
        model.a = torch.nn.Parameter(torch.ones(2, 2))
        model.b = torch.nn.Parameter(torch.ones(2, 2))
        groups = [
            {"params": [p], "qwen38_recipe": module.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}
            for p in (model.a, model.b)
        ]
        opt = module.Qwen38MuonAdamW(groups)
        return model, opt, MultiOptimizer(model, {"muon": opt}, ["muon"])

    source, source_opt, source_combined = make()
    source.a.grad = torch.full_like(source.a, 0.5)
    source_combined.step()
    source_combined.zero_grad()
    source_opt.param_groups[0]["lr"] = 0.0025
    parallel = types.SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path), {"model": source, "optimizer": source_combined}, parallel_state=parallel
    )
    target, target_opt, target_combined = make()
    with torch.no_grad():
        target.a.add_(2)
    before_weight = target.a.detach().clone()
    before_state = copy.deepcopy(target_opt.state_dict())
    if failure in ("partial", "missing_group"):
        metadata_path = Path(optimizer_dir(str(tmp_path))) / ".metadata"
        with metadata_path.open("rb") as handle:
            metadata = pickle.load(handle)
        suffix = "state.a.exp_avg" if failure == "partial" else "param_groups.a.lr"
        keys = [key for key in metadata.state_dict_metadata if key.endswith(suffix)]
        assert len(keys) == 1
        del metadata.state_dict_metadata[keys[0]]
        with metadata_path.open("wb") as handle:
            pickle.dump(metadata, handle)
    elif failure == "pre_hook":
        target_opt.register_load_state_dict_pre_hook(lambda *_: None)
    elif failure == "post_hook":
        target_opt.register_load_state_dict_post_hook(lambda *_: None)

    if failure in ("partial", "missing_group"):
        with pytest.raises((RuntimeError, KeyError)):
            DistributedCheckpointer.load(
                str(tmp_path), {"model": target, "optimizer": target_combined}, parallel_state=parallel
            )
        torch.testing.assert_close(target.a, before_weight, rtol=0, atol=0)
        assert target_opt.state_dict()["param_groups"] == before_state["param_groups"]
        assert not target_opt.state.get(target.b)
        assert not target_opt.state
        return

    DistributedCheckpointer.load(
        str(tmp_path), {"model": target, "optimizer": target_combined}, parallel_state=parallel
    )
    assert not target_opt.state.get(target.b)
    source.a.grad = target.a.grad = None
    source.b.grad = torch.full_like(source.b, 0.25)
    target.b.grad = source.b.grad.clone()
    source_combined.step()
    target_combined.step()
    torch.testing.assert_close(target.b, source.b, rtol=0, atol=0)


def test_multioptimizer_cuda_load_is_atomic_across_children(monkeypatch, tmp_path):
    import pickle

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.checkpoint.layout import optimizer_dir
    from veomni.optim.optimizer import MultiOptimizer

    def make():
        model = torch.nn.Module()
        model.a = torch.nn.Parameter(torch.ones(2, 2))
        model.b = torch.nn.Parameter(torch.ones(2, 2))
        opts = {
            name: module.Qwen38MuonAdamW(
                [{"params": [param], "qwen38_recipe": module.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}]
            )
            for name, param in (("a", model.a), ("b", model.b))
        }
        return model, opts, MultiOptimizer(model, opts, ["a", "b"])

    source, _, source_combined = make()
    source.a.grad = torch.ones_like(source.a)
    source.b.grad = torch.ones_like(source.b)
    source_combined.step()
    source_combined.zero_grad()
    parallel = types.SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path), {"model": source, "optimizer": source_combined}, parallel_state=parallel
    )
    target, opts, target_combined = make()
    with torch.no_grad():
        target.a.add_(3)
        target.b.add_(5)
    before = {name: param.detach().clone() for name, param in target.named_parameters()}
    metadata_path = Path(optimizer_dir(str(tmp_path))) / ".metadata"
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    keys = [key for key in metadata.state_dict_metadata if key.endswith("param_groups.b.qwen38_plan")]
    assert len(keys) == 1
    del metadata.state_dict_metadata[keys[0]]
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)
    with pytest.raises(RuntimeError, match="inventory|hook"):
        DistributedCheckpointer.load(
            str(tmp_path), {"model": target, "optimizer": target_combined}, parallel_state=parallel
        )
    for name, param in target.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)
    assert all(not opt.state for opt in opts.values())


@pytest.mark.parametrize("kind", ["adamw", "amsgrad", "offload", "full", "master"])
def test_multioptimizer_cuda_load_initializes_adam_without_synthetic_step(monkeypatch, kind, tmp_path):
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.optim import optimizer as factory

    def make(*, target=False):
        model = torch.nn.Module()
        model.a = torch.nn.Parameter(torch.ones(2, 2))
        model.b = torch.nn.Parameter(torch.ones(2, 2))
        model.c = torch.nn.Parameter(torch.ones(2, 2))
        qwen = module.Qwen38MuonAdamW(
            [{"params": [model.a], "qwen38_recipe": module.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}]
        )
        if kind == "adamw":
            adam = torch.optim.AdamW([model.b, model.c])
        elif kind == "amsgrad":
            adam = torch.optim.AdamW([model.b, model.c], amsgrad=not target)
        else:
            cls = {
                "offload": factory.CPUOffloadAdamW,
                "full": factory.FullFP32AdamW,
                "master": factory.MasterFP32AdamW,
            }[kind]
            adam = cls([model.b, model.c])
        return model, adam, factory.MultiOptimizer(model, {"qwen": qwen, "adam": adam}, ["qwen", "adam"])

    source, source_adam, source_combined = make()
    source.a.grad = torch.ones_like(source.a)
    source.b.grad = torch.ones_like(source.b)
    for _ in range(3):
        source_combined.step()
    source_combined.zero_grad()
    parallel = types.SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path), {"model": source, "optimizer": source_combined}, parallel_state=parallel
    )
    target, target_adam, target_combined = make(target=True)
    before = target.b.detach().clone()
    monkeypatch.setattr(
        module,
        "_cuda_step_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("synthetic step")),
    )
    if kind == "amsgrad":
        with pytest.raises(RuntimeError, match="layout|group|inventory"):
            DistributedCheckpointer.load(
                str(tmp_path), {"model": target, "optimizer": target_combined}, parallel_state=parallel
            )
        torch.testing.assert_close(target.b, before, rtol=0, atol=0)
        assert not target_adam.state
        return
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target, "optimizer": target_combined}, parallel_state=parallel
    )
    assert not target_adam.state.get(target.c)
    for key in source_adam.state[source.b]:
        torch.testing.assert_close(target_adam.state[target.b][key], source_adam.state[source.b][key], rtol=0, atol=0)
    monkeypatch.undo()
    source.a.grad = target.a.grad = torch.ones_like(source.a)
    source.b.grad = target.b.grad = torch.ones_like(source.b)
    target_combined.step()
    source_combined.step()
    torch.testing.assert_close(target.b, source.b, rtol=0, atol=0)


def test_multioptimizer_cuda_checkpoint_preserves_lazy_non_muon_child(monkeypatch, tmp_path):
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.optim.optimizer import MultiOptimizer

    def make():
        model = torch.nn.Module()
        model.a = torch.nn.Parameter(torch.ones(2, 2))
        model.b = torch.nn.Parameter(torch.ones(2, 2))
        qwen = module.Qwen38MuonAdamW(
            [{"params": [model.a], "qwen38_recipe": module.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}]
        )
        adam = torch.optim.AdamW([model.b])
        return model, adam, MultiOptimizer(model, {"qwen": qwen, "adam": adam}, ["qwen", "adam"])

    source, source_adam, optimizer = make()
    source.a.grad = torch.ones_like(source.a)
    optimizer.step()
    optimizer.zero_grad()
    parallel = types.SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    assert not source_adam.state
    target, target_adam, restored = make()
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    assert not target_adam.state
    source.b.grad = target.b.grad = torch.ones_like(source.b)
    optimizer.step()
    restored.step()
    torch.testing.assert_close(target.b, source.b, rtol=0, atol=0)


def test_multioptimizer_rejects_empty_group_without_synthetic_step(tmp_path):
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.optim.optimizer import MultiOptimizer

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2, 2))
    qwen = module.Qwen38MuonAdamW(
        [{"params": [model.weight], "qwen38_recipe": module.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}]
    )
    empty = torch.optim.AdamW([{"params": []}])
    combined = MultiOptimizer(model, {"qwen": qwen, "empty": empty}, ["qwen", "empty"])
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="empty parameter group"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": combined}, parallel_state=types.SimpleNamespace(dp_mode="ddp")
        )
    assert not empty.state and torch.equal(model.weight, before)
    assert not list(tmp_path.iterdir())


def test_non_cuda_malformed_layout_does_not_clear_zero_state_resume():
    from types import SimpleNamespace

    from torch.distributed.tensor import Shard

    from veomni.checkpoint.dcp_checkpointer import OptimizerState
    from veomni.distributed.parallel_plan import SpecInfo

    model = torch.nn.Linear(2, 2, bias=False)
    model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", None, 0, None, (2, 2), True)}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.state[model.weight]["saved"] = torch.tensor(17.0)
    optimizer._veomni_resume_zero_optimizer_state = True
    wrapper = OptimizerState(model, optimizer, parallel_state=SimpleNamespace(dp_mode="fsdp2"), load=True)
    with pytest.raises(RuntimeError, match="owner-aware checkpoint codec"):
        wrapper.load_state_dict({})
    assert optimizer.state[model.weight]["saved"].item() == 17
    assert optimizer._veomni_resume_zero_optimizer_state


def test_polar_disables_ambient_autocast():
    x = torch.randn(8, 17)
    ref = module.polar_express_fp32(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = module.polar_express_fp32(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, ref, rtol=0, atol=0)


def test_dcp_source_metadata_requires_recipe_before_partial_load(tmp_path):
    import pickle

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.checkpoint.layout import optimizer_dir
    from veomni.optim.optimizer import MultiOptimizer

    def make():
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(2, 2))
        leaf = module.Qwen38MuonAdamW(
            [{"params": [model.weight], "qwen38_recipe": module.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}]
        )
        return model, MultiOptimizer(model, {"muon": leaf}, ["muon"])

    source, optimizer = make()
    parallel = types.SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    metadata_path = Path(optimizer_dir(str(tmp_path))) / ".metadata"
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    keys = [key for key in metadata.state_dict_metadata if key.endswith("param_groups.weight.qwen38_recipe")]
    assert len(keys) == 1
    del metadata.state_dict_metadata[keys[0]]
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)
    target, restored = make()
    with torch.no_grad():
        target.weight.add_(2)
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="inventory"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def _cold_process_worker(rank, world, rendezvous, path, phase):
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        torch.manual_seed(99)
        initial = torch.randn(12, 9).bfloat16()
        gradients = [torch.randn_like(initial) * 0.01 for _ in range(4)]
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(distribute_tensor(initial, mesh, [Shard(0)]))
        plan = [("muon", 0, 3), ("adamw", 3, 6), ("muon", 6, 9), ("adamw", 9, 12)]
        opt = make_optimizer(model.weight, plan)
        if phase == "save":
            for grad in gradients[:3]:
                model.weight.grad = distribute_tensor(grad, mesh, [Shard(0)])
                opt.step()
            msd, osd = get_state_dict(model, opt)
            dcp.save({"model": msd, "optimizer": osd}, checkpoint_id=path)
        else:
            model.weight.grad = torch.zeros_like(model.weight)
            opt.step()
            msd, osd = get_state_dict(model, opt)
            dcp.load({"model": msd, "optimizer": osd}, checkpoint_id=path)
            set_state_dict(model, opt, model_state_dict=msd, optim_state_dict=osd)
            model.weight.grad = distribute_tensor(gradients[3], mesh, [Shard(0)])
            opt.step()
            reference_p = torch.nn.Parameter(initial.clone())
            reference = make_optimizer(reference_p, plan)
            for grad in gradients:
                reference_p.grad = grad
                reference.step()
            for key in ("master_weight", "exp_avg", "exp_avg_sq"):
                actual = opt.state[model.weight][key].to_local()
                expected = distribute_tensor(reference.state[reference_p][key], mesh, [Shard(0)]).to_local()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_SHARD_LAUNCH)
def test_fresh_process_dcp_next_update_is_exact(tmp_path):
    import torch.multiprocessing as mp

    for phase in ("save", "resume"):
        mp.start_processes(
            _cold_process_worker,
            args=(_CI_SHARD_LAUNCH.nproc, str(tmp_path / f"rdzv-{phase}"), str(tmp_path / "checkpoint"), phase),
            nprocs=_CI_SHARD_LAUNCH.nproc,
            join=True,
            start_method="fork",
        )


def _ple_asymmetric_worker(rank, world, rendezvous, path, phase, mode):
    from types import SimpleNamespace

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.distributed.parallel_plan import SpecInfo
    from veomni.optim.optimizer import MultiOptimizer

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (2 if mode == "persistent" else 1, 2), mesh_dim_names=("ple_fsdp", "ple"))
        placements = [Shard(1), Shard(0)]
        model = torch.nn.Module()
        model.ngram_embedding = torch.nn.ModuleDict(
            {name: torch.nn.Embedding(8, 8, dtype=torch.bfloat16) for name in ("all", "partial", "never")}
        )
        for embedding in model.ngram_embedding.values():
            full = torch.arange(64).reshape(8, 8).bfloat16() / 64
            embedding.weight = torch.nn.Parameter(distribute_tensor(full, mesh, placements))
        model.lm_head = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        model.lm_head.weight.data.fill_(0.125)
        groups = module.add_qwen38_plans(
            model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
        )
        leaf = module.Qwen38MuonAdamW(groups, lr=1e-4)
        optimizer = MultiOptimizer(model, {"ple": leaf}, ["ple"]) if mode != "single" else leaf
        model._fqn2spec_info = {
            name: SpecInfo("ple", Shard(0), name, mesh, 1, 1, (8, 8))
            for name, _ in model.named_parameters()
            if "ngram_embedding" in name
        }
        ps = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=["ple"], extra_parallel_fsdp_device_mesh={"ple": mesh}
        )
        partial = model.ngram_embedding["partial"].weight
        never = model.ngram_embedding["never"].weight
        active = rank % 2 == 0

        def update(first):
            leaf.zero_grad(set_to_none=True)
            for name, param in model.named_parameters():
                if "never" in name or (first and "partial" in name and not active):
                    continue
                scale = 0.125 if first else 0.25
                if "ngram_embedding" in name:
                    # The owner received local row requests; other owners never
                    # enter lookup and retain grad=None for this table.
                    local = param.to_local()
                    torch.nn.functional.embedding(torch.tensor([0, 1, 1]), local).sum().mul(scale).backward()
                else:
                    param.grad = torch.ones_like(param) * scale
            leaf.step()

        if phase == "save":
            update(True)
            leaf.zero_grad(set_to_none=True)
            before = {p: dict(s) for p, s in leaf.state.items()}
            if mode == "single":
                with pytest.raises(RuntimeError, match="require MultiOptimizer"):
                    DistributedCheckpointer.save(path, {"model": model, "optimizer": optimizer}, parallel_state=ps)
                assert set(leaf.state) == set(before)
                return
            DistributedCheckpointer.save(path, {"model": model, "optimizer": optimizer}, parallel_state=ps)
            assert set(leaf.state) == set(before)
            assert partial in leaf.state if active else partial not in leaf.state
            assert never not in leaf.state
            update(False)
            torch.save(
                {
                    name: {
                        "weight": _local_test(p).clone(),
                        "state": {k: _local_test(v).clone() for k, v in leaf.state.get(p, {}).items()},
                    }
                    for name, p in model.named_parameters()
                },
                Path(path) / f"expected-{rank}.pt",
            )
        else:
            # Deliberately wrong model contents expose stale lazy master copies.
            for p in model.parameters():
                p.data.fill_(-1)
            DistributedCheckpointer.load(path, {"model": model, "optimizer": optimizer}, parallel_state=ps)
            assert partial in leaf.state if active else partial not in leaf.state
            if active:
                assert leaf.state[partial]["step"].item() == 1
            assert never not in leaf.state
            update(False)
            assert leaf.state[partial]["step"].item() == (2 if active else 1)
            expected = torch.load(Path(path) / f"expected-{rank}.pt", weights_only=True)
            for name, p in model.named_parameters():
                torch.testing.assert_close(_local_test(p), expected[name]["weight"], rtol=0, atol=0)
                actual = {k: _local_test(v) for k, v in leaf.state.get(p, {}).items()}
                assert actual.keys() == expected[name]["state"].keys()
                for key in actual:
                    torch.testing.assert_close(actual[key], expected[name]["state"][key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _local_test(tensor):
    from torch.distributed.tensor import DTensor

    return tensor.to_local() if isinstance(tensor, DTensor) else tensor.detach()


def _ple_owner_launch(mode):
    return LaunchSpec(nproc=4 if mode == "persistent" else 2, device="cpu")


@pytest.mark.launch_plan(provider=lambda context: _ple_owner_launch(context.params["mode"]))
@pytest.mark.parametrize("mode", ["single", "multi", "persistent"])
def test_ple_asymmetric_owner_checkpoint_and_cold_resume(tmp_path, mode):
    import torch.multiprocessing as mp

    for phase in ("save",) if mode == "single" else ("save", "resume"):
        mp.start_processes(
            _ple_asymmetric_worker,
            args=(
                _ple_owner_launch(mode).nproc,
                str(tmp_path / f"rdzv-{phase}"),
                str(tmp_path / "checkpoint"),
                phase,
                mode,
            ),
            nprocs=_ple_owner_launch(mode).nproc,
            join=True,
            start_method="fork",
        )


def test_unused_ple_snapshot_does_not_initialize_or_advance_live_state(tmp_path):
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.optim.optimizer import MultiOptimizer

    def build():
        model = torch.nn.Module()
        model.ngram_embedding = torch.nn.Embedding(7, 4, dtype=torch.bfloat16)
        groups = module.add_qwen38_plans(
            model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
        )
        leaf = module.Qwen38MuonAdamW(groups, lr=1e-4)
        return model, leaf, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    model, opt, optimizer = build()
    parallel = types.SimpleNamespace(dp_mode="ddp")
    checkpoint = str(tmp_path / "unused-ple")
    DistributedCheckpointer.save(checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    assert not opt.state

    restored_model, restored_opt, restored = build()
    DistributedCheckpointer.load(checkpoint, {"model": restored_model, "optimizer": restored}, parallel_state=parallel)
    assert not restored_opt.state
    for current, target in zip(model.parameters(), restored_model.parameters(), strict=True):
        current.grad = torch.ones_like(current)
        target.grad = torch.ones_like(target)
    optimizer.step()
    restored.step()
    for current, target in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(current, target, rtol=0, atol=0)
        assert opt.state[current].keys() == restored_opt.state[target].keys()
        for key in opt.state[current]:
            torch.testing.assert_close(opt.state[current][key], restored_opt.state[target][key], rtol=0, atol=0)

    # A raw pre-versioned PLE checkpoint must not silently broadcast a scalar.
    direct = opt.state_dict()
    direct["param_groups"][0].pop("qwen38_ple_step_layout")
    with pytest.raises(RuntimeError, match="explicit migration"):
        opt.load_state_dict(direct)


def test_ple_load_hooks_receive_saved_state_before_lazy_normalization():
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(7, 4, dtype=torch.bfloat16)

    def build():
        groups = module.add_qwen38_plans(
            model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
        )
        return module.Qwen38MuonAdamW(groups, lr=1e-4)

    source = build()
    saved = copy.deepcopy(source.state_dict())
    key = saved["param_groups"][0]["params"][0]
    assert saved["state"][key]["step"].item() == 0
    target = model.ngram_embedding.weight
    lazy = build()
    lazy.load_state_dict(copy.deepcopy(saved))
    assert target not in lazy.state

    observed = []
    read_only = build()
    read_only.register_load_state_dict_pre_hook(
        lambda optimizer, incoming: observed.append(incoming["state"][key]["step"])
    )
    read_only.load_state_dict(copy.deepcopy(saved))
    assert observed[0].item() == 0
    assert read_only.state[target]["step"].item() == 0

    resumed = build()
    events = []

    def before_load(optimizer, incoming):
        assert optimizer is resumed
        assert incoming["state"][key]["step"].item() == 0
        incoming["state"][key]["exp_avg"].fill_(0.5)
        events.append("before")

    def after_load(optimizer):
        assert optimizer.state[target]["step"].item() == 0
        torch.testing.assert_close(
            optimizer.state[target]["exp_avg"], torch.full_like(target, 0.5, dtype=torch.float32)
        )
        events.append("after")

    resumed.register_load_state_dict_pre_hook(before_load)
    resumed.register_load_state_dict_post_hook(after_load)
    resumed.load_state_dict(saved)
    assert events == ["before", "after"]
    target.grad = torch.zeros_like(target)
    resumed.step()
    torch.testing.assert_close(resumed.state[target]["exp_avg"], torch.full_like(target, 0.45, dtype=torch.float32))


def test_ple_zero_step_pre_hook_master_update_is_preserved():
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(7, 4, dtype=torch.bfloat16)
    groups = module.add_qwen38_plans(
        model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
    )
    source = module.Qwen38MuonAdamW(groups, lr=1e-4, chunk_size=2)
    saved = copy.deepcopy(source.state_dict())
    target = model.ngram_embedding.weight
    resumed = module.Qwen38MuonAdamW(groups, lr=1e-4, chunk_size=2)

    def modify_master(optimizer, incoming):
        key = incoming["param_groups"][0]["params"][0]
        incoming["state"][key]["master_weight"][-1, -1].add_(1e-6)

    resumed.register_load_state_dict_pre_hook(modify_master)
    resumed.load_state_dict(saved)
    expected = target.detach().float()
    expected[-1, -1] += 1e-6
    torch.testing.assert_close(resumed.state[target]["master_weight"], expected, rtol=0, atol=0)


def test_ple_zero_step_pre_hook_step_dtype_mismatch_is_rejected():
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(7, 4, dtype=torch.bfloat16)
    groups = module.add_qwen38_plans(
        model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
    )
    source = module.Qwen38MuonAdamW(groups, lr=1e-4)
    saved = copy.deepcopy(source.state_dict())
    target = model.ngram_embedding.weight
    resumed = module.Qwen38MuonAdamW(groups, lr=1e-4)

    def modify_step(optimizer, incoming):
        key = incoming["param_groups"][0]["params"][0]
        incoming["state"][key]["step"] = incoming["state"][key]["step"].double()

    resumed.register_load_state_dict_pre_hook(modify_step)
    with pytest.raises(RuntimeError, match="step must be an FP32 scalar"):
        resumed.load_state_dict(saved)
    assert target not in resumed.state


def test_ple_pre_hook_can_migrate_unreadable_step():
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(7, 4, dtype=torch.bfloat16)
    groups = module.add_qwen38_plans(
        model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
    )
    source = module.Qwen38MuonAdamW(groups, lr=1e-4)
    saved = copy.deepcopy(source.state_dict())
    key = saved["param_groups"][0]["params"][0]
    saved["state"][key]["step"] = torch.empty((), device="meta")
    target = model.ngram_embedding.weight
    resumed = module.Qwen38MuonAdamW(groups, lr=1e-4)

    def migrate(optimizer, incoming):
        incoming["state"][key]["step"] = torch.tensor(0.0)

    resumed.register_load_state_dict_pre_hook(migrate)
    resumed.load_state_dict(saved)
    assert resumed.state[target]["step"].item() == 0


def test_ple_pre_hook_can_repair_parameter_groups():
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(7, 4, dtype=torch.bfloat16)
    groups = module.add_qwen38_plans(
        model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
    )
    source = module.Qwen38MuonAdamW(groups, lr=1e-4)
    saved = copy.deepcopy(source.state_dict())
    damaged = {**saved, "param_groups": []}
    resumed = module.Qwen38MuonAdamW(groups, lr=1e-4)
    resumed.register_load_state_dict_pre_hook(lambda optimizer, incoming: copy.deepcopy(saved))
    resumed.load_state_dict(damaged)
    assert resumed.state[model.ngram_embedding.weight]["step"].item() == 0


def test_ple_zero_step_pre_hook_stride_change_is_preserved():
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(1, 2, dtype=torch.bfloat16)
    groups = module.add_qwen38_plans(
        model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
    )
    source = module.Qwen38MuonAdamW(groups, lr=1e-4)
    saved = copy.deepcopy(source.state_dict())
    resumed = module.Qwen38MuonAdamW(groups, lr=1e-4)

    def change_stride(optimizer, incoming):
        key = incoming["param_groups"][0]["params"][0]
        master = incoming["state"][key]["master_weight"]
        incoming["state"][key]["master_weight"] = master.as_strided(master.shape, (99, 1))

    resumed.register_load_state_dict_pre_hook(change_stride)
    resumed.load_state_dict(saved)
    assert resumed.state[model.ngram_embedding.weight]["master_weight"].stride() == (99, 1)


def test_ple_zero_step_pre_hook_state_alias_is_preserved():
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(2, 2, dtype=torch.bfloat16)
    groups = module.add_qwen38_plans(
        model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
    )
    source = module.Qwen38MuonAdamW(groups, lr=1e-4)
    saved = copy.deepcopy(source.state_dict())
    resumed = module.Qwen38MuonAdamW(groups, lr=1e-4)

    def alias_state(optimizer, incoming):
        key = incoming["param_groups"][0]["params"][0]
        entry = incoming["state"][key]
        entry["exp_avg_sq"] = entry["exp_avg"]

    resumed.register_load_state_dict_pre_hook(alias_state)
    resumed.load_state_dict(saved)
    state = resumed.state[model.ngram_embedding.weight]
    assert state["exp_avg"].untyped_storage().data_ptr() == state["exp_avg_sq"].untyped_storage().data_ptr()


def test_ple_read_only_hook_retains_saved_zero_step_state():
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(7, 4, dtype=torch.bfloat16)
    groups = module.add_qwen38_plans(
        model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
    )
    source = module.Qwen38MuonAdamW(groups, lr=1e-4)
    saved = copy.deepcopy(source.state_dict())
    target = model.ngram_embedding.weight
    resumed = module.Qwen38MuonAdamW(groups, lr=1e-4)
    resumed.register_load_state_dict_pre_hook(lambda optimizer, incoming: None)
    with torch.no_grad():
        target.fill_(3.0)
    resumed.load_state_dict(saved)
    saved_master = resumed.state[target]["master_weight"].clone()
    with torch.no_grad():
        target.fill_(7.0)
    target.grad = torch.zeros_like(target)
    resumed.step()
    torch.testing.assert_close(resumed.state[target]["master_weight"], saved_master, rtol=0, atol=0)


def test_dcp_internal_filter_does_not_materialize_lazy_ple(tmp_path):
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.optim.optimizer import MultiOptimizer

    def build():
        model = torch.nn.Module()
        model.ngram_embedding = torch.nn.Embedding(7, 4, dtype=torch.bfloat16)
        model.embed_tokens = torch.nn.Embedding(7, 4, dtype=torch.bfloat16)
        model.lm_head = torch.nn.Linear(4, 7, bias=False, dtype=torch.bfloat16)
        groups = module.add_qwen38_plans(
            model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
        )
        leaf = module.Qwen38MuonAdamW(groups, lr=1e-4)
        return model, leaf, MultiOptimizer(model, {"qwen": leaf}, ["qwen"])

    source_model, source, optimizer = build()
    source_model.lm_head.weight.grad = torch.ones_like(source_model.lm_head.weight)
    optimizer.step()
    optimizer.zero_grad()
    parallel = types.SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path), {"model": source_model, "optimizer": optimizer}, parallel_state=parallel
    )
    target_model, target, restored = build()
    DistributedCheckpointer.load(
        str(tmp_path), {"model": target_model, "optimizer": restored}, parallel_state=parallel
    )
    assert target_model.ngram_embedding.weight not in target.state
    assert target_model.embed_tokens.weight not in target.state
    assert target_model.lm_head.weight in target.state
    for field, value in source.state[source_model.lm_head.weight].items():
        torch.testing.assert_close(target.state[target_model.lm_head.weight][field], value, rtol=0, atol=0)
