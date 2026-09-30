"""Logical-slice, precision, and cold-resume tests independent of model kernels."""

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


PACKAGE = "_qwen38_optimizer_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).parents[2] / "veomni/optim")]
sys.modules.setdefault(PACKAGE, package)
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.qwen38_muon", Path(package.__path__[0]) / "qwen38_muon.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


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
        # Uneven row shards cut straight through interleaved Q/gate blocks.
        # Column sharding exercises the other FSDP matrix axis. Expert-axis
        # sharding must keep expert-local orthogonalization communication-free.
        for shape, axis in [((12, 9), 0), ((12, 9), 1), ((4, 8, 9), 0), ((4, 8, 9), 1)]:
            torch.manual_seed(123)
            full = torch.randn(shape).bfloat16()
            model = torch.nn.Module()
            model.register_parameter("weight", torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(axis)])))
            p = model.weight
            plan = (
                [("muon", 0, 3), ("adamw", 3, 6), ("muon", 6, 9), ("adamw", 9, 12)]
                if len(shape) == 2
                else [("muon", 0, 4), ("muon", 4, 8)]
            )
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
                    path = str(Path(checkpoint_dir) / f"ndim{len(shape)}-axis{axis}")
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


def test_distributed_updates_and_dcp_resume(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _distributed_worker,
        args=(2, str(tmp_path / "rdzv"), str(tmp_path / "dcp")),
        nprocs=2,
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


def test_receipt_includes_muon_states(capsys):
    p = torch.nn.Parameter(torch.ones(8, 5))
    opt = make_optimizer(p, [("muon", 0, 8)])
    p.grad = torch.ones_like(p)
    opt.step()
    lines = capsys.readouterr().out.splitlines()
    receipt = json.loads(
        next(line.removeprefix("QWEN38_MUON_STATE ") for line in lines if line.startswith("QWEN38_MUON_STATE "))
    )
    assert receipt["initialized"] == receipt["configured"] == 1
    assert sum(receipt["state_bytes"].values()) == 8 * 5 * 4 * 3


def test_polar_disables_ambient_autocast():
    x = torch.randn(8, 17)
    ref = module.polar_express_fp32(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = module.polar_express_fp32(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, ref, rtol=0, atol=0)


def test_dcp_source_metadata_requires_recipe_before_partial_load():
    # Execute the actual planner class without importing unrelated GPU kernels.
    import ast
    from types import SimpleNamespace

    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    source = Path(__file__).parents[2] / "veomni/checkpoint/dcp_checkpointer.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "_ModelStrictLoadPlanner"
    )
    scope = {"DefaultLoadPlanner": DefaultLoadPlanner, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    planner = scope["_ModelStrictLoadPlanner"](False, strict_optimizer_recipe=True)
    planner.metadata = SimpleNamespace(state_dict_metadata={})
    planner.mappings = {"optimizer.param_groups.0.qwen38_recipe": ("optimizer", "param_groups", 0, "qwen38_recipe")}
    with pytest.raises(RuntimeError, match="recipe/state is incomplete"):
        planner.create_local_plan()


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


def test_fresh_process_dcp_next_update_is_exact(tmp_path):
    import torch.multiprocessing as mp

    for phase in ("save", "resume"):
        mp.start_processes(
            _cold_process_worker,
            args=(2, str(tmp_path / f"rdzv-{phase}"), str(tmp_path / "checkpoint"), phase),
            nprocs=2,
            join=True,
            start_method="fork",
        )
