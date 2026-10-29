"""CPU reference and cold-state policy checks for optimizer-only offload."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import (
    _ZERO_OPTIMIZER_STATE_MARKER,
    OptimizerState,
    _ModelStrictLoadPlanner,
    _requires_exact_optimizer_recipe,
    _requires_exact_optimizer_state_dtypes,
)
from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW, select_cpu_offload_groups
from veomni.optim.master_fp32_adamw import FullFP32AdamW, MasterFP32AdamW


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
_CI_LAYOUT_LAUNCH = LaunchSpec(nproc=4, device="cpu")


def _extra_parallel_launch(hsdp):
    return LaunchSpec(nproc=8 if hsdp else 4, device="cpu")


def test_cpu_offload_pattern_requires_a_trainable_match():
    model = torch.nn.Linear(4, 3)
    model.weight.requires_grad_(False)
    groups = [{"params": list(model.parameters())}]
    with pytest.raises(ValueError, match="no trainable parameters"):
        select_cpu_offload_groups(model, groups, ["weight"])
    selected = select_cpu_offload_groups(model, groups, ["bias"])
    assert selected[0]["cpu_offload"] and selected[0]["params"] == [model.bias]
    assert not selected[1]["cpu_offload"] and selected[1]["params"] == [model.weight]
    both = select_cpu_offload_groups(model, groups, ["weight|bias"])
    assert both[0]["cpu_offload"] and both[0]["params"] == [model.weight, model.bias]


@pytest.mark.parametrize("standard_dtype", [torch.float32, torch.bfloat16])
def test_dcp_mixed_optimizer_only_enforces_dtype_for_exact_leaf(tmp_path, standard_dtype):
    from veomni.optim.optimizer import MultiOptimizer

    def build(dtype):
        model = torch.nn.Module()
        model.p = torch.nn.Parameter(torch.ones(3))
        model.q = torch.nn.Parameter(torch.ones(3, dtype=dtype))
        exact = CPUOffloadAdamW([model.p], lr=0.001)
        standard = torch.optim.AdamW([model.q], lr=0.001)
        return (
            model,
            exact,
            standard,
            MultiOptimizer(model, {"exact": exact, "standard": standard}, ["exact", "standard"]),
        )

    source_model, source_exact, source_standard, source = build(torch.float32)
    source_model.p.grad = torch.ones_like(source_model.p)
    source_model.q.grad = torch.ones_like(source_model.q)
    source_exact.step()
    source_standard.step()
    parallel = SimpleNamespace(dp_mode="ddp")
    dcp.save({"optimizer": OptimizerState(source_model, source, parallel_state=parallel)}, checkpoint_id=tmp_path)

    target_model, target_exact, target_standard, target = build(standard_dtype)
    dcp.load(
        {"optimizer": OptimizerState(target_model, target, parallel_state=parallel, load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(
            strict_model=False, strict_optimizer_dtype=True, optimizer=target, model=target_model
        ),
    )
    assert target_standard.state[target_model.q]["exp_avg"].dtype == standard_dtype
    torch.testing.assert_close(
        target_exact.state[target_model.p]["master_weight"], source_exact.state[source_model.p]["master_weight"]
    )


def test_dcp_mixed_exact_optimizer_rejects_parameter_missing_from_model():
    from veomni.checkpoint.dcp_checkpointer import _exact_optimizer_parameter_names

    model = torch.nn.Linear(3, 2)
    foreign = torch.nn.Parameter(torch.ones(3))
    exact = CPUOffloadAdamW([foreign], lr=0.001)
    ordinary = torch.optim.AdamW(model.parameters(), lr=0.001)
    with pytest.raises(RuntimeError, match="absent from the DCP model inventory"):
        _exact_optimizer_parameter_names(model, {"exact": exact, "ordinary": ordinary})


def test_dcp_mixed_optimizer_respects_wrapper_level_exact_policy():
    from veomni.checkpoint.dcp_checkpointer import _exact_optimizer_parameter_names

    model = torch.nn.Linear(3, 2)
    ordinary = torch.optim.AdamW(model.parameters(), lr=0.001)
    wrapper = SimpleNamespace(requires_exact_state_dtypes=True, optimizers_dict={"ordinary": ordinary})
    assert _exact_optimizer_parameter_names(model, wrapper) is None


def test_dcp_rejects_unknown_exact_optimizer_state_field(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = CPUOffloadAdamW(model.parameters(), lr=0.001)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.state[model.weight]["max_exp_avg_sq"] = torch.ones_like(model.weight)
    parallel = SimpleNamespace(dp_mode="ddp")
    dcp.save({"optimizer": OptimizerState(model, optimizer, parallel_state=parallel)}, checkpoint_id=tmp_path)
    target_model = torch.nn.Linear(3, 2, bias=False)
    target = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
    with pytest.raises(CheckpointException, match="unsupported precision-sensitive optimizer state field"):
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=parallel, load=True)},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(
                strict_model=False, strict_optimizer_dtype=True, optimizer=target, model=target_model
            ),
        )


@pytest.mark.parametrize("multi", [False, True])
def test_dcp_allows_custom_step_in_optimizer_param_group(tmp_path, multi):
    from veomni.optim.optimizer import MultiOptimizer

    def build():
        model = torch.nn.Linear(3, 2, bias=False)
        exact = CPUOffloadAdamW([{"params": [model.weight], "step": 7}], lr=0.001)
        container = MultiOptimizer(model, {"exact": exact}, ["exact"]) if multi else exact
        return model, exact, container

    source_model, exact, source = build()
    source_model.weight.grad = torch.ones_like(source_model.weight)
    exact.step()
    parallel = SimpleNamespace(dp_mode="ddp")
    dcp.save({"optimizer": OptimizerState(source_model, source, parallel_state=parallel)}, checkpoint_id=tmp_path)
    target_model, target_exact, target = build()
    dcp.load(
        {"optimizer": OptimizerState(target_model, target, parallel_state=parallel, load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(
            strict_model=False, strict_optimizer_dtype=True, optimizer=target, model=target_model
        ),
    )
    assert target_exact.param_groups[0]["step"] == 7


def test_dcp_failed_metadata_validation_does_not_leave_lazy_state_handoff(tmp_path):
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    source_model = torch.nn.Linear(3, 2)
    source = CPUOffloadAdamW(source_model.parameters(), lr=0.001)
    source_model.bias.grad = torch.ones_like(source_model.bias)
    source.step()
    source.state[source_model.bias]["unexpected"] = torch.ones_like(source_model.bias)
    parallel = SimpleNamespace(dp_mode="ddp")
    dcp.save({"optimizer": OptimizerState(source_model, source, parallel_state=parallel)}, checkpoint_id=tmp_path)

    target_model = torch.nn.Linear(3, 2)
    target = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
    wrapper = OptimizerState(target_model, target, parallel_state=parallel, load=True)
    with pytest.raises(CheckpointException, match="unsupported precision-sensitive optimizer state field"):
        dcp.load(
            {"optimizer": wrapper},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(
                strict_model=False, strict_optimizer_dtype=True, optimizer=target, model=target_model
            ),
        )
    assert not getattr(target, "_veomni_absent_optimizer_state_bases", None)

    complete_model = torch.nn.Linear(3, 2)
    complete = CPUOffloadAdamW(complete_model.parameters(), lr=0.001)
    for param in complete_model.parameters():
        param.grad = torch.ones_like(param)
    complete.step()
    wrapper.load_state_dict(get_optimizer_state_dict(complete_model, complete))
    torch.testing.assert_close(
        target.state[target_model.weight]["master_weight"], complete.state[complete_model.weight]["master_weight"]
    )


def test_distributed_checkpointer_clears_lazy_handoff_on_load_failure(monkeypatch):
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    model = torch.nn.Linear(3, 2)
    optimizer = CPUOffloadAdamW(model.parameters(), lr=0.001)
    monkeypatch.setattr(DistributedCheckpointer, "_legacy_fused_dir", lambda *_args: "/unused")
    monkeypatch.setattr(DistributedCheckpointer, "_create_storage_reader", lambda *_args: object())

    def fail_after_plan(**_kwargs):
        optimizer._veomni_absent_optimizer_state_bases = {"optimizer.state.weight"}
        raise RuntimeError("storage failure")

    monkeypatch.setattr(dcp, "load", fail_after_plan)
    with pytest.raises(RuntimeError, match="storage failure"):
        DistributedCheckpointer.load(
            "/unused", {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert not getattr(optimizer, "_veomni_absent_optimizer_state_bases", None)


@pytest.mark.parametrize(
    "saved_dtype,target_dtype", [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)]
)
def test_dcp_mixed_optimizer_still_rejects_exact_leaf_dtype_migration(tmp_path, saved_dtype, target_dtype):
    from veomni.optim.optimizer import MultiOptimizer

    def build(dtype):
        model = torch.nn.Module()
        model.p = torch.nn.Parameter(torch.ones(3))
        model.q = torch.nn.Parameter(torch.ones(3))
        exact = CPUOffloadAdamW([model.p], lr=0.001, resident_moment_dtype=dtype)
        standard = torch.optim.AdamW([model.q], lr=0.001)
        return (
            model,
            exact,
            standard,
            MultiOptimizer(model, {"exact": exact, "standard": standard}, ["exact", "standard"]),
        )

    source_model, exact, standard, source = build(saved_dtype)
    source_model.p.grad = torch.ones_like(source_model.p)
    source_model.q.grad = torch.ones_like(source_model.q)
    exact.step()
    standard.step()
    parallel = SimpleNamespace(dp_mode="ddp")
    dcp.save({"optimizer": OptimizerState(source_model, source, parallel_state=parallel)}, checkpoint_id=tmp_path)
    target_model, _, _, target = build(target_dtype)
    with pytest.raises(CheckpointException, match="optimizer dtype mismatch"):
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=parallel, load=True)},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(
                strict_model=False, strict_optimizer_dtype=True, optimizer=target, model=target_model
            ),
        )


@pytest.mark.parametrize("offload", [False, True])
def test_cpu_update_and_cold_resume_match_fp32_reference(offload):
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
    reference = torch.nn.Parameter(p.float().detach())
    opt = CPUOffloadAdamW([{"params": [p], "cpu_offload": offload}], lr=0.001)
    ref = torch.optim.AdamW([reference], lr=0.001, betas=(0.9, 0.95), weight_decay=0)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        reference.grad = p.grad.float()
        opt.step()
        ref.step()
        torch.testing.assert_close(opt.state[p]["master_weight"], reference, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(p, reference.bfloat16(), rtol=0, atol=0)
    clone = torch.nn.Parameter(p.detach().clone())
    resumed = CPUOffloadAdamW([{"params": [clone], "cpu_offload": offload}], lr=0.001)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    clone.grad = p.grad.clone()
    opt.step()
    resumed.step()
    torch.testing.assert_close(clone, p, rtol=0, atol=0)
    for name in ("master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(resumed.state[clone][name], opt.state[p][name], rtol=0, atol=0)


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("moment_dtype", [torch.float32, torch.bfloat16])
def test_resume_hooks_see_complete_state_and_preserve_their_changes(offload, moment_dtype):
    def make(param):
        return CPUOffloadAdamW(
            [{"params": [param], "cpu_offload": offload}],
            lr=0.001,
            resident_moment_dtype=moment_dtype,
            cpu_moment_dtype=moment_dtype,
        )

    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    source = make(param)
    param.grad = torch.arange(4).bfloat16()
    source.step()
    saved = copy.deepcopy(source.state_dict())
    saved["state"]["global_metadata"] = {"version": 7}
    key = saved["param_groups"][0]["params"][0]
    target = torch.nn.Parameter(param.detach().clone())
    resumed = make(target)
    events = []

    def before_load(optimizer, incoming):
        events.append("before")
        assert optimizer is resumed
        assert incoming["state"][key]["master_weight"].dtype == torch.float32
        replacement = copy.deepcopy(incoming)
        replacement["param_groups"][0]["lr"] = 0.002
        replacement["state"][key]["exp_avg"].add_(0.125)
        return replacement

    def after_load(optimizer):
        events.append("after")
        assert optimizer.state["global_metadata"] == {"version": 7}
        state = optimizer.state[target]
        assert set(state) == {"step", "master_weight", "exp_avg", "exp_avg_sq"}
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == moment_dtype
        state["exp_avg"].add_(0.25)

    resumed.register_load_state_dict_pre_hook(before_load)
    resumed.register_load_state_dict_post_hook(after_load)
    resumed.load_state_dict(saved)
    assert events == ["before", "after"]
    assert resumed.param_groups[0]["lr"] == 0.002

    expected = copy.deepcopy(saved)
    expected["param_groups"][0]["lr"] = 0.002
    expected["state"][key]["exp_avg"].add_(0.125).add_(0.25)
    reference_param = torch.nn.Parameter(param.detach().clone())
    reference = make(reference_param)
    reference.load_state_dict(expected)
    for parameter, optimizer in ((target, resumed), (reference_param, reference)):
        parameter.grad = torch.arange(4).flip(0).bfloat16()
        optimizer.step()
    torch.testing.assert_close(target, reference_param, rtol=0, atol=0)
    for name in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(resumed.state[target][name], reference.state[reference_param][name], rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["master_weight", "exp_avg", "exp_avg_sq", "step", "all"])
def test_dcp_rejects_missing_master_optimizer_state(tmp_path, missing):
    fields = {
        "master_weight": torch.tensor([2.0]),
        "exp_avg": torch.zeros(1),
        "exp_avg_sq": torch.zeros(1),
        "step": torch.tensor(1.0),
    }
    saved = {k: v for k, v in fields.items() if k != missing} if missing != "all" else {}
    dcp.save(
        {"optimizer": {"state": {"weight": saved}}, "model": {"weight": torch.tensor([2.0])}}, checkpoint_id=tmp_path
    )
    target = {"optimizer": {"state": {"weight": copy.deepcopy(fields)}}, "model": {"weight": torch.tensor([9.0])}}
    with pytest.raises(CheckpointException, match="precision-sensitive optimizer state"):
        dcp.load(
            target,
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(strict_model=True, strict_optimizer_dtype=True),
        )


def test_master_optimizer_rejects_implicit_precision_migration():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    source = FullFP32AdamW([p])
    p.grad = torch.ones_like(p)
    source.step()
    with pytest.raises(RuntimeError, match="migration"):
        MasterFP32AdamW([p]).load_state_dict(source.state_dict())


def test_optimizer_state_wrapper_preserves_strict_optimizer_policy():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = CPUOffloadAdamW([p], lr=0.001)
    wrapper = SimpleNamespace(optimizer=optimizer)
    assert _requires_exact_optimizer_state_dtypes(wrapper)
    assert not _requires_exact_optimizer_recipe(wrapper)
    assert _requires_exact_optimizer_recipe(SimpleNamespace(optimizer=SimpleNamespace(requires_exact_recipe=True)))


@pytest.mark.parametrize("field", ["master_weight", "exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("saved_shape", [(2,), (4,), (1, 3)])
def test_dcp_rejects_optimizer_shape_mismatch_before_read(tmp_path, field, saved_shape):
    fields = {name: torch.zeros(3) for name in ("master_weight", "exp_avg", "exp_avg_sq")}
    fields["step"] = torch.tensor(1.0)
    saved = copy.deepcopy(fields)
    saved[field] = torch.ones(saved_shape)
    dcp.save({"optimizer": {"state": {"weight": saved}}}, checkpoint_id=tmp_path)
    target = {"optimizer": {"state": {"weight": copy.deepcopy(fields)}}}
    before = copy.deepcopy(target)
    with pytest.raises(CheckpointException, match="optimizer shape mismatch"):
        dcp.load(
            target,
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True),
        )
    for name, tensor in target["optimizer"]["state"]["weight"].items():
        torch.testing.assert_close(tensor, before["optimizer"]["state"]["weight"][name], rtol=0, atol=0)


@pytest.mark.parametrize("partial_target", [False, True])
@pytest.mark.parametrize("container", [False, True])
def test_dcp_validates_optimizer_inventory_with_partial_target(tmp_path, partial_target, container):
    model = torch.nn.Linear(3, 2).bfloat16()
    opt = CPUOffloadAdamW(model.parameters(), lr=0.001)
    # Persist complete state, then resume into either a fresh or partially initialized optimizer.
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    opt.step()
    opt.zero_grad(set_to_none=True)
    state = OptimizerState(model, opt, parallel_state=SimpleNamespace(dp_mode="ddp"))
    dcp.save({"optimizer": state}, checkpoint_id=tmp_path)
    target_model = torch.nn.Linear(3, 2).bfloat16()
    target_opt = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
    if partial_target:
        target_model.weight.grad = torch.ones_like(target_model.weight)
        target_opt.step()
        target_opt.zero_grad(set_to_none=True)
    inventory = SimpleNamespace(optimizers_dict={"precise": target_opt}) if container else target_opt
    target_state = OptimizerState(target_model, target_opt, parallel_state=SimpleNamespace(dp_mode="ddp"), load=True)
    planner = _ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=inventory)
    if partial_target:
        with pytest.raises(CheckpointException, match="Incomplete precision-sensitive optimizer target state"):
            dcp.load({"optimizer": target_state}, checkpoint_id=tmp_path, planner=planner)
    else:
        dcp.load({"optimizer": target_state}, checkpoint_id=tmp_path, planner=planner)
        for param, target in zip(model.parameters(), target_model.parameters(), strict=True):
            for key in ("master_weight", "exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(opt.state[param][key], target_opt.state[target][key], rtol=0, atol=0)


def test_dcp_keeps_partial_ordinary_adamw_state_supported(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.weight.grad = torch.ones_like(model.weight)
    opt.step()
    opt.zero_grad(set_to_none=True)
    dcp.save(
        {"optimizer": OptimizerState(model, opt, parallel_state=SimpleNamespace(dp_mode="ddp"))},
        checkpoint_id=tmp_path,
    )
    target = torch.nn.Linear(3, 2)
    resumed = torch.optim.AdamW(target.parameters(), lr=0.001)
    target.weight.grad = torch.ones_like(target.weight)
    resumed.step()
    resumed.zero_grad(set_to_none=True)
    dcp.load(
        {"optimizer": OptimizerState(target, resumed, parallel_state=SimpleNamespace(dp_mode="ddp"), load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(strict_model=False, optimizer=resumed),
    )
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(opt.state[model.weight][key], resumed.state[target.weight][key], rtol=0, atol=0)
    assert target.bias not in resumed.state


def test_dcp_keeps_default_state_for_unstepped_parameter(tmp_path):
    fields = {
        "master_weight": torch.tensor([2.0]),
        "exp_avg": torch.tensor([0.5]),
        "exp_avg_sq": torch.tensor([0.25]),
        "step": torch.tensor(3.0),
    }
    dcp.save({"optimizer": {"state": {"weight": copy.deepcopy(fields)}}}, checkpoint_id=tmp_path)
    target_weight = copy.deepcopy(fields)
    target_bias = {name: torch.zeros_like(tensor) for name, tensor in fields.items()}
    target = {"optimizer": {"state": {"weight": target_weight, "bias": target_bias}}}

    weight = torch.nn.Parameter(torch.zeros(1))
    bias = torch.nn.Parameter(torch.zeros(1))
    optimizer = SimpleNamespace(
        requires_exact_state_dtypes=True,
        param_groups=[{"params": [weight, bias]}],
        state={weight: copy.deepcopy(fields), bias: copy.deepcopy(target_bias)},
    )
    dcp.load(
        target,
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=optimizer),
    )
    for name in fields:
        torch.testing.assert_close(target["optimizer"]["state"]["weight"][name], fields[name], rtol=0, atol=0)
        torch.testing.assert_close(
            target["optimizer"]["state"]["bias"][name], torch.zeros_like(fields[name]), rtol=0, atol=0
        )


def test_dcp_drops_wholly_absent_optimizer_state_after_resume(tmp_path):
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    source_model = torch.nn.Linear(1, 1).bfloat16()
    source_optimizer = CPUOffloadAdamW(source_model.parameters(), lr=0.001)
    source_model.weight.grad = torch.ones_like(source_model.weight)
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)
    assert source_model.bias not in source_optimizer.state
    parallel_state = SimpleNamespace(dp_mode="ddp")
    dcp.save(
        {"optimizer": OptimizerState(source_model, source_optimizer, parallel_state=parallel_state)},
        checkpoint_id=tmp_path,
    )

    target_model = torch.nn.Linear(1, 1).bfloat16()
    target_optimizer = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
    # Simulate a fused resume that materializes DCP placeholders before the
    # restored model value is available. A stale master/step would survive.
    get_optimizer_state_dict(target_model, target_optimizer)
    with torch.no_grad():
        target_model.bias.fill_(7.0)
    dcp.load(
        {"optimizer": OptimizerState(target_model, target_optimizer, parallel_state=parallel_state, load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=target_optimizer),
    )
    assert target_model.bias not in target_optimizer.state
    assert target_optimizer.state[target_model.weight]["step"].item() == 1

    target_model.weight.grad = torch.zeros_like(target_model.weight)
    target_model.bias.grad = torch.zeros_like(target_model.bias)
    target_optimizer.step()
    bias_state = target_optimizer.state[target_model.bias]
    assert bias_state["step"].item() == 1
    torch.testing.assert_close(bias_state["master_weight"], torch.full_like(bias_state["master_weight"], 7.0))
    assert target_optimizer.state[target_model.weight]["step"].item() == 2


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("multi", [False, True])
def test_dcp_load_hooks_observe_saved_state_and_keep_their_changes(tmp_path, empty, multi):
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    from veomni.optim.optimizer import MultiOptimizer

    source_model = torch.nn.Linear(1, 1).bfloat16()
    source = CPUOffloadAdamW(source_model.parameters(), lr=0.001)
    if not empty:
        source_model.weight.grad = torch.ones_like(source_model.weight)
        source.step()
    source_container = MultiOptimizer(source_model, {"all": source}, ["all"]) if multi else source
    parallel = SimpleNamespace(dp_mode="ddp")
    if empty and not multi:
        # Exercise an explicitly empty checkpoint without asking Torch DCP's
        # state-dict producer to synthesize optimizer state for every parameter.
        saved = source.state_dict()
        saved["param_groups"][0]["params"] = ["weight", "bias"]
        saved[_ZERO_OPTIMIZER_STATE_MARKER] = True
    else:
        saved = OptimizerState(source_model, source_container, parallel_state=parallel)
    dcp.save({"optimizer": saved}, checkpoint_id=tmp_path)

    target_model = torch.nn.Linear(1, 1).bfloat16()
    target = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
    target_container = MultiOptimizer(target_model, {"all": target}, ["all"]) if multi else target
    get_optimizer_state_dict(target_model, target)
    events = []

    def before_load(optimizer, incoming):
        assert set(incoming["state"]) == (set() if empty else {"weight"})
        events.append("before")

    def after_load(optimizer):
        assert target_model.bias not in optimizer.state
        optimizer.state[target_model.bias] = {
            "step": torch.tensor(7.0),
            "master_weight": target_model.bias.detach().float().clone(),
            "exp_avg": torch.full_like(target_model.bias, 0.75, dtype=torch.float32),
            "exp_avg_sq": torch.zeros_like(target_model.bias, dtype=torch.float32),
        }
        events.append("after")

    target.register_load_state_dict_pre_hook(before_load)
    target.register_load_state_dict_post_hook(after_load)
    dcp.load(
        {"optimizer": OptimizerState(target_model, target_container, parallel_state=parallel, load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=target_container),
    )
    assert events == ["before", "after"]
    assert target.state[target_model.bias]["step"].item() == 7
    assert target.state[target_model.bias]["exp_avg"].item() == 0.75


def test_state_receipt_ignores_unassigned_metadata_after_resume():
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    source = CPUOffloadAdamW([param])
    param.grad = torch.ones_like(param)
    source.step()
    saved = copy.deepcopy(source.state_dict())
    saved["state"]["custom_metadata"] = {"value": 1}
    clone = torch.nn.Parameter(param.detach().clone())
    target = CPUOffloadAdamW([clone])
    target.load_state_dict(saved)
    clone.grad = torch.ones_like(clone)
    target.step()
    assert target.state["custom_metadata"] == {"value": 1}
    model = torch.nn.Module()
    model.register_parameter("weight", clone)
    with pytest.raises(RuntimeError, match="DCP cannot serialize non-parameter optimizer state"):
        OptimizerState(model, target, parallel_state=SimpleNamespace(dp_mode="ddp")).state_dict()

    called = []

    def save_metadata_separately(optimizer):
        called.append(optimizer.state.pop("custom_metadata"))

    target.register_state_dict_pre_hook(save_metadata_separately)
    exported = OptimizerState(model, target, parallel_state=SimpleNamespace(dp_mode="ddp")).state_dict()
    assert called == [{"value": 1}]
    assert "weight" in exported["state"]


def test_dcp_drops_wholly_absent_state_from_ddp_wrapped_model(tmp_path):
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
    from torch.nn.parallel import DistributedDataParallel

    if dist.is_initialized():
        pytest.skip("requires ownership of the default process group")

    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'ddp-init'}",
        rank=0,
        world_size=1,
    )
    try:
        parallel_state = SimpleNamespace(dp_mode="ddp")
        source_model = DistributedDataParallel(torch.nn.Linear(1, 1).bfloat16())
        source_optimizer = CPUOffloadAdamW(source_model.parameters(), lr=0.001)
        source_model.module.weight.grad = torch.ones_like(source_model.module.weight)
        source_optimizer.step()
        source_optimizer.zero_grad(set_to_none=True)
        assert source_model.module.bias not in source_optimizer.state
        dcp.save(
            {"optimizer": OptimizerState(source_model, source_optimizer, parallel_state=parallel_state)},
            checkpoint_id=tmp_path / "checkpoint",
        )

        target_model = DistributedDataParallel(torch.nn.Linear(1, 1).bfloat16())
        target_optimizer = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
        get_optimizer_state_dict(target_model, target_optimizer)
        with torch.no_grad():
            target_model.module.bias.fill_(7.0)
        dcp.load(
            {
                "optimizer": OptimizerState(
                    target_model,
                    target_optimizer,
                    parallel_state=parallel_state,
                    load=True,
                )
            },
            checkpoint_id=tmp_path / "checkpoint",
            planner=_ModelStrictLoadPlanner(
                strict_model=False,
                strict_optimizer_dtype=True,
                optimizer=target_optimizer,
            ),
        )
        assert target_model.module.bias not in target_optimizer.state
        assert target_optimizer.state[target_model.module.weight]["step"].item() == 1

        target_model.module.weight.grad = torch.zeros_like(target_model.module.weight)
        target_model.module.bias.grad = torch.zeros_like(target_model.module.bias)
        target_optimizer.step()
        bias_state = target_optimizer.state[target_model.module.bias]
        assert bias_state["step"].item() == 1
        torch.testing.assert_close(bias_state["master_weight"], torch.full_like(bias_state["master_weight"], 7.0))
        assert target_optimizer.state[target_model.module.weight]["step"].item() == 2
    finally:
        dist.destroy_process_group()


def test_dcp_rejects_adamw_multi_optimizer_as_muon(tmp_path):
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = torch.nn.Linear(4, 2, bias=False)

    parallel_state = SimpleNamespace(dp_mode="ddp")
    source_model = Model().bfloat16()
    source_child = CPUOffloadAdamW([{"params": [source_model.down_proj.weight]}], lr=0.001)
    source = MultiOptimizer(source_model, {"all": source_child}, ["all"])
    source_model.down_proj.weight.grad = torch.ones_like(source_model.down_proj.weight)
    source.step()
    source.zero_grad()
    dcp.save(
        {"optimizer": OptimizerState(source_model, source, parallel_state=parallel_state)},
        checkpoint_id=tmp_path,
    )

    target_model = Model().bfloat16()
    groups = add_qwen38_plans(target_model, [{"params": [target_model.down_proj.weight], "cpu_offload": False}])
    target_child = Qwen38MuonAdamW(groups, lr=0.001)
    target = MultiOptimizer(target_model, {"all": target_child}, ["all"])
    with pytest.raises(CheckpointException, match="recipe/state is incomplete"):
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=parallel_state, load=True)},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(
                strict_model=False,
                strict_optimizer_dtype=True,
                strict_optimizer_recipe=True,
                optimizer=target,
            ),
        )


def test_dcp_zero_step_sparse_multi_optimizer_restores_lazy_state(tmp_path):
    from torch.distributed.checkpoint import FileSystemReader

    from veomni.optim.optimizer import MultiOptimizer

    model = torch.nn.Linear(3, 2).bfloat16()
    child = CPUOffloadAdamW(model.parameters(), lr=0.001)
    optimizer = MultiOptimizer(model, {"all": child}, ["all"])
    parallel_state = SimpleNamespace(dp_mode="ddp")
    dcp.save(
        {"optimizer": OptimizerState(model, optimizer, parallel_state=parallel_state)},
        checkpoint_id=tmp_path,
    )
    metadata = FileSystemReader(tmp_path).read_metadata()
    assert "optimizer.__veomni_zero_optimizer_state__" in metadata.state_dict_metadata

    target_model = torch.nn.Linear(3, 2).bfloat16()
    target_child = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
    target = MultiOptimizer(target_model, {"all": target_child}, ["all"])
    dcp.load(
        {"optimizer": OptimizerState(target_model, target, parallel_state=parallel_state, load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=target),
    )
    assert not target_child.state
    assert not target._veomni_resume_zero_optimizer_state


def test_dcp_rejects_zero_state_marker_with_tensor_state_and_clears_stale_flag(tmp_path):
    from veomni.optim.optimizer import MultiOptimizer

    model = torch.nn.Linear(3, 2).bfloat16()
    child = CPUOffloadAdamW(model.parameters(), lr=0.001)
    optimizer = MultiOptimizer(model, {"all": child}, ["all"])
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    parallel_state = SimpleNamespace(dp_mode="ddp")
    state_dict = OptimizerState(model, optimizer, parallel_state=parallel_state).state_dict()
    assert any(key.startswith("state.") for key in state_dict)
    state_dict[_ZERO_OPTIMIZER_STATE_MARKER] = True
    dcp.save({"optimizer": state_dict}, checkpoint_id=tmp_path)

    target_model = torch.nn.Linear(3, 2).bfloat16()
    target_child = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
    target = MultiOptimizer(target_model, {"all": target_child}, ["all"])
    target._veomni_resume_zero_optimizer_state = True
    with pytest.raises(CheckpointException, match="zero-state optimizer marker conflicts"):
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=parallel_state, load=True)},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=target),
        )
    assert not target._veomni_resume_zero_optimizer_state


def test_dcp_zero_step_exact_recipe_multi_optimizer_restores_lazy_state(tmp_path):
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = torch.nn.Linear(4, 2, bias=False)

    def make_optimizer(model):
        groups = add_qwen38_plans(model, [{"params": [model.down_proj.weight], "cpu_offload": True}])
        child = Qwen38MuonAdamW(groups, lr=0.001)
        return child, MultiOptimizer(model, {"all": child}, ["all"])

    parallel_state = SimpleNamespace(dp_mode="ddp")
    model = Model().bfloat16()
    _, optimizer = make_optimizer(model)
    dcp.save(
        {"optimizer": OptimizerState(model, optimizer, parallel_state=parallel_state)},
        checkpoint_id=tmp_path,
    )

    target_model = Model().bfloat16()
    target_child, target = make_optimizer(target_model)
    dcp.load(
        {"optimizer": OptimizerState(target_model, target, parallel_state=parallel_state, load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(
            strict_model=False,
            strict_optimizer_dtype=True,
            strict_optimizer_recipe=True,
            optimizer=target,
        ),
    )
    assert not target_child.state
    assert not target._veomni_resume_zero_optimizer_state


def _extra_parallel_resume_worker(rank, world, rendezvous, hsdp=False):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        if hsdp:
            mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("replicate", "fsdp", "ep"))
            replicate, fsdp, ep = mesh.get_coordinate()
            state_mesh = mesh["replicate", "fsdp"]
            state_placements = [Replicate(), Shard(0)]
        else:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("fsdp", "ep"))
            fsdp, ep = mesh.get_coordinate()
            state_mesh = mesh["fsdp"]
            state_placements = [Shard(0)]
        # Deliberately uneven FSDP/EP shards exercise DTensor metadata recovery.
        global_shape = (5, 7)
        full = torch.arange(global_shape[0] * global_shape[1], dtype=torch.float32).reshape(global_shape)
        local = torch.tensor_split(torch.tensor_split(full, 2, dim=0)[fsdp], 2, dim=1)[ep].clone()
        param = torch.nn.Parameter(
            DTensor.from_local(
                local,
                mesh,
                [Replicate(), Shard(0), Shard(1)] if hsdp else [Shard(0), Shard(1)],
                run_check=False,
                shape=global_shape,
                stride=(global_shape[1], 1),
            )
        )
        state_tensors = {
            name: DTensor.from_local(local.clone() + index, state_mesh, state_placements, run_check=False)
            for index, name in enumerate(("master_weight", "exp_avg", "exp_avg_sq"), start=1)
        }
        optimizer = CPUOffloadAdamW([{"params": [param], "cpu_offload": True}], lr=0.001)
        optimizer.state[param] = {"step": torch.tensor(2.0), **state_tensors}
        saved = copy.deepcopy(optimizer.state_dict())

        resumed = CPUOffloadAdamW([{"params": [param], "cpu_offload": True}], lr=0.001)
        resumed.load_state_dict(saved)
        for name, expected in state_tensors.items():
            actual = resumed.state[param][name]
            assert actual.device_mesh == param.device_mesh
            assert actual.placements == param.placements
            torch.testing.assert_close(actual.to_local(), expected.to_local(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _subgroup_mesh_worker(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

    from veomni.checkpoint.dcp_checkpointer import _extra_parallel_first_mesh

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=8, timeout=timedelta(seconds=20)
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("replicate", "fsdp", "ep"))
        replacement = [None, None, None]
        for axis in range(3):
            for first in range(2):
                for second in range(2):
                    if axis == 0:
                        ranks = [replica * 4 + first * 2 + second for replica in range(2)]
                    elif axis == 1:
                        ranks = [first * 4 + shard * 2 + second for shard in range(2)]
                    else:
                        ranks = [first * 4 + second * 2 + expert for expert in range(2)]
                    group = dist.new_group(ranks)
                    if rank in ranks:
                        replacement[axis] = group
        alternate = DeviceMesh.from_group(replacement, "cpu", mesh=mesh.mesh, mesh_dim_names=mesh.mesh_dim_names)
        # Only the first complete HSDP replica constructs the reordered view.
        # Its process groups already exist; no world-wide collective may occur.
        if rank < 4:
            for source in (mesh, alternate):
                reordered = _extra_parallel_first_mesh(source)
                assert _extra_parallel_first_mesh(source) is reordered
                assert reordered.mesh_dim_names == ("replicate", "ep", "fsdp")
                assert torch.equal(reordered.mesh, source.mesh.permute(0, 2, 1))
                for new_dim, old_dim in enumerate((0, 2, 1)):
                    assert reordered.get_group(new_dim) is source.get_group(old_dim)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=8, device="cpu"))
def test_hsdp_subgroup_reordered_mesh_reuses_existing_groups(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _subgroup_mesh_worker, args=(str(tmp_path / "rdzv-subgroup"),), nprocs=8, join=True, start_method="fork"
    )


@pytest.mark.parametrize("hsdp", [False, True])
@pytest.mark.launch_plan(provider=lambda context: _extra_parallel_launch(context.params["hsdp"]))
def test_extra_parallel_fsdp_state_resume_restores_parameter_mesh(tmp_path, hsdp):
    import torch.multiprocessing as mp

    mp.start_processes(
        _extra_parallel_resume_worker,
        args=(_extra_parallel_launch(hsdp).nproc, str(tmp_path / "rdzv"), hsdp),
        nprocs=_extra_parallel_launch(hsdp).nproc,
        join=True,
        start_method="fork",
    )


def _restore_fsdp_only_extra_parallel_worker(rank, world, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    from veomni.checkpoint.dcp_checkpointer import drop_extra_parallel_dim, restore_extra_parallel_dim

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        fsdp_rank, ep_rank = mesh.get_coordinate()
        fsdp_mesh = mesh["ep_fsdp"]
        # The EP extent is even globally, but each FSDP-only DTensor exposes
        # only its EP-local extent, so the original shape cannot be inferred.
        full = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
        ep_local = torch.tensor_split(full, 2, dim=0)[ep_rank]
        local = torch.tensor_split(ep_local, 2, dim=1)[fsdp_rank].clone()
        fsdp_only = DTensor.from_local(
            local,
            fsdp_mesh,
            [Shard(1)],
            run_check=False,
            shape=ep_local.shape,
            stride=(ep_local.shape[1], 1),
        )

        with pytest.raises(RuntimeError, match="pass the original global_shape"):
            restore_extra_parallel_dim(fsdp_only, mesh, fsdp_mesh)

        restored = restore_extra_parallel_dim(fsdp_only, mesh, fsdp_mesh, global_shape=full.shape)
        assert restored.device_mesh == mesh
        assert tuple(restored.placements) == (Shard(1), Shard(0))
        assert restored.shape == full.shape
        torch.testing.assert_close(restored.to_local(), local, rtol=0, atol=0)

        full_layout = DTensor.from_local(
            local,
            mesh,
            [Shard(1), Shard(0)],
            run_check=False,
            shape=full.shape,
            stride=(full.shape[1], 1),
        )
        restored_full = restore_extra_parallel_dim(full_layout, mesh, fsdp_mesh, global_shape=full.shape)
        assert restored_full.shape == full.shape
        torch.testing.assert_close(restored_full.to_local(), local, rtol=0, atol=0)

        # When EP and FSDP shard the same tensor dim, global checkpoint metadata
        # must follow EP-outer / FSDP-inner physical storage. A same-config
        # round-trip alone can miss a block-transposed global layout, so rebuild
        # the complete tensor and compare its values.
        same_dim_full = torch.arange(6, dtype=torch.float32)
        same_dim_ep_local = torch.tensor_split(same_dim_full, 2, dim=0)[ep_rank]
        same_dim_local = torch.tensor_split(same_dim_ep_local, 2, dim=0)[fsdp_rank].clone()
        same_dim_fsdp_only = DTensor.from_local(
            same_dim_local,
            fsdp_mesh,
            [Shard(0)],
            run_check=False,
            shape=same_dim_ep_local.shape,
            stride=(1,),
        )
        same_dim_restored = restore_extra_parallel_dim(
            same_dim_fsdp_only,
            mesh,
            fsdp_mesh,
            ep_shard_dim=0,
            fsdp_shard_dim=0,
            global_shape=same_dim_full.shape,
        )
        assert same_dim_restored.device_mesh.mesh_dim_names == ("ep", "ep_fsdp")
        torch.testing.assert_close(same_dim_restored.to_local(), same_dim_local, rtol=0, atol=0)
        torch.testing.assert_close(same_dim_restored.full_tensor(), same_dim_full, rtol=0, atol=0)

        # A 6-element extent on a 2x2 mesh is the smallest case where flat and
        # EP-outer / FSDP-inner chunking differ: [2, 1, 2, 1] versus
        # [2, 2, 1, 1]. The drop must reconstruct the parameter mesh and retain
        # this rank's physical local shard without rejecting uneven ranks.
        same_dim_dropped = drop_extra_parallel_dim(
            same_dim_restored,
            fsdp_mesh,
            fsdp_shard_dim=0,
            extra_parallel_name="ep",
        )
        assert same_dim_dropped.device_mesh == fsdp_mesh
        assert tuple(same_dim_dropped.placements) == (Shard(0),)
        assert same_dim_dropped.shape == (same_dim_full.shape[0] // 2,)
        torch.testing.assert_close(same_dim_dropped.to_local(), same_dim_local, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_LAYOUT_LAUNCH)
def test_restore_fsdp_only_extra_parallel_uses_original_global_shape(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _restore_fsdp_only_extra_parallel_worker,
        args=(_CI_LAYOUT_LAUNCH.nproc, str(tmp_path / "rdzv")),
        nprocs=_CI_LAYOUT_LAUNCH.nproc,
        join=True,
        start_method="fork",
    )


def _extra_parallel_dcp_resume_worker(rank, world, rendezvous, checkpoint, hsdp, optimizer_kind, same_dim=False):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    from veomni.distributed.parallel_plan import SpecInfo
    from veomni.optim.master_fp32_adamw import FullFP32AdamW
    from veomni.optim.optimizer import MultiOptimizer

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        if hsdp:
            mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("replicate", "fsdp", "ep"))
            placements = [Replicate(), Shard(0), Shard(0)] if same_dim else [Replicate(), Shard(1), Shard(0)]
        else:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("fsdp", "ep"))
            placements = [Shard(0), Shard(0)] if same_dim else [Shard(1), Shard(0)]
        torch.manual_seed(17)
        full = torch.randn(5, 7, dtype=torch.bfloat16)
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(distribute_tensor(full, mesh, placements))
        model._fqn2spec_info = {
            "weight": SpecInfo(
                para_name="ep",
                placement=Shard(0),
                fqn="weight",
                para_fsdp_mesh=mesh,
                fsdp_shard_dim=0 if same_dim else 1,
            )
        }
        parallel_state = SimpleNamespace(
            dp_mode="fsdp2",
            any_extra_parallel_enabled=True,
            extra_parallel_names=("ep",),
            extra_parallel_fsdp_device_mesh={"ep": mesh},
        )

        def make_optimizer(model, gradient=None):
            if optimizer_kind == "cpu_offload_adamw":
                child = CPUOffloadAdamW([{"params": [model.weight], "cpu_offload": True}], lr=0.001)
            else:
                child = FullFP32AdamW([model.weight], lr=0.001)
            optimizer = MultiOptimizer(model, {"ep": child}, ["ep"])
            if gradient is not None:
                model.weight.grad = distribute_tensor(gradient, mesh, placements)
                optimizer.step()
                model.weight.grad = None
            return child, optimizer

        source_child, source = make_optimizer(model, torch.randn_like(full))
        dcp.save(
            {"optimizer": OptimizerState(model, source, parallel_state=parallel_state)},
            checkpoint_id=checkpoint,
        )
        dist.barrier()

        target_model = torch.nn.Module()
        target_model.weight = torch.nn.Parameter(distribute_tensor(torch.zeros_like(full), mesh, placements))
        target_model._fqn2spec_info = model._fqn2spec_info
        target_child, target = make_optimizer(target_model)
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=parallel_state, load=True)},
            checkpoint_id=checkpoint,
            planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=target),
        )
        for name in ("master_weight", "exp_avg", "exp_avg_sq"):
            expected = source_child.state[model.weight][name]
            actual = target_child.state[target_model.weight][name]
            assert actual.device_mesh == target_model.weight.device_mesh
            assert actual.placements == target_model.weight.placements
            assert actual.shape == target_model.weight.shape
            torch.testing.assert_close(actual.to_local(), expected.to_local(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("optimizer_kind", ["cpu_offload_adamw", "full_fp32_adamw"])
@pytest.mark.parametrize("hsdp", [False, True])
@pytest.mark.parametrize("same_dim", [False, True])
@pytest.mark.launch_plan(provider=lambda context: _extra_parallel_launch(context.params["hsdp"]))
def test_extra_parallel_dcp_resume_restores_parameter_layout(tmp_path, hsdp, optimizer_kind, same_dim):
    import torch.multiprocessing as mp

    world = _extra_parallel_launch(hsdp).nproc
    mp.start_processes(
        _extra_parallel_dcp_resume_worker,
        args=(world, str(tmp_path / "rdzv"), str(tmp_path / "checkpoint"), hsdp, optimizer_kind, same_dim),
        nprocs=world,
        join=True,
        start_method="fork",
    )


@pytest.mark.parametrize("saved_field", ["momentum_buffer", "unknown", "custom.nested"])
def test_dcp_rejects_incompatible_saved_fields_instead_of_treating_parameter_as_lazy(tmp_path, saved_field):
    dcp.save({"optimizer": {"state": {"weight": {saved_field: torch.ones(1)}}}}, checkpoint_id=tmp_path)
    fields = {
        "master_weight": torch.ones(1),
        "exp_avg": torch.zeros(1),
        "exp_avg_sq": torch.zeros(1),
        "step": torch.tensor(0.0),
    }
    target = {"optimizer": {"state": {"weight": fields}}}
    with pytest.raises(CheckpointException, match="missing part of precision-sensitive optimizer state"):
        dcp.load(
            target,
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True),
        )


@pytest.mark.parametrize("scope", ["model", "optimizer"])
@pytest.mark.parametrize("strict_model", [False, True])
def test_dcp_rejects_unversioned_same_dimension_layout_before_read(tmp_path, scope, strict_model):
    from veomni.checkpoint.dcp_checkpointer import _EXTRA_PARALLEL_LAYOUT_MARKER

    dcp.save({scope: {"weight": torch.ones(4)}}, checkpoint_id=tmp_path)
    untouched = torch.full((4,), 9.0)
    target = {scope: {"weight": untouched, _EXTRA_PARALLEL_LAYOUT_MARKER: True}}
    with pytest.raises(CheckpointException, match="layout migration"):
        dcp.load(target, checkpoint_id=tmp_path, planner=_ModelStrictLoadPlanner(strict_model=strict_model))
    torch.testing.assert_close(untouched, torch.full((4,), 9.0), rtol=0, atol=0)


def test_cpu_offload_state_receipt_survives_dcp_zero_lr_initialization(monkeypatch):
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    import veomni.optim.cpu_offload_adamw as module

    model = torch.nn.Linear(2, 2).bfloat16()
    optimizer = CPUOffloadAdamW([{"params": list(model.parameters()), "cpu_offload": True}], lr=0.001)
    messages = []
    monkeypatch.setattr(
        module.logger, "info", lambda message, *args: messages.append(message % args if args else message)
    )
    get_optimizer_state_dict(model, optimizer)
    assert optimizer.state and not optimizer._state_receipt_emitted
    assert not messages
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert optimizer._state_receipt_emitted
    assert sum("OPTIMIZER_STATE_RECEIPT" in message for message in messages) == 1
    assert sum("CPU_OPTIMIZER_STEP" in message for message in messages) == 1


def test_versioned_extra_parallel_checkpoint_exports_only_model_tensors(tmp_path, monkeypatch):
    import veomni.checkpoint.dcp_checkpointer as module

    weights = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    dcp.save({"model": {"weight": weights, module._EXTRA_PARALLEL_LAYOUT_MARKER: True}}, checkpoint_id=tmp_path)
    plan, size, keys = module._get_sharding_plan(tmp_path, shard_size=None)
    assert set(plan) == {"weight"} and keys == ["model.weight"]
    assert size == weights.numel() * weights.element_size()
    monkeypatch.setattr(module, "empty_cache", lambda: None)
    exported = module._process_shard(plan, str(tmp_path))
    assert set(exported) == {"weight"}
    torch.testing.assert_close(exported["weight"], weights, rtol=0, atol=0)
