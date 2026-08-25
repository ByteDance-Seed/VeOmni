import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from veomni.distributed.torch_compile import CompileConfig
from veomni.ops.kernels import moe as moe_kernels
from veomni.trainer import base as trainer_base


def _unexpected_triton_kernel(*args, **kwargs):
    raise AssertionError("The CPU dispatch tests must not execute a Triton kernel.")


# Load the common dispatch and public Triton boundary with narrow CPU substitutes
# for unavailable Triton kernel modules. The production dispatch, planner,
# executor, token permutation, and collectives remain unchanged.
_MOE_PACKAGE_NAME = "veomni.distributed.moe"
_MOE_LAYER_NAME = f"{_MOE_PACKAGE_NAME}.moe_layer"
_EP_LOAD_BALANCE_NAME = f"{_MOE_PACKAGE_NAME}.ep_load_balance"
_GROUP_GEMM_KERNEL_NAME = "veomni.ops.kernels.moe._kernels.kernel.group_gemm"
_MOE_KERNEL_NAME = "veomni.ops.kernels.moe._kernels.kernel.moe"
_SCATTER_NAME = "veomni.ops.kernels.moe._scatter"
_GROUP_GEMM_NAME = "veomni.ops.kernels.moe.group_gemm"
_MISSING_PARENT_ATTRIBUTE = object()
_saved_group_gemm_parent_attribute = getattr(moe_kernels, "group_gemm", _MISSING_PARENT_ATTRIBUTE)
_saved_modules = {
    name: sys.modules.get(name)
    for name in (
        _MOE_PACKAGE_NAME,
        _MOE_LAYER_NAME,
        _EP_LOAD_BALANCE_NAME,
        _GROUP_GEMM_KERNEL_NAME,
        _MOE_KERNEL_NAME,
        _SCATTER_NAME,
        _GROUP_GEMM_NAME,
    )
}
_moe_package = types.ModuleType(_MOE_PACKAGE_NAME)
_moe_package.__path__ = [str(Path(__file__).parents[2] / "veomni" / "distributed" / "moe")]
sys.modules[_MOE_PACKAGE_NAME] = _moe_package

_group_gemm_kernel = types.ModuleType(_GROUP_GEMM_KERNEL_NAME)
_group_gemm_kernel.group_gemm_same_mn = _unexpected_triton_kernel
_group_gemm_kernel.group_gemm_same_nk = _unexpected_triton_kernel
sys.modules[_GROUP_GEMM_KERNEL_NAME] = _group_gemm_kernel
_moe_kernel = types.ModuleType(_MOE_KERNEL_NAME)
_moe_kernel.expert_histogram = _unexpected_triton_kernel
_moe_kernel.moe_gather = _unexpected_triton_kernel
_moe_kernel.moe_scatter = _unexpected_triton_kernel
sys.modules[_MOE_KERNEL_NAME] = _moe_kernel
_scatter = types.ModuleType(_SCATTER_NAME)
_scatter.compute_expert_scatter_index = _unexpected_triton_kernel
sys.modules[_SCATTER_NAME] = _scatter

moe_layer = importlib.import_module(f"{_MOE_PACKAGE_NAME}.moe_layer")
ep_load_balance = importlib.import_module(f"{_MOE_PACKAGE_NAME}.ep_load_balance")
for _name in (
    "EPGroupGemm",
    "EPMergedFc1GroupGemm",
    "dispatch_to_ep_class",
):
    setattr(_moe_package, _name, getattr(moe_layer, _name))
group_gemm = importlib.import_module(_GROUP_GEMM_NAME)

for _name, _saved in _saved_modules.items():
    if _saved is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _saved
if _saved_group_gemm_parent_attribute is _MISSING_PARENT_ATTRIBUTE:
    delattr(moe_kernels, "group_gemm")
else:
    moe_kernels.group_gemm = _saved_group_gemm_parent_attribute

_import_seam_restored_without_stub_leaks = all(
    (name not in sys.modules if saved is None else sys.modules.get(name) is saved)
    for name, saved in _saved_modules.items()
)
_import_seam_parent_attribute_restored = (
    not hasattr(moe_kernels, "group_gemm")
    if _saved_group_gemm_parent_attribute is _MISSING_PARENT_ATTRIBUTE
    else getattr(moe_kernels, "group_gemm", _MISSING_PARENT_ATTRIBUTE) is _saved_group_gemm_parent_attribute
)


def test_cpu_import_seam_restores_loaded_moe_submodules():
    assert _import_seam_restored_without_stub_leaks
    assert _import_seam_parent_attribute_restored
    assert getattr(moe_kernels, "group_gemm", None) is not group_gemm


def test_balance_apis_are_reexported_from_common_moe_package():
    package_dir = Path(__file__).parents[2] / "veomni" / "distributed" / "moe"
    package_modules = (_MOE_PACKAGE_NAME, _MOE_LAYER_NAME, _EP_LOAD_BALANCE_NAME)
    saved_modules = {name: sys.modules.get(name) for name in package_modules}
    spec = importlib.util.spec_from_file_location(
        _MOE_PACKAGE_NAME,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None and spec.loader is not None
    moe_package = importlib.util.module_from_spec(spec)
    sys.modules[_MOE_PACKAGE_NAME] = moe_package
    sys.modules[_MOE_LAYER_NAME] = moe_layer
    sys.modules[_EP_LOAD_BALANCE_NAME] = ep_load_balance
    try:
        spec.loader.exec_module(moe_package)
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    for name in (
        "EPBalancePlan",
        "ExpertReplicaTransfer",
        "attach_qwen3_5_moe_ep_load_balancers",
        "cat_local_and_replica_weights",
    ):
        assert name in moe_package.__all__
        assert getattr(moe_package, name) is getattr(ep_load_balance, name)
    assert all(
        (name not in sys.modules if saved is None else sys.modules.get(name) is saved)
        for name, saved in saved_modules.items()
    )


def test_disabled_dispatch_preserves_the_existing_preprocess_path(monkeypatch):
    ep_group = object()
    selected_experts = torch.tensor([[0], [1]])
    routing_weights = torch.tensor([[0.25], [0.75]])
    hidden_states = torch.tensor([[2.0], [3.0]])
    calls = []

    monkeypatch.setattr(moe_layer, "get_parallel_state", lambda: SimpleNamespace(ep_group=ep_group))

    def fake_preprocess(*, expert_mask, num_experts, ep_group):
        calls.append(("preprocess", expert_mask.clone(), num_experts, ep_group))
        return (1, 1), (1, 1), torch.tensor([[1], [1]]), torch.tensor([2])

    def fake_pre_all2all(**kwargs):
        calls.append(("pre", kwargs.copy()))
        return hidden_states.clone(), torch.ones((2, 2)), torch.tensor([0, 1]), hidden_states.shape

    def fake_post_all2all(**kwargs):
        calls.append(("post", kwargs.copy()))
        return kwargs["expert_outputs"] + 1

    class FakeEPClass:
        @staticmethod
        def apply(tokens, cumsum, *args):
            calls.append(("apply", tokens.clone(), cumsum.clone(), args))
            return tokens * 2

    monkeypatch.setattr(moe_layer, "preprocess", fake_preprocess)
    monkeypatch.setattr(moe_layer, "token_pre_all2all", fake_pre_all2all)
    monkeypatch.setattr(moe_layer, "tokens_post_all2all", fake_post_all2all)

    output = moe_layer.dispatch_to_ep_class(
        FakeEPClass,
        2,
        routing_weights,
        selected_experts,
        hidden_states,
        "weight",
    )

    assert torch.equal(output, hidden_states * 2 + 1)
    assert [call[0] for call in calls] == ["preprocess", "pre", "apply", "post"]
    expected_mask = torch.nn.functional.one_hot(selected_experts, num_classes=2).permute(2, 1, 0)
    assert torch.equal(calls[0][1], expected_mask)
    assert calls[0][2:] == (2, ep_group)
    assert calls[2][3] == ("weight",)
    assert calls[3][1]["selected_experts"] is selected_experts
    assert calls[3][1]["routing_weights"] is routing_weights
    assert calls[3][1]["num_experts"] == 2


def test_opslot_adapter_adds_balancer_only_when_private_attribute_exists():
    received = []

    def raw_forward(**kwargs):
        received.append(kwargs)
        return kwargs

    adapter = moe_kernels._make_moe_experts_adapter(raw_forward)
    experts = SimpleNamespace(
        num_experts=2,
        gate_up_proj=torch.ones((1, 2, 1)),
        down_proj=torch.ones((1, 1, 1)),
    )
    hidden_states = torch.ones((1, 1))
    selected_experts = torch.zeros((1, 1), dtype=torch.long)
    routing_weights = torch.ones((1, 1))

    adapter(experts, hidden_states, selected_experts, routing_weights)
    assert "load_balancer" not in received[-1]
    assert set(received[-1]) == {
        "num_experts",
        "routing_weights",
        "selected_experts",
        "hidden_states",
        "fc1_1_weight",
        "fc1_2_weight",
        "fc2_weight",
        "fc1_1_2_weight",
        "swiglu_limit",
    }

    load_balancer = object()
    experts._veomni_ep_load_balancer = load_balancer
    adapter(experts, hidden_states, selected_experts, routing_weights)
    assert received[-1]["load_balancer"] is load_balancer


@pytest.mark.parametrize("ep_enabled", [False, True])
def test_public_group_gemm_rejects_balancing_outside_merged_ep_before_collectives(monkeypatch, ep_enabled):
    class UnexpectedBalancer:
        def build_plan(self, selected_experts):
            raise AssertionError("Invalid load-balancing calls must fail before planning or collectives.")

    monkeypatch.setattr(group_gemm, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=ep_enabled))
    merged_weight = None if ep_enabled else torch.ones((1, 2, 1))
    split_weight = torch.ones((1, 1, 1)) if ep_enabled else None

    with pytest.raises(ValueError, match="load balancing.*merged.*expert parallel"):
        group_gemm.group_gemm_fused_moe_forward(
            num_experts=2,
            routing_weights=torch.ones((1, 1)),
            selected_experts=torch.zeros((1, 1), dtype=torch.long),
            hidden_states=torch.ones((1, 1)),
            fc1_1_weight=split_weight,
            fc1_2_weight=split_weight,
            fc2_weight=torch.ones((1, 1, 1)),
            fc1_1_2_weight=merged_weight,
            load_balancer=UnexpectedBalancer(),
        )


def _merged_weights_case(case: str) -> tuple[torch.Tensor, torch.Tensor]:
    gate_up_proj = torch.ones((1, 4, 3), dtype=torch.float32)
    down_proj = torch.ones((1, 3, 2), dtype=torch.float32)
    if case == "gate_rank":
        gate_up_proj = torch.ones((4, 3))
    elif case == "down_rank":
        down_proj = torch.ones((3, 2))
    elif case == "gate_width_odd":
        gate_up_proj = torch.ones((1, 3, 3))
    elif case == "intermediate_mismatch":
        down_proj = torch.ones((1, 3, 1))
    elif case == "hidden_mismatch":
        down_proj = torch.ones((1, 4, 2))
    elif case == "expert_rows_mismatch":
        down_proj = torch.ones((2, 3, 2))
    elif case == "dtype_mismatch":
        down_proj = down_proj.double()
    elif case == "device_mismatch":
        gate_up_proj = torch.ones((1, 4, 3), device="meta")
    elif case == "layout_mismatch":
        gate_up_proj = torch.sparse_coo_tensor(
            torch.tensor([[0], [0], [0]]),
            torch.ones(1),
            size=(1, 4, 3),
            check_invariants=True,
        )
    else:
        raise AssertionError(f"Unknown merged-weight case: {case}")
    return gate_up_proj, down_proj


@pytest.mark.parametrize(
    "case",
    [
        "gate_rank",
        "down_rank",
        "gate_width_odd",
        "intermediate_mismatch",
        "hidden_mismatch",
        "expert_rows_mismatch",
        "dtype_mismatch",
        "device_mismatch",
        "layout_mismatch",
    ],
)
def test_public_group_gemm_validates_merged_weight_relationships_before_planning(monkeypatch, case):
    class UnexpectedBalancer:
        def build_plan(self, selected_experts):
            raise AssertionError("Invalid merged weights must fail before planner collectives.")

    def unexpected_dispatch(*args, load_balancer=None, **kwargs):
        return load_balancer.build_plan(args[3])

    gate_up_proj, down_proj = _merged_weights_case(case)
    monkeypatch.setattr(group_gemm, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True))
    monkeypatch.setattr(group_gemm, "dispatch_to_ep_class", unexpected_dispatch)

    with pytest.raises(ValueError, match="merged expert weights"):
        group_gemm.group_gemm_fused_moe_forward(
            num_experts=2,
            routing_weights=torch.ones((1, 1)),
            selected_experts=torch.zeros((1, 1), dtype=torch.long),
            hidden_states=torch.ones((1, 3)),
            fc1_1_weight=None,
            fc1_2_weight=None,
            fc2_weight=down_proj,
            fc1_1_2_weight=gate_up_proj,
            load_balancer=UnexpectedBalancer(),
        )


def test_public_group_gemm_accepts_compatible_merged_weight_boundary(monkeypatch):
    class PlannerReached(RuntimeError):
        pass

    class ExpectedBalancer:
        def build_plan(self, selected_experts):
            raise PlannerReached

    def planner_dispatch(*args, load_balancer=None, **kwargs):
        return load_balancer.build_plan(args[3])

    gate_up_proj, down_proj = (
        torch.ones((1, 4, 3), dtype=torch.bfloat16),
        torch.ones((1, 3, 2), dtype=torch.bfloat16),
    )
    monkeypatch.setattr(group_gemm, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True))
    monkeypatch.setattr(group_gemm, "dispatch_to_ep_class", planner_dispatch)

    with pytest.raises(PlannerReached):
        group_gemm.group_gemm_fused_moe_forward(
            num_experts=2,
            routing_weights=torch.ones((1, 1), dtype=torch.bfloat16),
            selected_experts=torch.zeros((1, 1), dtype=torch.long),
            hidden_states=torch.ones((1, 3), dtype=torch.bfloat16),
            fc1_1_weight=None,
            fc1_2_weight=None,
            fc2_weight=down_proj,
            fc1_1_2_weight=gate_up_proj,
            load_balancer=ExpectedBalancer(),
        )


def test_common_dispatch_validates_merged_weights_before_controller_plan(monkeypatch):
    ep_group = object()

    class UnexpectedBalancer:
        def __init__(self):
            self.ep_group = ep_group

        def build_plan(self, selected_experts):
            raise AssertionError("Invalid merged weights must fail before controller planning.")

    monkeypatch.setattr(moe_layer, "get_parallel_state", lambda: SimpleNamespace(ep_group=ep_group))

    with pytest.raises(ValueError, match="merged expert weights"):
        moe_layer.dispatch_to_ep_class(
            moe_layer.EPMergedFc1GroupGemm,
            2,
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones((1, 3)),
            torch.ones((1, 4, 3), dtype=torch.float32),
            torch.ones((1, 3, 2), dtype=torch.float64),
            None,
            load_balancer=UnexpectedBalancer(),
        )


class _RecordingBalancer:
    def __init__(self, ep_group):
        self.ep_group = ep_group
        self.layer_index = 7
        self.plan = None

    def build_plan(self, selected_experts):
        self.plan = ep_load_balance.build_ep_balance_plan(
            selected_experts,
            num_experts=2,
            ep_group=self.ep_group,
            max_replicas_per_rank=1,
        )
        return self.plan


class _BalanceMonitor:
    def __init__(self):
        self.records = []

    def record_ep_balance(self, *record):
        self.records.append(record)


def _cpu_merged_expert(tokens, cumsum, gate_up_proj, down_proj, swiglu_limit):
    assert swiglu_limit is None
    outputs = []
    start = 0
    for expert, end in enumerate(cumsum.tolist()):
        expert_tokens = tokens[start:end]
        outputs.append(expert_tokens * gate_up_proj[expert, 0, 0] * down_proj[expert, 0, 0])
        start = end
    return torch.cat(outputs, dim=0) if outputs else tokens.new_empty(tokens.shape)


def _exercise_two_rank_dispatch(rank: int, group: dist.ProcessGroup) -> None:
    from veomni.utils import moe_monitor

    selected_experts = torch.zeros((6 if rank == 0 else 4, 1), dtype=torch.long)
    routing_weights = torch.linspace(0.25, 1.0, selected_experts.shape[0]).reshape(-1, 1)
    routing_weights_before = routing_weights.clone()
    hidden_states = torch.arange(
        1 if rank == 0 else 7,
        7 if rank == 0 else 11,
        dtype=torch.float32,
    ).reshape(-1, 1)
    gate_up_proj = nn.Parameter(torch.tensor([[[2.0], [11.0]]] if rank == 0 else [[[5.0], [13.0]]]))
    down_proj = nn.Parameter(torch.tensor([[[3.0]]] if rank == 0 else [[[7.0]]]))
    balancer = _RecordingBalancer(group)
    monitor = _BalanceMonitor()
    events = []
    observed_cumsums = []

    original_start = ep_load_balance.ExpertReplicaTransfer.start
    original_all_to_all = moe_layer.all_to_all
    original_apply = moe_layer.EPMergedFc1GroupGemm.apply
    original_get_parallel_state = moe_layer.get_parallel_state
    original_group_gemm_get_parallel_state = group_gemm.get_parallel_state
    original_cat_weights = moe_layer.cat_local_and_replica_weights

    class TrackingTransfer:
        @staticmethod
        def start(local_weight, plan, ep_group):
            weight_name = "gate" if local_weight is gate_up_proj else "down"
            events.append(f"start_{weight_name}")
            transfer = original_start(local_weight, plan, ep_group)

            class TrackingWait:
                def wait(self):
                    events.append(f"wait_{weight_name}")
                    return transfer.wait()

            return TrackingWait()

    def tracking_all_to_all(*args, **kwargs):
        events.append("all_to_all")
        return original_all_to_all(*args, **kwargs)

    def cpu_apply(tokens, cumsum, *args):
        observed_cumsums.append(cumsum.detach().cpu().clone())
        return _cpu_merged_expert(tokens, cumsum, *args)

    def tracking_cat(local_weight, replica_weight, plan, ep_group):
        weight_name = "gate" if local_weight is gate_up_proj else "down"
        events.append(f"cat_{weight_name}")
        return original_cat_weights(local_weight, replica_weight, plan, ep_group)

    moe_layer.ExpertReplicaTransfer = TrackingTransfer
    moe_layer.all_to_all = tracking_all_to_all
    moe_layer.cat_local_and_replica_weights = tracking_cat
    moe_layer.EPMergedFc1GroupGemm.apply = staticmethod(cpu_apply)
    moe_layer.get_parallel_state = lambda: SimpleNamespace(ep_group=group)
    group_gemm.get_parallel_state = lambda: SimpleNamespace(ep_enabled=True)
    moe_monitor.set_active_monitor(monitor)
    try:
        output = group_gemm.group_gemm_fused_moe_forward(
            num_experts=2,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            fc1_1_weight=None,
            fc1_2_weight=None,
            fc2_weight=down_proj,
            fc1_1_2_weight=gate_up_proj,
            load_balancer=balancer,
        )
        expected = hidden_states * 6.0 * routing_weights
        assert torch.allclose(output, expected)
        assert torch.equal(routing_weights, routing_weights_before)
        assert balancer.plan.replicas == (
            ep_load_balance.ExpertReplica(
                source_expert=0,
                source_rank=0,
                source_local_expert=0,
                target_rank=1,
                target_slot=0,
                alias_expert=3,
                moved_tokens=5,
            ),
        )
        first_token_all_to_all = events.index("all_to_all")
        assert events[:first_token_all_to_all] == ["start_gate", "start_down"]
        assert events[first_token_all_to_all + 1 : first_token_all_to_all + 5] == [
            "wait_gate",
            "wait_down",
            "cat_gate",
            "cat_down",
        ]
        assert observed_cumsums[0].tolist() == ([5, 5] if rank == 0 else [0, 5])
        assert monitor.records == [(7, (10, 0), (5, 5), 1, 5)]

        output.sum().backward()
        weighted_input_sum = hidden_states.mul(routing_weights).sum()
        dist.all_reduce(weighted_input_sum, group=group)
        if rank == 0:
            assert torch.allclose(gate_up_proj.grad[0, 0, 0], weighted_input_sum * 3.0)
            assert torch.allclose(down_proj.grad[0, 0, 0], weighted_input_sum * 2.0)
        else:
            assert torch.count_nonzero(gate_up_proj.grad).item() == 0
            assert torch.count_nonzero(down_proj.grad).item() == 0
    finally:
        moe_monitor.set_active_monitor(None)
        moe_layer.ExpertReplicaTransfer = ep_load_balance.ExpertReplicaTransfer
        moe_layer.all_to_all = original_all_to_all
        moe_layer.cat_local_and_replica_weights = original_cat_weights
        moe_layer.EPMergedFc1GroupGemm.apply = original_apply
        moe_layer.get_parallel_state = original_get_parallel_state
        group_gemm.get_parallel_state = original_group_gemm_get_parallel_state

    dist.barrier(group=group)
    balanced_selected = torch.full((2, 1), rank, dtype=torch.long)
    balanced_hidden = torch.tensor([[2.0], [4.0]], requires_grad=True)
    balanced_routing = torch.tensor([[0.5], [0.75]])
    balanced_balancer = _RecordingBalancer(group)
    balanced_gate = nn.Parameter(torch.tensor([[[2.0], [0.0]]] if rank == 0 else [[[5.0], [0.0]]]))
    balanced_down = nn.Parameter(torch.tensor([[[3.0]]] if rank == 0 else [[[7.0]]]))

    class UnexpectedTransfer:
        @staticmethod
        def start(*args, **kwargs):
            raise AssertionError("A no-replica plan must not start weight P2P.")

    def unexpected_cat(*args, **kwargs):
        raise AssertionError("A no-replica plan must not concatenate temporary weights.")

    moe_layer.ExpertReplicaTransfer = UnexpectedTransfer
    moe_layer.cat_local_and_replica_weights = unexpected_cat
    moe_layer.EPMergedFc1GroupGemm.apply = staticmethod(_cpu_merged_expert)
    moe_layer.get_parallel_state = lambda: SimpleNamespace(ep_group=group)
    group_gemm.get_parallel_state = lambda: SimpleNamespace(ep_enabled=True)
    try:
        balanced_output = group_gemm.group_gemm_fused_moe_forward(
            num_experts=2,
            routing_weights=balanced_routing,
            selected_experts=balanced_selected,
            hidden_states=balanced_hidden,
            fc1_1_weight=None,
            fc1_2_weight=None,
            fc2_weight=balanced_down,
            fc1_1_2_weight=balanced_gate,
            load_balancer=balanced_balancer,
        )
        expected_scale = 6.0 if rank == 0 else 35.0
        assert torch.allclose(balanced_output, balanced_hidden * balanced_routing * expected_scale)
        assert balanced_balancer.plan.replicas == ()
    finally:
        moe_layer.ExpertReplicaTransfer = ep_load_balance.ExpertReplicaTransfer
        moe_layer.cat_local_and_replica_weights = original_cat_weights
        moe_layer.EPMergedFc1GroupGemm.apply = original_apply
        moe_layer.get_parallel_state = original_get_parallel_state
        group_gemm.get_parallel_state = original_group_gemm_get_parallel_state


def _two_rank_dispatch_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        _exercise_two_rank_dispatch(rank, dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def test_two_rank_balanced_dispatch_output_routing_and_owner_gradient(tmp_path: Path):
    rendezvous = f"file://{tmp_path / 'dispatch-two-rank'}"
    mp.spawn(_two_rank_dispatch_worker, args=(2, rendezvous), nprocs=2, join=True)


def _trainer_args(*, enabled: bool) -> SimpleNamespace:
    fsdp_config = SimpleNamespace(
        reshard_after_forward=True,
        mixed_precision=None,
        forward_prefetch=False,
        offload=False,
        max_load_broadcast_size=0,
    )
    train = SimpleNamespace(
        optimizer=SimpleNamespace(type="adamw", muon_expert_zero_comm=False),
        chunk_mbs_config=SimpleNamespace(enable=False),
        checkpoint=SimpleNamespace(load_path=None),
        init_device="meta",
        accelerator=SimpleNamespace(fsdp_config=fsdp_config),
        gradient_checkpointing=SimpleNamespace(enable=False, enable_reentrant=False, early_stop=True),
        broadcast_model_weights_from_rank0=True,
        ep_sharded_stream_load=False,
        torch_compile=CompileConfig(),
        moe_ep_load_balance=SimpleNamespace(enabled=enabled, max_replicas_per_rank=1),
    )
    model = SimpleNamespace(
        lora_config=None,
        fqn_to_index_mapping=None,
        model_path="unused",
        basic_modules=[],
    )
    return SimpleNamespace(model=model, train=train)


class _OrderModel(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.events = events

    def train(self, mode: bool = True):
        self.events.append("train")
        return super().train(mode)


def _install_ep_load_balance_test_module(monkeypatch):
    monkeypatch.setitem(sys.modules, _MOE_PACKAGE_NAME, _moe_package)
    monkeypatch.setitem(sys.modules, _EP_LOAD_BALANCE_NAME, ep_load_balance)


def test_trainer_attaches_after_parallelization_and_training(monkeypatch):
    events = []
    model = _OrderModel(events)
    trainer = SimpleNamespace(args=_trainer_args(enabled=True), model=model)

    def fake_build(model_arg, **kwargs):
        assert model_arg is model
        events.append("build")
        return model_arg

    def fake_attach(model_arg, max_replicas_per_rank):
        assert model_arg is model
        assert model_arg.training
        assert max_replicas_per_rank == 1
        events.append("attach")
        return 1

    monkeypatch.setattr(trainer_base, "build_parallelize_model", fake_build)
    monkeypatch.setattr(trainer_base, "should_skip_hf_weight_load", lambda *args: False)
    monkeypatch.setattr(ep_load_balance, "attach_qwen3_5_moe_ep_load_balancers", fake_attach)
    _install_ep_load_balance_test_module(monkeypatch)

    trainer_base.BaseTrainer._build_parallelized_model(trainer)

    assert trainer.model is model
    assert events == ["build", "train", "attach"]


def test_trainer_disabled_is_noop_and_enabled_requires_a_real_expert(monkeypatch):
    monkeypatch.setattr(trainer_base, "build_parallelize_model", lambda model, **kwargs: model)
    monkeypatch.setattr(trainer_base, "should_skip_hf_weight_load", lambda *args: False)
    _install_ep_load_balance_test_module(monkeypatch)

    def unexpected_attach(*args, **kwargs):
        raise AssertionError("Disabled configuration must not attach controllers.")

    monkeypatch.setattr(ep_load_balance, "attach_qwen3_5_moe_ep_load_balancers", unexpected_attach)
    disabled = SimpleNamespace(args=_trainer_args(enabled=False), model=_OrderModel([]))
    trainer_base.BaseTrainer._build_parallelized_model(disabled)

    monkeypatch.setattr(ep_load_balance, "attach_qwen3_5_moe_ep_load_balancers", lambda *args, **kwargs: 0)
    enabled = SimpleNamespace(args=_trainer_args(enabled=True), model=_OrderModel([]))
    with pytest.raises(RuntimeError, match="no compatible Qwen3.5 MoE expert modules"):
        trainer_base.BaseTrainer._build_parallelized_model(enabled)


def test_torchrun_two_rank_balanced_dispatch_output_routing_and_owner_gradient():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Run this test with torchrun --standalone --nproc-per-node=2.")
    dist.init_process_group("gloo")
    try:
        _exercise_two_rank_dispatch(dist.get_rank(), dist.group.WORLD)
    finally:
        dist.destroy_process_group()
