import importlib
import random
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


# The existing MoE package initializer imports Triton kernels eagerly. Load this
# pure CPU planner submodule without that unrelated optional dependency.
_MOE_PACKAGE_NAME = "veomni.distributed.moe"
_previous_moe_package = sys.modules.get(_MOE_PACKAGE_NAME)
_moe_package = types.ModuleType(_MOE_PACKAGE_NAME)
_moe_package.__path__ = [str(Path(__file__).parents[2] / "veomni" / "distributed" / "moe")]
sys.modules[_MOE_PACKAGE_NAME] = _moe_package
ep_load_balance = importlib.import_module(f"{_MOE_PACKAGE_NAME}.ep_load_balance")
if _previous_moe_package is None:
    del sys.modules[_MOE_PACKAGE_NAME]
else:
    sys.modules[_MOE_PACKAGE_NAME] = _previous_moe_package
attach_qwen3_5_moe_ep_load_balancers = ep_load_balance.attach_qwen3_5_moe_ep_load_balancers
build_ep_balance_plan = ep_load_balance.build_ep_balance_plan


def _selected_rows(counts: list[list[int]], seed: int = 0) -> list[torch.Tensor]:
    rows = []
    for rank, row in enumerate(counts):
        values = [expert for expert, count in enumerate(row) for _ in range(count)]
        random.Random(seed + rank).shuffle(values)
        rows.append(torch.tensor(values, dtype=torch.long).reshape(-1, 1))
    return rows


def _plans_from_counts(
    counts: list[list[int]], max_replicas_per_rank: int, seed: int = 0
) -> list[ep_load_balance.EPBalancePlan]:
    selected_rows = _selected_rows(counts, seed=seed)
    count_tensor = torch.tensor(counts, dtype=torch.long)
    return [
        ep_load_balance._build_ep_balance_plan_from_counts(
            selected_experts=selected,
            tokens_per_expert_by_rank=count_tensor,
            num_experts=count_tensor.shape[1],
            ep_rank=rank,
            max_replicas_per_rank=max_replicas_per_rank,
        )
        for rank, selected in enumerate(selected_rows)
    ]


def _assert_global_invariants(counts: list[list[int]], plans: list[ep_load_balance.EPBalancePlan]) -> None:
    reference = plans[0]
    ep_size = len(plans)
    total_tokens = sum(sum(row) for row in counts)

    assert all(plan.replicas == reference.replicas for plan in plans)
    assert all(plan.rank_loads_before == reference.rank_loads_before for plan in plans)
    assert all(plan.rank_loads_after == reference.rank_loads_after for plan in plans)
    assert sum(reference.rank_loads_before) == total_tokens
    assert sum(reference.rank_loads_after) == total_tokens

    for rank, plan in enumerate(plans):
        assert sum(plan.input_splits) == sum(counts[rank])
        assert sum(plan.output_splits) == plan.tokens_per_local_physical_expert.sum().item()
        assert tuple(plan.tokens_per_local_physical_expert.sum(dim=1).tolist()) == plan.output_splits
        assert plan.tokens_per_local_physical_expert.shape[0] == ep_size

    for sender in range(ep_size):
        for receiver in range(ep_size):
            assert plans[sender].input_splits[receiver] == plans[receiver].output_splits[sender]

    if reference.replicas:
        stride = reference.num_local_experts + reference.max_replicas_per_rank
        assert all(
            plan.selected_physical_experts.min().item() >= 0
            for plan in plans
            if plan.selected_physical_experts.numel()
        )
        assert all(
            plan.selected_physical_experts.max().item() < ep_size * stride
            for plan in plans
            if plan.selected_physical_experts.numel()
        )
        for replica in reference.replicas:
            alias_count = sum((plan.selected_physical_experts == replica.alias_expert).sum().item() for plan in plans)
            assert alias_count == replica.moved_tokens


def test_balanced_plan_is_a_true_logical_namespace_noop():
    counts = [[2, 0, 1, 1], [0, 2, 1, 1]]
    plans = _plans_from_counts(counts, max_replicas_per_rank=2)
    selected_rows = _selected_rows(counts)

    assert plans[0].replicas == ()
    assert plans[0].rank_loads_before == (4, 4)
    assert plans[0].rank_loads_after == (4, 4)
    assert plans[0].tokens_per_local_physical_expert.tolist() == [[2, 0], [0, 2]]
    assert all(torch.equal(plan.selected_physical_experts, selected_rows[rank]) for rank, plan in enumerate(plans))
    _assert_global_invariants(counts, plans)


def test_one_hot_expert_moves_exact_concrete_occurrences():
    counts = [[6, 0], [4, 0]]
    plans = _plans_from_counts(counts, max_replicas_per_rank=1)
    replica = plans[0].replicas[0]

    assert replica == ep_load_balance.ExpertReplica(
        source_expert=0,
        source_rank=0,
        source_local_expert=0,
        target_rank=1,
        target_slot=0,
        alias_expert=3,
        moved_tokens=5,
    )
    assert plans[0].rank_loads_before == (10, 0)
    assert plans[0].rank_loads_after == (5, 5)
    assert (plans[0].selected_physical_experts == replica.alias_expert).sum().item() == 5
    assert (plans[1].selected_physical_experts == replica.alias_expert).sum().item() == 0
    _assert_global_invariants(counts, plans)


def test_local_rewrite_uses_flattened_occurrence_order():
    selected = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    counts = torch.tensor([[3, 1], [0, 0]], dtype=torch.long)

    plan = ep_load_balance._build_ep_balance_plan_from_counts(
        selected_experts=selected,
        tokens_per_expert_by_rank=counts,
        num_experts=2,
        ep_rank=0,
        max_replicas_per_rank=1,
    )

    alias = plan.replicas[0].alias_expert
    assert plan.replicas[0].moved_tokens == 1
    assert plan.selected_physical_experts.tolist() == [[alias, 2], [0, 0]]


@pytest.mark.parametrize("logical_dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_augmented_physical_ids_promote_narrow_logical_dtypes_to_long(logical_dtype):
    num_experts = 128
    counts = torch.zeros((2, num_experts), dtype=torch.long)
    counts[0, 0] = 2
    selected = torch.zeros((2, 1), dtype=logical_dtype)

    plan = ep_load_balance._build_ep_balance_plan_from_counts(
        selected_experts=selected,
        tokens_per_expert_by_rank=counts,
        num_experts=num_experts,
        ep_rank=0,
        max_replicas_per_rank=64,
    )

    assert plan.replicas[0].alias_expert == 192
    assert plan.selected_physical_experts.dtype == torch.long
    assert plan.selected_physical_experts.device == selected.device
    assert plan.selected_physical_experts.shape == selected.shape
    assert plan.selected_physical_experts.flatten().tolist() == [192, 0]


def test_balanced_noop_physical_ids_are_long_without_changing_shape_or_device():
    selected = torch.tensor([[0]], dtype=torch.int8)
    counts = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)

    plan = ep_load_balance._build_ep_balance_plan_from_counts(
        selected_experts=selected,
        tokens_per_expert_by_rank=counts,
        num_experts=2,
        ep_rank=0,
        max_replicas_per_rank=1,
    )

    assert plan.replicas == ()
    assert plan.selected_physical_experts.dtype == torch.long
    assert plan.selected_physical_experts.device == selected.device
    assert plan.selected_physical_experts.shape == selected.shape
    assert plan.selected_physical_experts.tolist() == [[0]]


def test_one_hot_expert_fans_out_across_multiple_replicas():
    counts = [[100, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    plans = _plans_from_counts(counts, max_replicas_per_rank=1)

    assert plans[0].rank_loads_after == (25, 25, 25, 25)
    assert [(r.target_rank, r.target_slot, r.moved_tokens) for r in plans[0].replicas] == [
        (1, 0, 25),
        (2, 0, 25),
        (3, 0, 25),
    ]
    assert len({r.target_rank for r in plans[0].replicas}) == 3
    assert all(r.target_rank != r.source_rank for r in plans[0].replicas)
    _assert_global_invariants(counts, plans)


def test_replica_slots_are_never_overcommitted():
    counts = [[100, 0, 0, 0], [0, 80, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    plans = _plans_from_counts(counts, max_replicas_per_rank=1)

    targets = [replica.target_rank for replica in plans[0].replicas]
    assert all(targets.count(rank) <= 1 for rank in range(4))
    assert all(replica.target_rank != replica.source_rank for replica in plans[0].replicas)
    _assert_global_invariants(counts, plans)


def test_ties_choose_low_owner_expert_and_target_ids_deterministically():
    counts = [[6, 6, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0], [0, 0, 0, 0, 0, 0]]
    first = _plans_from_counts(counts, max_replicas_per_rank=2)
    second = _plans_from_counts(counts, max_replicas_per_rank=2)

    assert [(r.source_expert, r.source_rank, r.target_rank, r.target_slot) for r in first[0].replicas[:2]] == [
        (0, 0, 2, 0),
        (2, 1, 0, 0),
    ]
    assert first[0].replicas == second[0].replicas
    assert torch.equal(first[0].selected_physical_experts, second[0].selected_physical_experts)
    _assert_global_invariants(counts, first)


def test_no_move_is_accepted_without_strict_spread_improvement():
    counts = [[1, 0], [0, 0]]
    plans = _plans_from_counts(counts, max_replicas_per_rank=1)

    assert plans[0].replicas == ()
    assert plans[0].rank_loads_after == (1, 0)
    _assert_global_invariants(counts, plans)


def test_randomized_plans_are_deterministic_and_conservative():
    rng = random.Random(20260818)
    for case in range(150):
        ep_size = rng.randint(2, 5)
        num_local_experts = rng.randint(1, 4)
        num_experts = ep_size * num_local_experts
        max_replicas = rng.randint(1, num_local_experts)
        counts = [[rng.randint(0, 12) for _ in range(num_experts)] for _ in range(ep_size)]

        plans = _plans_from_counts(counts, max_replicas, seed=case)
        repeated = _plans_from_counts(counts, max_replicas, seed=case)

        assert plans[0].replicas == repeated[0].replicas
        assert plans[0].rank_loads_after == repeated[0].rank_loads_after
        assert max(plans[0].rank_loads_after) - min(plans[0].rank_loads_after) <= max(
            plans[0].rank_loads_before
        ) - min(plans[0].rank_loads_before)
        for plan, again in zip(plans, repeated, strict=True):
            assert torch.equal(plan.selected_physical_experts, again.selected_physical_experts)
            assert torch.equal(
                plan.tokens_per_local_physical_expert,
                again.tokens_per_local_physical_expert,
            )
        _assert_global_invariants(counts, plans)


def _gloo_plan_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        selected = torch.zeros((8, 1), dtype=torch.long)
        plan = build_ep_balance_plan(
            selected_experts=selected,
            num_experts=2,
            ep_group=dist.group.WORLD,
            max_replicas_per_rank=1,
        )
        expected = torch.full_like(selected, 3) if rank == 0 else torch.zeros_like(selected)
        assert torch.equal(plan.selected_physical_experts, expected)
        assert plan.input_splits == ((0, 8) if rank == 0 else (8, 0))
    finally:
        dist.destroy_process_group()


def test_public_builder_gathers_real_ep_histograms_and_rewrites_locally(tmp_path: Path):
    rendezvous = f"file://{tmp_path / 'ep-plan-rendezvous'}"
    mp.spawn(_gloo_plan_worker, args=(2, rendezvous), nprocs=2, join=True)


class _OrdinaryExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(4, 8, 4))
        self.down_proj = nn.Parameter(torch.empty(4, 4, 4))


def _qwen_experts(
    *,
    num_experts: int = 4,
    expert_rows: int | None = None,
    merged: bool = True,
    runtime_name: str = "Qwen3_5MoeExperts",
) -> nn.Module:
    def init(self):
        nn.Module.__init__(self)
        self.num_experts = num_experts
        rows = num_experts if expert_rows is None else expert_rows
        ndim_shape = (rows, 8, 4) if merged else (8, 4)
        self.gate_up_proj = nn.Parameter(torch.empty(ndim_shape))
        self.down_proj = nn.Parameter(torch.empty(ndim_shape))

    runtime_type = type(runtime_name, (nn.Module,), {"__init__": init})
    return runtime_type()


def test_attachment_accepts_post_fsdp2_ep_local_expert_rows(monkeypatch):
    group = object()
    monkeypatch.setattr(ep_load_balance, "get_parallel_state", lambda: SimpleNamespace(ep_group=group))
    monkeypatch.setattr(ep_load_balance.dist, "get_world_size", lambda group=None: 2)
    experts = _qwen_experts(num_experts=4, expert_rows=2)

    attached = attach_qwen3_5_moe_ep_load_balancers(nn.ModuleList([experts]), max_replicas_per_rank=2)

    assert attached == 1
    assert experts._veomni_ep_load_balancer.num_experts == 4
    assert experts._veomni_ep_load_balancer.ep_group is group


def test_attachment_accepts_fsdp2_dynamic_runtime_class(monkeypatch):
    group = object()
    monkeypatch.setattr(ep_load_balance, "get_parallel_state", lambda: SimpleNamespace(ep_group=group))
    monkeypatch.setattr(ep_load_balance.dist, "get_world_size", lambda group=None: 2)
    experts = _qwen_experts(expert_rows=2, runtime_name="FSDPQwen3_5MoeExperts")

    attached = attach_qwen3_5_moe_ep_load_balancers(nn.ModuleList([experts]), max_replicas_per_rank=2)

    assert attached == 1
    assert experts._veomni_ep_load_balancer.ep_group is group


def test_attachment_targets_only_runtime_merged_qwen_experts(monkeypatch):
    group = object()
    monkeypatch.setattr(ep_load_balance, "get_parallel_state", lambda: SimpleNamespace(ep_group=group))
    monkeypatch.setattr(ep_load_balance.dist, "get_world_size", lambda group=None: 2)
    first = _qwen_experts(expert_rows=2)
    incompatible = _qwen_experts(merged=False)
    ordinary = _OrdinaryExperts()
    second = _qwen_experts(expert_rows=2)
    model = nn.ModuleList([first, incompatible, ordinary, second])

    attached = attach_qwen3_5_moe_ep_load_balancers(model, max_replicas_per_rank=2)

    assert attached == 2
    assert not hasattr(incompatible, "_veomni_ep_load_balancer")
    assert not hasattr(ordinary, "_veomni_ep_load_balancer")
    for layer_index, module in enumerate((first, second)):
        controller = module._veomni_ep_load_balancer
        assert controller.layer_index == layer_index
        assert controller.ep_group is group
        assert controller.max_replicas_per_rank == 2
        assert callable(controller.build_plan)
        with pytest.raises(FrozenInstanceError):
            controller.layer_index = 99


@pytest.mark.parametrize(
    ("num_experts", "max_replicas", "message"),
    [
        (3, 1, "divisible"),
        (4, 3, "num_local_experts"),
        (4, 0, "positive"),
    ],
)
def test_attachment_validates_before_install(monkeypatch, num_experts, max_replicas, message):
    group = object()
    monkeypatch.setattr(ep_load_balance, "get_parallel_state", lambda: SimpleNamespace(ep_group=group))
    monkeypatch.setattr(ep_load_balance.dist, "get_world_size", lambda group=None: 2)
    experts = _qwen_experts(num_experts=num_experts)

    with pytest.raises(ValueError, match=message):
        attach_qwen3_5_moe_ep_load_balancers(nn.ModuleList([experts]), max_replicas)

    assert not hasattr(experts, "_veomni_ep_load_balancer")


def test_attachment_rejects_global_pre_fsdp_rows_before_installing_any_controller(monkeypatch):
    group = object()
    monkeypatch.setattr(ep_load_balance, "get_parallel_state", lambda: SimpleNamespace(ep_group=group))
    monkeypatch.setattr(ep_load_balance.dist, "get_world_size", lambda group=None: 2)
    valid = _qwen_experts(num_experts=4, expert_rows=2)
    global_rows = _qwen_experts(num_experts=4, expert_rows=4)

    with pytest.raises(ValueError, match=r"global/pre-FSDP.*after.*EP sharding.*FSDP2"):
        attach_qwen3_5_moe_ep_load_balancers(nn.ModuleList([valid, global_rows]), max_replicas_per_rank=2)

    assert not hasattr(valid, "_veomni_ep_load_balancer")
    assert not hasattr(global_rows, "_veomni_ep_load_balancer")


def test_attachment_rejects_either_wrong_ep_local_weight_row_count(monkeypatch):
    group = object()
    monkeypatch.setattr(ep_load_balance, "get_parallel_state", lambda: SimpleNamespace(ep_group=group))
    monkeypatch.setattr(ep_load_balance.dist, "get_world_size", lambda group=None: 2)
    experts = _qwen_experts(num_experts=4, expert_rows=2)
    experts.gate_up_proj = nn.Parameter(torch.empty(1, 8, 4))

    with pytest.raises(
        ValueError,
        match=r"exactly 2 EP-local expert rows.*gate_up_proj=1.*down_proj=2",
    ):
        attach_qwen3_5_moe_ep_load_balancers(nn.ModuleList([experts]), max_replicas_per_rank=2)

    assert not hasattr(experts, "_veomni_ep_load_balancer")
