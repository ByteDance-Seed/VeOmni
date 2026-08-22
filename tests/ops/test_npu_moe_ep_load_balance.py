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
import torch.nn.functional as F

import veomni.distributed  # noqa: F401 -- ensure parent attribute restoration is covered
from veomni.ops.kernels import moe as moe_kernels
from veomni.utils.device import IS_NPU_AVAILABLE


def _unexpected_accelerator_kernel(*args, **kwargs):
    raise AssertionError("CPU plumbing tests must not execute an accelerator kernel.")


def _cpu_token_permute(hidden_states, selected_experts):
    flat_experts = selected_experts.reshape(-1).to(torch.long)
    order = torch.argsort(flat_experts, stable=True)
    token_ids = torch.arange(hidden_states.shape[0], device=hidden_states.device).repeat_interleave(
        selected_experts.shape[-1]
    )
    return hidden_states[token_ids[order]], order


def _cpu_token_unpermute(hidden_states, row_ids_map, probs):
    flat_outputs = hidden_states.new_zeros((row_ids_map.numel(), hidden_states.shape[-1]))
    flat_outputs[row_ids_map.to(torch.long)] = hidden_states
    flat_weights = probs.reshape(-1).to(hidden_states.dtype)
    weighted = flat_outputs * flat_weights.unsqueeze(-1)
    return weighted.reshape(probs.shape[0], probs.shape[1], hidden_states.shape[-1]).sum(dim=1)


def _cpu_swiglu(hidden_states, dim=-1):
    gate, up = hidden_states.chunk(2, dim=dim)
    return F.silu(gate) * up


def _cpu_group_gemm(hidden_states, weight, group_list):
    counts = [int(value) for value in group_list.detach().cpu().tolist()]
    assert sum(counts) == hidden_states.shape[0]
    outputs = []
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        if count:
            outputs.append(hidden_states[start:end] @ weight[expert])
        start = end
    if outputs:
        return torch.cat(outputs, dim=0)
    return hidden_states.new_empty((0, weight.shape[-1]))


# Import the real VeOmni NPU wrapper with only its unavailable hardware boundary
# replaced. Planner, plan validation, replica transfer, token all-to-all, and
# combine code remain the production implementations.
_DISTRIBUTED_MOE_NAME = "veomni.distributed.moe"
_MOE_LAYER_NAME = f"{_DISTRIBUTED_MOE_NAME}.moe_layer"
_EP_LOAD_BALANCE_NAME = f"{_DISTRIBUTED_MOE_NAME}.ep_load_balance"
_MOE_COMM_NAME = f"{_DISTRIBUTED_MOE_NAME}.comm"
_MOE_UTILS_NAME = f"{_DISTRIBUTED_MOE_NAME}.moe_utils"
_TRITON_GROUP_GEMM_NAME = "veomni.ops.kernels.moe._kernels.kernel.group_gemm"
_NPU_GROUP_GEMM_KERNEL_NAME = "veomni.ops.kernels.moe._kernels.kernel.npu_group_gemm"
_NPU_MOE_NAME = "veomni.ops.kernels.moe.npu_group_gemm"
_MISSING_ATTRIBUTE = object()
_SEAM_NAMES = (
    "torch_npu",
    _DISTRIBUTED_MOE_NAME,
    _MOE_LAYER_NAME,
    _EP_LOAD_BALANCE_NAME,
    _MOE_COMM_NAME,
    _MOE_UTILS_NAME,
    _TRITON_GROUP_GEMM_NAME,
    _NPU_GROUP_GEMM_KERNEL_NAME,
    _NPU_MOE_NAME,
)
_saved_modules = {name: sys.modules.get(name) for name in _SEAM_NAMES}
_saved_parent_attributes = {}
for _child_name in _SEAM_NAMES:
    if "." not in _child_name:
        continue
    _parent_name, _attribute_name = _child_name.rsplit(".", 1)
    _parent_module = sys.modules.get(_parent_name)
    if _parent_module is not None:
        _saved_parent_attributes[_child_name] = (
            _parent_module,
            _attribute_name,
            getattr(_parent_module, _attribute_name, _MISSING_ATTRIBUTE),
        )

_fake_torch_npu = types.ModuleType("torch_npu")
_fake_torch_npu.npu_moe_token_permute = _cpu_token_permute
_fake_torch_npu.npu_moe_token_unpermute = _cpu_token_unpermute
_fake_torch_npu.npu_swiglu = _cpu_swiglu
sys.modules["torch_npu"] = _fake_torch_npu

_isolated_moe_package = types.ModuleType(_DISTRIBUTED_MOE_NAME)
_isolated_moe_package.__path__ = [str(Path(__file__).parents[2] / "veomni" / "distributed" / "moe")]
sys.modules[_DISTRIBUTED_MOE_NAME] = _isolated_moe_package

_triton_group_gemm_stub = types.ModuleType(_TRITON_GROUP_GEMM_NAME)
_triton_group_gemm_stub.group_gemm_same_mn = _unexpected_accelerator_kernel
_triton_group_gemm_stub.group_gemm_same_nk = _unexpected_accelerator_kernel
sys.modules[_TRITON_GROUP_GEMM_NAME] = _triton_group_gemm_stub

_npu_group_gemm_stub = types.ModuleType(_NPU_GROUP_GEMM_KERNEL_NAME)
_npu_group_gemm_stub.npu_group_gemm = _cpu_group_gemm
sys.modules[_NPU_GROUP_GEMM_KERNEL_NAME] = _npu_group_gemm_stub

try:
    moe_layer = importlib.import_module(_MOE_LAYER_NAME)
    ep_load_balance = importlib.import_module(_EP_LOAD_BALANCE_NAME)
    npu_moe = importlib.import_module(_NPU_MOE_NAME)
finally:
    for _name, _saved in _saved_modules.items():
        if _saved is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _saved
    for _parent_module, _attribute_name, _saved_attribute in _saved_parent_attributes.values():
        if _saved_attribute is _MISSING_ATTRIBUTE:
            if hasattr(_parent_module, _attribute_name):
                delattr(_parent_module, _attribute_name)
        else:
            setattr(_parent_module, _attribute_name, _saved_attribute)

_import_seam_modules_restored = all(
    (name not in sys.modules if saved is None else sys.modules.get(name) is saved)
    for name, saved in _saved_modules.items()
)
_import_seam_parent_attributes_restored = all(
    (
        not hasattr(parent_module, attribute_name)
        if saved_attribute is _MISSING_ATTRIBUTE
        else getattr(parent_module, attribute_name, _MISSING_ATTRIBUTE) is saved_attribute
    )
    for parent_module, attribute_name, saved_attribute in _saved_parent_attributes.values()
)


def test_cpu_import_seam_restores_modules_and_parent_attributes():
    assert _import_seam_modules_restored
    assert _import_seam_parent_attributes_restored
    assert getattr(moe_kernels, "npu_group_gemm", None) is not npu_moe


class _UnexpectedBalancer:
    def __init__(self, ep_group=None):
        self.ep_group = ep_group
        self.layer_index = 0
        self.build_calls = 0

    def build_plan(self, selected_experts):
        self.build_calls += 1
        raise AssertionError("Invalid NPU balancing input must fail before controller planning.")


def _call_public_npu(load_balancer, *, fc1_1_weight=None, fc1_2_weight=None, merged=None):
    return npu_moe.npu_fused_moe_forward(
        num_experts=2,
        routing_weights=torch.ones((1, 1)),
        selected_experts=torch.zeros((1, 1), dtype=torch.long),
        hidden_states=torch.ones((1, 1)),
        fc1_1_weight=fc1_1_weight,
        fc1_2_weight=fc1_2_weight,
        fc2_weight=torch.ones((1, 1, 1)),
        fc1_1_2_weight=merged,
        load_balancer=load_balancer,
    )


def test_disabled_public_npu_path_preserves_existing_call_shape(monkeypatch):
    ep_group = object()
    calls = []

    def fake_ep_forward(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.tensor([17.0])

    monkeypatch.setattr(npu_moe, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True, ep_group=ep_group))
    monkeypatch.setattr(npu_moe, "npu_ep_fused_moe_forward", fake_ep_forward)
    output = npu_moe.npu_fused_moe_forward(
        2,
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.long),
        torch.ones((1, 1)),
        None,
        None,
        torch.ones((1, 1, 1)),
        torch.ones((1, 2, 1)),
    )

    assert torch.equal(output, torch.tensor([17.0]))
    assert len(calls) == 1
    assert "load_balancer" not in calls[0][1]
    assert calls[0][1] == {"ep_group": ep_group, "swiglu_limit": None}


def test_public_npu_rejects_balancer_without_ep_before_planning(monkeypatch):
    balancer = _UnexpectedBalancer()
    monkeypatch.setattr(npu_moe, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False, ep_group=None))

    with pytest.raises(ValueError, match="load balancing.*expert parallel"):
        _call_public_npu(balancer, merged=torch.ones((1, 2, 1)))
    assert balancer.build_calls == 0


def test_npu_ep_rejects_split_fc1_before_planning():
    balancer = _UnexpectedBalancer()

    with pytest.raises(ValueError, match="load balancing.*merged"):
        npu_moe.npu_ep_fused_moe_forward(
            2,
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones((1, 1)),
            torch.ones((1, 1, 1)),
            torch.ones((1, 1, 1)),
            torch.ones((1, 1, 1)),
            load_balancer=balancer,
        )
    assert balancer.build_calls == 0


@pytest.mark.parametrize(
    ("gate_up", "down"),
    [
        (torch.ones((1, 3, 1)), torch.ones((1, 1, 1))),
        (torch.ones((1, 2, 1), dtype=torch.float32), torch.ones((1, 1, 1), dtype=torch.float64)),
    ],
)
def test_npu_ep_validates_merged_shapes_before_planning(monkeypatch, gate_up, down):
    ep_group = object()
    balancer = _UnexpectedBalancer(ep_group)
    monkeypatch.setattr(moe_layer, "get_parallel_state", lambda: SimpleNamespace(ep_group=ep_group))

    with pytest.raises(ValueError, match="compatible merged expert weights"):
        npu_moe.npu_ep_fused_moe_forward(
            2,
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones((1, 1)),
            None,
            None,
            down,
            fc1_1_2_weight=gate_up,
            ep_group=ep_group,
            load_balancer=balancer,
        )
    assert balancer.build_calls == 0


def test_npu_ep_rejects_group_mismatch_before_planning():
    balancer = _UnexpectedBalancer(object())

    with pytest.raises(ValueError, match="process group"):
        npu_moe.npu_ep_fused_moe_forward(
            2,
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones((1, 1)),
            None,
            None,
            torch.ones((1, 1, 1)),
            fc1_1_2_weight=torch.ones((1, 2, 1)),
            ep_group=object(),
            load_balancer=balancer,
        )
    assert balancer.build_calls == 0


def test_npu_ep_rejects_invalid_plan_before_parameter_or_token_communication(monkeypatch):
    ep_group = object()
    replica = ep_load_balance.ExpertReplica(0, 0, 0, 1, 0, 3, 1)
    invalid_plan = ep_load_balance.EPBalancePlan(
        ep_size=2,
        num_experts=2,
        num_local_experts=1,
        max_replicas_per_rank=1,
        selected_physical_experts=torch.tensor([[4]]),
        input_splits=(1, 0),
        output_splits=(1, 0),
        tokens_per_local_physical_expert=torch.tensor([[1, 0], [0, 0]]),
        replicas=(replica,),
        rank_loads_before=(1, 0),
        rank_loads_after=(0, 1),
    )

    class InvalidPlanBalancer:
        layer_index = 0

        def __init__(self):
            self.ep_group = ep_group
            self.build_calls = 0

        def build_plan(self, selected_experts):
            self.build_calls += 1
            return invalid_plan

    class UnexpectedTransfer:
        @staticmethod
        def start(*args, **kwargs):
            raise AssertionError("Invalid plans must fail before parameter P2P.")

    def unexpected_dispatch(*args, **kwargs):
        raise AssertionError("Invalid plans must fail before token all-to-all.")

    balancer = InvalidPlanBalancer()
    monkeypatch.setattr(moe_layer, "get_parallel_state", lambda: SimpleNamespace(ep_group=ep_group))
    monkeypatch.setattr(npu_moe, "ExpertReplicaTransfer", UnexpectedTransfer, raising=False)
    monkeypatch.setattr(npu_moe, "alltoall_dispatch", unexpected_dispatch)
    with pytest.raises(ValueError, match="physical expert IDs"):
        npu_moe.npu_ep_fused_moe_forward(
            2,
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones((1, 1)),
            None,
            None,
            torch.ones((1, 1, 1)),
            fc1_1_2_weight=torch.ones((1, 2, 1)),
            ep_group=ep_group,
            load_balancer=balancer,
        )
    assert balancer.build_calls == 1


class _RecordingBalancer:
    def __init__(self, ep_group, balance_module=ep_load_balance):
        self.ep_group = ep_group
        self.layer_index = 7
        self.balance_module = balance_module
        self.plan = None
        self.build_calls = 0

    def build_plan(self, selected_experts):
        self.build_calls += 1
        self.plan = self.balance_module.build_ep_balance_plan(
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


class _ExpertWeights(nn.Module):
    def __init__(self, rank, *, dtype=torch.float32, device="cpu"):
        super().__init__()
        gate_up = [[[2.0], [1.0]]] if rank == 0 else [[[5.0], [1.0]]]
        down = [[[3.0]]] if rank == 0 else [[[7.0]]]
        self.gate_up = nn.Parameter(torch.tensor(gate_up, dtype=dtype, device=device))
        self.down = nn.Parameter(torch.tensor(down, dtype=dtype, device=device))


def _reference_owner_gradients(hidden_states, routing_weights, group):
    gate_up = torch.tensor([2.0, 1.0], dtype=hidden_states.dtype, device=hidden_states.device, requires_grad=True)
    down = torch.tensor(3.0, dtype=hidden_states.dtype, device=hidden_states.device, requires_grad=True)
    output = F.silu(hidden_states * gate_up[0]) * (hidden_states * gate_up[1]) * down * routing_weights
    output.sum().backward()
    gate_grad = gate_up.grad.detach().clone()
    down_grad = down.grad.detach().clone()
    dist.all_reduce(gate_grad, group=group)
    dist.all_reduce(down_grad, group=group)
    return gate_grad, down_grad


def _exercise_cpu_two_rank_npu_plumbing(rank, group):
    from veomni.utils import moe_monitor

    selected_experts = torch.zeros((6 if rank == 0 else 4, 1), dtype=torch.long)
    routing_weights = torch.linspace(0.25, 1.0, selected_experts.shape[0]).reshape(-1, 1)
    routing_bytes = routing_weights.contiguous().view(torch.uint8).clone()
    hidden_states = torch.arange(1 if rank == 0 else 7, 7 if rank == 0 else 11, dtype=torch.float32).reshape(-1, 1)
    weights = _ExpertWeights(rank)
    parameter_names_before = tuple(dict(weights.named_parameters()))
    balancer = _RecordingBalancer(group)
    monitor = _BalanceMonitor()
    events = []
    observed_group_lists = []

    original_start = npu_moe.ExpertReplicaTransfer.start
    original_all_to_all = npu_moe.all_to_all
    original_cat = npu_moe.cat_local_and_replica_weights
    original_group_gemm = npu_moe.npu_group_gemm
    original_stream_synchronize = npu_moe.stream_synchronize
    original_get_parallel_state = npu_moe.get_parallel_state
    original_moe_layer_get_parallel_state = moe_layer.get_parallel_state

    class TrackingTransfer:
        @staticmethod
        def start(local_weight, plan, ep_group):
            name = "gate" if local_weight is weights.gate_up else "down"
            events.append(f"start_{name}")
            transfer = original_start(local_weight, plan, ep_group)

            class TrackingWait:
                def wait(self):
                    events.append(f"wait_{name}")
                    return transfer.wait()

            return TrackingWait()

    def tracking_all_to_all(*args, **kwargs):
        events.append("all_to_all")
        return original_all_to_all(*args, **kwargs)

    def tracking_cat(local_weight, replica_weight, plan, ep_group):
        name = "gate" if local_weight is weights.gate_up else "down"
        events.append(f"cat_{name}")
        return original_cat(local_weight, replica_weight, plan, ep_group)

    def tracking_group_gemm(hidden, weight, group_list):
        observed_group_lists.append(group_list.detach().cpu().clone())
        return original_group_gemm(hidden, weight, group_list)

    npu_moe.ExpertReplicaTransfer = TrackingTransfer
    npu_moe.all_to_all = tracking_all_to_all
    npu_moe.cat_local_and_replica_weights = tracking_cat
    npu_moe.npu_group_gemm = tracking_group_gemm
    npu_moe.stream_synchronize = lambda: None
    npu_moe.get_parallel_state = lambda: SimpleNamespace(ep_enabled=True, ep_group=group)
    moe_layer.get_parallel_state = lambda: SimpleNamespace(ep_group=group)
    moe_monitor.set_active_monitor(monitor)
    try:
        output = npu_moe.npu_fused_moe_forward(
            2,
            routing_weights,
            selected_experts,
            hidden_states,
            None,
            None,
            weights.down,
            weights.gate_up,
            load_balancer=balancer,
        )
        expected = F.silu(hidden_states * 2.0) * hidden_states * 3.0 * routing_weights
        assert torch.allclose(output, expected)
        assert torch.equal(routing_weights.contiguous().view(torch.uint8), routing_bytes)
        assert balancer.build_calls == 1
        assert balancer.plan.replicas == (ep_load_balance.ExpertReplica(0, 0, 0, 1, 0, 3, 5),)
        assert events[:7] == [
            "start_gate",
            "start_down",
            "all_to_all",
            "wait_gate",
            "wait_down",
            "cat_gate",
            "cat_down",
        ]
        expected_counts = [5, 0] if rank == 0 else [0, 5]
        assert observed_group_lists[0].tolist() == expected_counts
        assert observed_group_lists[1].tolist() == expected_counts
        assert monitor.records == [(7, (10, 0), (5, 5), 1, 5)]

        output.sum().backward()
        expected_gate_grad, expected_down_grad = _reference_owner_gradients(hidden_states, routing_weights, group)
        if rank == 0:
            assert torch.allclose(weights.gate_up.grad[0, :, 0], expected_gate_grad)
            assert torch.allclose(weights.down.grad[0, 0, 0], expected_down_grad)
        else:
            assert torch.count_nonzero(weights.gate_up.grad).item() == 0
            assert torch.count_nonzero(weights.down.grad).item() == 0
        assert tuple(dict(weights.named_parameters())) == parameter_names_before
    finally:
        moe_monitor.set_active_monitor(None)
        npu_moe.ExpertReplicaTransfer = ep_load_balance.ExpertReplicaTransfer
        npu_moe.all_to_all = original_all_to_all
        npu_moe.cat_local_and_replica_weights = original_cat
        npu_moe.npu_group_gemm = original_group_gemm
        npu_moe.stream_synchronize = original_stream_synchronize
        npu_moe.get_parallel_state = original_get_parallel_state
        moe_layer.get_parallel_state = original_moe_layer_get_parallel_state

    dist.barrier(group=group)
    balanced_selected = torch.full((2, 1), rank, dtype=torch.long)
    balanced_hidden = torch.tensor([[0.5], [1.0]])
    balanced_routing = torch.tensor([[0.5], [0.75]])
    balanced_weights = _ExpertWeights(rank)
    balanced_balancer = _RecordingBalancer(group)

    class UnexpectedTransfer:
        @staticmethod
        def start(*args, **kwargs):
            raise AssertionError("A no-replica plan must not start parameter P2P.")

    def unexpected_cat(*args, **kwargs):
        raise AssertionError("A no-replica plan must not concatenate temporary weights.")

    npu_moe.ExpertReplicaTransfer = UnexpectedTransfer
    npu_moe.cat_local_and_replica_weights = unexpected_cat
    npu_moe.stream_synchronize = lambda: None
    npu_moe.get_parallel_state = lambda: SimpleNamespace(ep_enabled=True, ep_group=group)
    moe_layer.get_parallel_state = lambda: SimpleNamespace(ep_group=group)
    try:
        balanced_output = npu_moe.npu_fused_moe_forward(
            2,
            balanced_routing,
            balanced_selected,
            balanced_hidden,
            None,
            None,
            balanced_weights.down,
            balanced_weights.gate_up,
            load_balancer=balanced_balancer,
        )
        gate, up, down = (2.0, 1.0, 3.0) if rank == 0 else (5.0, 1.0, 7.0)
        expected = F.silu(balanced_hidden * gate) * (balanced_hidden * up) * down * balanced_routing
        assert torch.allclose(balanced_output, expected)
        assert balanced_balancer.build_calls == 1
        assert balanced_balancer.plan.replicas == ()
    finally:
        npu_moe.ExpertReplicaTransfer = ep_load_balance.ExpertReplicaTransfer
        npu_moe.cat_local_and_replica_weights = original_cat
        npu_moe.stream_synchronize = original_stream_synchronize
        npu_moe.get_parallel_state = original_get_parallel_state
        moe_layer.get_parallel_state = original_moe_layer_get_parallel_state


def _cpu_two_rank_worker(rank, world_size, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        _exercise_cpu_two_rank_npu_plumbing(rank, dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_plumbing_alias_routing_order_and_owner_gradients(tmp_path):
    rendezvous = f"file://{tmp_path / 'npu-moe-balance-world2'}"
    mp.spawn(_cpu_two_rank_worker, args=(2, rendezvous), nprocs=2, join=True)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Real NPU fused MoE balance requires torch_npu and Ascend hardware")
def test_real_npu_two_rank_alias_forward_backward_and_owner_gradient():
    world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        pytest.skip("Run with torchrun --standalone --nproc-per-node=2 on an Ascend host.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("hccl")

    real_npu_moe = importlib.import_module(_NPU_MOE_NAME)
    real_moe_layer = importlib.import_module(_MOE_LAYER_NAME)
    real_ep_load_balance = importlib.import_module(_EP_LOAD_BALANCE_NAME)
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    device = torch.device(f"npu:{local_rank}")
    selected_experts = torch.zeros((6 if rank == 0 else 4, 1), dtype=torch.long, device=device)
    routing_weights = torch.linspace(
        0.25, 1.0, selected_experts.shape[0], dtype=torch.bfloat16, device=device
    ).reshape(-1, 1)
    routing_before = routing_weights.clone()
    hidden_states = torch.linspace(
        0.1 if rank == 0 else 0.7,
        0.6 if rank == 0 else 1.0,
        selected_experts.shape[0],
        dtype=torch.bfloat16,
        device=device,
    ).reshape(-1, 1)
    weights = _ExpertWeights(rank, dtype=torch.bfloat16, device=device)
    parameter_names_before = tuple(dict(weights.named_parameters()))
    balancer = _RecordingBalancer(group, real_ep_load_balance)
    original_get_parallel_state = real_npu_moe.get_parallel_state
    original_moe_layer_get_parallel_state = real_moe_layer.get_parallel_state
    real_npu_moe.get_parallel_state = lambda: SimpleNamespace(ep_enabled=True, ep_group=group)
    real_moe_layer.get_parallel_state = lambda: SimpleNamespace(ep_group=group)
    try:
        output = real_npu_moe.npu_fused_moe_forward(
            2,
            routing_weights,
            selected_experts,
            hidden_states,
            None,
            None,
            weights.down,
            weights.gate_up,
            load_balancer=balancer,
        )
        assert torch.isfinite(output.float()).all().item()
        assert torch.equal(routing_weights, routing_before)
        assert balancer.build_calls == 1
        assert len(balancer.plan.replicas) == 1
        assert int(balancer.plan.tokens_per_local_physical_expert[:, -1].sum().item()) == (0 if rank == 0 else 5)

        output.float().sum().backward()
        assert torch.isfinite(weights.gate_up.grad.float()).all().item()
        assert torch.isfinite(weights.down.grad.float()).all().item()
        expected_gate_grad, expected_down_grad = _reference_owner_gradients(
            hidden_states.float(), routing_weights.float(), group
        )
        if rank == 0:
            assert torch.allclose(weights.gate_up.grad[0, :, 0].float(), expected_gate_grad, rtol=0.05, atol=0.05)
            assert torch.allclose(weights.down.grad[0, 0, 0].float(), expected_down_grad, rtol=0.05, atol=0.05)
        else:
            assert torch.count_nonzero(weights.gate_up.grad).item() == 0
            assert torch.count_nonzero(weights.down.grad).item() == 0
        assert tuple(dict(weights.named_parameters())) == parameter_names_before
    finally:
        real_npu_moe.get_parallel_state = original_get_parallel_state
        real_moe_layer.get_parallel_state = original_moe_layer_get_parallel_state
        if owns_group:
            dist.destroy_process_group()
