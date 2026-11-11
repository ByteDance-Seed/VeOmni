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

import gc
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


def _cpu_native_weight_grad(x, weight, *, group_list, split_item, group_type, group_list_type, **kwargs):
    assert (split_item, group_type, group_list_type) == (3, 2, 1)
    tokens, grad = x[0].T, weight[0]
    values = []
    start = 0
    for count in group_list.tolist():
        values.append(tokens[start : start + count].T @ grad[start : start + count])
        start += count
    return [torch.stack(values)]


# Import the real VeOmni NPU wrapper with only its unavailable hardware boundary
# replaced. Planner, plan validation, replica transfer, token all-to-all, and
# combine code remain the production implementations.
_DISTRIBUTED_MOE_NAME = "veomni.distributed.moe"
_EP_DISPATCH_NAME = f"{_DISTRIBUTED_MOE_NAME}.ep_dispatch"
_EP_LOAD_BALANCE_NAME = f"{_DISTRIBUTED_MOE_NAME}.ep_load_balance"
_MOE_COMM_NAME = f"{_DISTRIBUTED_MOE_NAME}.comm"
_MOE_UTILS_NAME = f"{_DISTRIBUTED_MOE_NAME}.moe_utils"
_NPU_GROUP_GEMM_KERNEL_NAME = "veomni.ops.kernels.moe._kernels.kernel.npu_group_gemm"
_NPU_MOE_NAME = "veomni.ops.kernels.moe.npu_group_gemm"
_NPU_MEMORY_NAME = "veomni.ops.kernels.moe.npu_ep_memory"
_MISSING_ATTRIBUTE = object()
_SEAM_NAMES = (
    "torch_npu",
    _DISTRIBUTED_MOE_NAME,
    _EP_DISPATCH_NAME,
    _EP_LOAD_BALANCE_NAME,
    _MOE_COMM_NAME,
    _MOE_UTILS_NAME,
    _NPU_GROUP_GEMM_KERNEL_NAME,
    _NPU_MEMORY_NAME,
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
_fake_torch_npu.npu_grouped_matmul = _cpu_native_weight_grad
sys.modules["torch_npu"] = _fake_torch_npu

_isolated_moe_package = types.ModuleType(_DISTRIBUTED_MOE_NAME)
_isolated_moe_package.__path__ = [str(Path(__file__).parents[2] / "veomni" / "distributed" / "moe")]
sys.modules[_DISTRIBUTED_MOE_NAME] = _isolated_moe_package

_npu_group_gemm_stub = types.ModuleType(_NPU_GROUP_GEMM_KERNEL_NAME)
_npu_group_gemm_stub.npu_group_gemm = _cpu_group_gemm
sys.modules[_NPU_GROUP_GEMM_KERNEL_NAME] = _npu_group_gemm_stub

try:
    ep_dispatch = importlib.import_module(_EP_DISPATCH_NAME)
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
    monkeypatch.setattr(ep_dispatch, "get_parallel_state", lambda: SimpleNamespace(ep_group=ep_group))

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
    monkeypatch.setattr(ep_dispatch, "get_parallel_state", lambda: SimpleNamespace(ep_group=ep_group))
    monkeypatch.setattr(npu_moe, "ExpertReplicaTransfer", UnexpectedTransfer)
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
    def __init__(self, ep_group, balance_module=ep_load_balance, *, optimize_memory=False):
        self.ep_group = ep_group
        self.layer_index = 7
        self.balance_module = balance_module
        self.plan = None
        self.build_calls = 0
        self.optimize_memory = optimize_memory
        self.buffer_pool = balance_module.ReplicaBufferPool() if optimize_memory else None

    def build_plan(self, selected_experts):
        self.build_calls += 1
        self.plan = self.balance_module.build_ep_balance_plan(
            selected_experts,
            num_experts=2,
            ep_group=self.ep_group,
            max_replicas_per_rank=1,
        )
        return self.plan


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


def _cpu_partial_bf16_worker(rank, rendezvous, optimize_memory):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        native_gather = dist.all_gather_into_tensor
        # Gloo requires flat output; the native HCCL boundary accepts stacked rows.
        dist.all_gather_into_tensor = lambda output, value, *args, **kwargs: native_gather(
            output.view(-1), value.view(-1), *args, **kwargs
        )
        state = SimpleNamespace(ep_enabled=True, ep_group=dist.group.WORLD)
        npu_moe.get_parallel_state = lambda: state
        ep_dispatch.get_parallel_state = lambda: state
        npu_moe.stream_synchronize = lambda: None
        generator = torch.Generator().manual_seed(1009 * rank)
        operands = {
            "x": torch.randn(16, 8, generator=generator).bfloat16(),
            "gate": torch.randn(1, 8, 8, generator=generator).bfloat16() / 8**0.5,
            "down": torch.randn(1, 8, 4, generator=generator).bfloat16() / 4**0.5,
            "route": torch.ones(16, 1, dtype=torch.bfloat16),
        }
        dy = torch.randn(16, 8, generator=generator).bfloat16()
        selected = torch.zeros(16, 1, dtype=torch.long)
        selected[-1] = 1
        results = []
        for enabled in (False, True):
            values = {key: value.clone().requires_grad_() for key, value in operands.items()}
            balancer = _RecordingBalancer(dist.group.WORLD, optimize_memory=optimize_memory) if enabled else None
            output = npu_moe.npu_ep_fused_moe_forward(
                2,
                values["route"],
                selected,
                values["x"],
                None,
                None,
                values["down"],
                values["gate"],
                ep_group=dist.group.WORLD,
                load_balancer=balancer,
            )
            output.backward(dy)
            results.append({"output": output.detach(), **{key: value.grad for key, value in values.items()}})
            if enabled:
                assert 0 < sum(replica.moved_tokens for replica in balancer.plan.replicas) < 30
        for name in results[0]:
            torch.testing.assert_close(results[1][name], results[0][name], atol=1e-4, rtol=1e-3, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("optimize_memory", [False, True])
def test_partial_expert_bf16_owner_gradients_match_off(tmp_path, optimize_memory):
    rendezvous = f"file://{tmp_path / 'npu-partial-bf16'}"
    mp.spawn(_cpu_partial_bf16_worker, args=(rendezvous, optimize_memory), nprocs=2, join=True)


@pytest.mark.parametrize(
    "retained,outgoing,expected", [([5, 11], True, [14, 11]), ([0, 0], True, [9, 0]), ([0, 0], False, [0, 0])]
)
def test_owner_token_counts_exclude_incoming_replicas_and_preserve_physical_counts(retained, outgoing, expected):
    replicas = [SimpleNamespace(source_rank=1, source_local_expert=1, moved_tokens=9)]
    if outgoing:
        replicas.extend(SimpleNamespace(source_rank=0, source_local_expert=0, moved_tokens=value) for value in (7, 2))
    plan = SimpleNamespace(
        num_local_experts=2,
        replicas=tuple(replicas),
    )
    physical = torch.tensor([*retained, 9], dtype=torch.int32)
    before = physical.clone()
    owner = npu_moe.npu_ep_memory._owner_token_counts(physical, plan, rank=0)
    assert owner.dtype == torch.int64
    assert owner.tolist() == expected
    assert torch.equal(physical, before)


def _cpu_restore_owner_rows_worker(rank, rendezvous, top_k):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    try:
        npu_moe.stream_synchronize = lambda: None
        experts = 4 * top_k
        selected = torch.arange(top_k).repeat(20, 1)
        selected[-1] = torch.arange(top_k) + rank * top_k
        tokens = torch.arange(60, dtype=torch.float32).reshape(20, 3) + rank * 1000
        plan = ep_load_balance.build_ep_balance_plan(selected, experts, dist.group.WORLD, 1)
        assert len(plan.replicas) >= 2
        physical_experts = 4 * plan.tokens_per_local_physical_expert.shape[1]
        physical, _ = npu_moe.alltoall_dispatch(
            tokens,
            plan.selected_physical_experts,
            plan.input_splits,
            plan.output_splits,
            physical_experts,
            plan.tokens_per_local_physical_expert,
            dist.group.WORLD,
        )
        gradient = physical[:, :2] * 0.25 + 1
        physical_before, gradient_before = physical.clone(), gradient.clone()
        references, counts = [], []
        for expert in range(rank * top_k, (rank + 1) * top_k):
            parts = []
            for sender in range(4):
                sender_selected = torch.arange(top_k).repeat(20, 1)
                sender_selected[-1] = torch.arange(top_k) + sender * top_k
                sender_tokens = torch.arange(60, dtype=torch.float32).reshape(20, 3) + sender * 1000
                parts.append(sender_tokens.repeat_interleave(top_k, 0)[sender_selected.reshape(-1) == expert])
            rows = torch.cat(parts)
            references.append(rows)
            counts.append(len(rows))
        reference = torch.cat(references)
        host_counts = plan.tokens_per_local_physical_expert.sum(0)
        host_before = host_counts.clone()
        prepared_counts = npu_moe.npu_ep_memory._owner_token_counts(host_counts, plan, rank)
        assert prepared_counts.tolist() == counts
        assert torch.equal(host_counts, host_before)
        for device_counts in (None, prepared_counts):
            for _ in range(2):
                owner, owner_grad, owner_counts = npu_moe.npu_ep_memory.restore_owner_wgrad_inputs(
                    physical,
                    gradient,
                    plan,
                    dist.group.WORLD,
                    device_counts=device_counts,
                )
                assert torch.equal(owner, reference)
                assert torch.equal(owner_grad, reference[:, :2] * 0.25 + 1)
                assert owner_counts.tolist() == counts
                assert torch.equal(physical, physical_before)
                assert torch.equal(gradient, gradient_before)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("top_k", [1, 2])
def test_replica_prefixes_restore_exact_owner_rows_with_multiple_targets(tmp_path, top_k):
    rendezvous = f"file://{tmp_path / 'npu-owner-order'}"
    mp.spawn(_cpu_restore_owner_rows_worker, args=(rendezvous, top_k), nprocs=4, join=True)


def _cpu_memory_lifetime_worker(rank, world_size, rendezvous):
    from torch.utils.checkpoint import DefaultDeviceType, checkpoint

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    group = dist.group.WORLD
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(DefaultDeviceType, "get_device_type", staticmethod(lambda: "cpu"))
            patch.setattr(npu_moe, "stream_synchronize", lambda: None)
            patch.setattr(npu_moe, "_is_triton_ascend_available", lambda: False)
            patch.setattr(npu_moe, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True, ep_group=group))
            patch.setattr(ep_dispatch, "get_parallel_state", lambda: SimpleNamespace(ep_group=group))
            for frozen in ((), ("gate_up",), ("down",), ("gate_up", "down")):
                for limit in (None, 0.7):
                    references = []
                    for optimize in (False, True):
                        weights = _ExpertWeights(rank)
                        for name in frozen:
                            getattr(weights, name).requires_grad_(False)
                        balancer = _RecordingBalancer(group, optimize_memory=optimize)
                        outputs, inputs = [], []
                        for hotspot in (0, 1):
                            hidden = torch.linspace(0.1, 0.8, 6 if rank == 0 else 4).reshape(-1, 1)
                            hidden.requires_grad_(True)
                            outputs.append(
                                npu_moe.npu_fused_moe_forward(
                                    2,
                                    torch.ones_like(hidden),
                                    torch.full_like(hidden, hotspot, dtype=torch.long),
                                    hidden,
                                    None,
                                    None,
                                    weights.down,
                                    weights.gate_up,
                                    swiglu_limit=limit,
                                    load_balancer=balancer,
                                )
                            )
                            inputs.append(hidden)
                        # Neither live graph may be overwritten by later pool borrowers.
                        borrowers = []
                        if optimize:
                            for parameter in (weights.gate_up, weights.down):
                                lease = balancer.buffer_pool.acquire(parameter, parameter.shape)
                                lease.tensor.fill_(123)
                                borrowers.append(lease)
                        loss = sum(output.sum() for output in outputs)
                        loss.backward(retain_graph=True)
                        if optimize:
                            for parameter in (weights.gate_up, weights.down):
                                shape = (balancer.plan.max_replicas_per_rank, *parameter.shape[1:])
                                lease = balancer.buffer_pool.acquire(parameter, shape)
                                lease.tensor.fill_(456)
                                borrowers.append(lease)
                        loss.backward()
                        observed = [output.detach().clone() for output in outputs]
                        observed += [hidden.grad.clone() for hidden in inputs]
                        observed += [
                            parameter.grad.clone() for parameter in weights.parameters() if parameter.requires_grad
                        ]
                        if optimize:
                            for result, expected in zip(observed, references, strict=True):
                                torch.testing.assert_close(result, expected)
                        else:
                            references = observed
                        del loss, outputs, inputs, borrowers
                        gc.collect()
                        # Inference -> training must refresh values and keep valid autograd storage.
                        if optimize:
                            hidden = torch.ones((4, 1), requires_grad=True)
                            selected = torch.zeros((4, 1), dtype=torch.long)
                            with torch.inference_mode():
                                npu_moe.npu_fused_moe_forward(
                                    2,
                                    torch.ones_like(hidden),
                                    selected,
                                    hidden,
                                    None,
                                    None,
                                    weights.down,
                                    weights.gate_up,
                                    load_balancer=balancer,
                                )
                            with torch.no_grad():
                                weights.down.mul_(2)
                            refreshed = npu_moe.npu_fused_moe_forward(
                                2,
                                torch.ones_like(hidden),
                                selected,
                                hidden,
                                None,
                                None,
                                weights.down,
                                weights.gate_up,
                                load_balancer=balancer,
                            )
                            torch.testing.assert_close(refreshed, F.silu(hidden * 2) * hidden * 6)
                            refreshed.sum().backward()
                            reference_input = hidden.detach().clone().requires_grad_(True)
                            (F.silu(reference_input * 2) * reference_input * 6).sum().backward()
                            torch.testing.assert_close(hidden.grad, reference_input.grad)
            for reentrant in (False, True):
                weights = _ExpertWeights(rank)
                balancer = _RecordingBalancer(group, optimize_memory=True)
                hidden = torch.linspace(0.1, 0.8, 4).reshape(-1, 1).requires_grad_(True)

                def forward(tokens, weights=weights, balancer=balancer):
                    return npu_moe.npu_fused_moe_forward(
                        2,
                        torch.ones_like(tokens),
                        torch.zeros_like(tokens, dtype=torch.long),
                        tokens,
                        None,
                        None,
                        weights.down,
                        weights.gate_up,
                        load_balancer=balancer,
                    )

                output = checkpoint(forward, hidden, use_reentrant=reentrant)
                output.sum().backward()
                reference_input = hidden.detach().clone().requires_grad_(True)
                expected = F.silu(reference_input * 2) * reference_input * 3
                expected.sum().backward()
                torch.testing.assert_close(output, expected)
                torch.testing.assert_close(hidden.grad, reference_input.grad)
                gate_grad, down_grad = _reference_owner_gradients(hidden.detach(), torch.ones_like(hidden), group)
                if rank == 0:
                    torch.testing.assert_close(weights.gate_up.grad[0, :, 0], gate_grad)
                    torch.testing.assert_close(weights.down.grad[0, 0, 0], down_grad)
                else:
                    assert not weights.gate_up.grad.count_nonzero()
                    assert not weights.down.grad.count_nonzero()
    finally:
        dist.destroy_process_group()


def test_cpu_memory_frozen_clamped_retained_graph_and_dynamic_hotspots(tmp_path):
    rendezvous = f"file://{tmp_path / 'npu-memory-lifetime'}"
    mp.spawn(_cpu_memory_lifetime_worker, args=(2, rendezvous), nprocs=2, join=True)


@pytest.mark.parametrize("counts", [[2, 0, 1, 3], [0, 0, 3, 2], [3, 2, 0, 0], [0, 0, 0, 0]])
def test_segmented_gmm_empty_segments_preserve_input_and_weight_gradients(counts):
    generator = torch.Generator().manual_seed(4)
    tokens = torch.randn(sum(counts), 3, generator=generator, requires_grad=True)
    local = torch.randn(2, 5, 3, generator=generator, requires_grad=True)
    replica = torch.randn(2, 5, 3, generator=generator, requires_grad=True)
    actual = npu_moe.npu_ep_memory.segmented_group_gemm(
        tokens, local, replica, torch.tensor(counts), sum(counts[:2]), _cpu_group_gemm
    )
    weights = torch.cat((local, replica))
    segments = tokens.split(counts)
    expected = torch.cat([segment @ weight.T for segment, weight in zip(segments, weights, strict=True)])
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (tokens, local, replica))
    expected_grads = torch.autograd.grad(expected.sum(), (tokens, local, replica))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("counts", [[2, 3, 0, 0], [0, 0, 2, 3]])
def test_segmented_gmm_no_grad_reuses_single_output(counts):
    tokens = torch.randn(5, 3)
    local = torch.randn(2, 5, 3)
    replica = torch.randn(2, 5, 3)
    outputs = []

    def gemm(*args):
        output = _cpu_group_gemm(*args)
        outputs.append(output)
        return output

    with torch.no_grad():
        actual = npu_moe.npu_ep_memory.segmented_group_gemm(
            tokens, local, replica, torch.tensor(counts), sum(counts[:2]), gemm
        )
    assert len(outputs) == 1 and actual is outputs[0]
    assert actual.is_contiguous()


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Real NPU fused MoE balance requires torch_npu and Ascend hardware")
@pytest.mark.parametrize("optimize_memory", [False, True])
def test_real_npu_two_rank_alias_forward_backward_and_owner_gradient(optimize_memory):
    world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        pytest.skip("Run with torchrun --standalone --nproc-per-node=2 on an Ascend host.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("hccl")

    real_npu_moe = importlib.import_module(_NPU_MOE_NAME)
    real_ep_dispatch = importlib.import_module(_EP_DISPATCH_NAME)
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
    balancer = _RecordingBalancer(group, real_ep_load_balance, optimize_memory=optimize_memory)
    original_get_parallel_state = real_npu_moe.get_parallel_state
    original_ep_dispatch_get_parallel_state = real_ep_dispatch.get_parallel_state
    real_npu_moe.get_parallel_state = lambda: SimpleNamespace(ep_enabled=True, ep_group=group)
    real_ep_dispatch.get_parallel_state = lambda: SimpleNamespace(ep_group=group)
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
        expected_plan = real_ep_load_balance._build_ep_balance_plan_from_counts(
            selected_experts.cpu(), torch.tensor([[6, 0], [4, 0]]), 2, rank, 1
        )
        for name in ("replicas", "input_splits", "output_splits", "rank_loads_before", "rank_loads_after"):
            assert getattr(balancer.plan, name) == getattr(expected_plan, name)
        assert balancer.plan.selected_physical_experts.device == device
        assert torch.equal(balancer.plan.selected_physical_experts.cpu(), expected_plan.selected_physical_experts)
        assert torch.equal(
            balancer.plan.tokens_per_local_physical_expert.cpu(), expected_plan.tokens_per_local_physical_expert
        )

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
        real_ep_dispatch.get_parallel_state = original_ep_dispatch_get_parallel_state
        if owns_group:
            dist.destroy_process_group()
