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

import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from veomni.distributed.moe import ep_load_balance
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state
from veomni.ops.kernels.moe_experts.standard import npu
from veomni.utils.device import IS_NPU_AVAILABLE, get_torch_device


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


@pytest.fixture(scope="module")
def real_npu_ep_group():
    world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        pytest.skip("Run with torchrun --standalone --nproc-per-node=2 on an Ascend host.")
    get_torch_device().set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("hccl")
    try:
        # Reuse the communicator across parameterized cases. Reinitializing an
        # HCCL default group against the same torchrun store can reuse stale keys.
        yield dist.group.WORLD
    finally:
        if owns_group:
            dist.destroy_process_group()
            clear_parallel_state()


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Real singleton FSDP2 requires Ascend hardware")
@pytest.mark.parametrize("reduce_dtype", [torch.float32, torch.bfloat16])
def test_real_npu_singleton_fsdp_gradient_accumulation(real_npu_ep_group, reduce_dtype):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from veomni.distributed.fsdp2.grad_scaling import configure_singleton_gradient_scaling

    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    mesh = init_device_mesh("npu", (dist.get_world_size(real_npu_ep_group), 1), mesh_dim_names=("ep", "efsdp"))[
        "efsdp"
    ]
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=reduce_dtype)
    reference = nn.Sequential(*(nn.Linear(32, 32, device=device, dtype=torch.float32) for _ in range(2)))
    candidate = nn.Sequential(*(nn.Linear(32, 32, device=device, dtype=torch.float32) for _ in range(2)))
    candidate.load_state_dict(reference.state_dict())
    for native_layer, hooked_layer in zip(reference, candidate, strict=True):
        fully_shard(native_layer, mesh=mesh, mp_policy=policy)
        fully_shard(hooked_layer, mesh=mesh, mp_policy=policy)
        native_divide = (
            getattr(native_layer, "set_gradient_divide_factor", None) or native_layer.set_reduce_scatter_divide_factor
        )
        native_divide(8)
        assert configure_singleton_gradient_scaling(
            hooked_layer, mesh_size=1, reduce_dtype=reduce_dtype, gradient_divide_factor=8
        )
    fully_shard(reference, mesh=mesh, mp_policy=policy)
    fully_shard(candidate, mesh=mesh, mp_policy=policy)

    generator = torch.Generator().manual_seed(17 + dist.get_rank(real_npu_ep_group))
    for _ in range(2):
        inputs = torch.randn(4, 32, generator=generator).to(device=device, dtype=torch.bfloat16)
        reference_inputs = inputs.clone().requires_grad_()
        candidate_inputs = inputs.clone().requires_grad_()
        upstream = torch.randn(4, 32, generator=generator).to(device=device, dtype=torch.bfloat16)
        saved = upstream.clone()
        reference_output, candidate_output = reference(reference_inputs), candidate(candidate_inputs)
        torch.testing.assert_close(candidate_output, reference_output, rtol=0, atol=0)
        reference_output.backward(upstream)
        candidate_output.backward(upstream)
        get_torch_device().synchronize()
        for actual, expected in zip(candidate.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad.to_local(), expected.grad.to_local(), rtol=0, atol=0)
        torch.testing.assert_close(candidate_inputs.grad, reference_inputs.grad, rtol=0, atol=0)
        torch.testing.assert_close(upstream, saved, rtol=0, atol=0)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Real NPU fused MoE balance requires torch_npu and Ascend hardware")
@pytest.mark.parametrize("optimize_memory", [False, True])
@pytest.mark.parametrize("mixed_routing", [False, True])
def test_real_npu_two_rank_alias_forward_backward_and_owner_gradient(
    optimize_memory, mixed_routing, real_npu_ep_group
):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    parallel = _init_parallel_state(
        dp_size=2, dp_shard_size=2, device_type="npu", extra_parallel_sizes=(2,), name="npu_replica_test"
    )
    group = parallel.ep_group
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
    if mixed_routing:
        selected_experts[-1] = 1
    hidden_states.requires_grad_(True)
    routing_weights.requires_grad_(True)
    weights = _ExpertWeights(rank, dtype=torch.bfloat16, device=device)
    reference_weights = _ExpertWeights(rank, dtype=torch.bfloat16, device=device)
    reference_hidden = hidden_states.detach().clone().requires_grad_(True)
    reference_routing = routing_weights.detach().clone().requires_grad_(True)
    parameter_names_before = tuple(dict(weights.named_parameters()))
    balancer = _RecordingBalancer(group, ep_load_balance, optimize_memory=optimize_memory)
    with use_parallel_state(parallel):
        reference = npu.npu_fused_moe_forward(
            2,
            reference_routing,
            selected_experts,
            reference_hidden,
            None,
            None,
            reference_weights.down,
            reference_weights.gate_up,
        )
        reference.float().sum().backward()
        output = npu.npu_fused_moe_forward(
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
        replica_tokens = 3 if mixed_routing else 5
        assert int(balancer.plan.tokens_per_local_physical_expert[:, -1].sum().item()) == (
            0 if rank == 0 else replica_tokens
        )
        expected_plan = ep_load_balance._build_ep_balance_plan_from_counts(
            selected_experts.cpu(), torch.tensor([[5, 1], [3, 1]] if mixed_routing else [[6, 0], [4, 0]]), 2, rank, 1
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
        for actual, expected in (
            (output, reference),
            (hidden_states.grad, reference_hidden.grad),
            (routing_weights.grad, reference_routing.grad),
            (weights.gate_up.grad, reference_weights.gate_up.grad),
            (weights.down.grad, reference_weights.down.grad),
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
        assert tuple(dict(weights.named_parameters())) == parameter_names_before


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Real NPU grouped matmul requires Ascend hardware")
@pytest.mark.parametrize(
    "in_dim,out_dim,backward", [(2048, 1024, False), (512, 2048, False), (1024, 2048, True), (2048, 512, True)]
)
@pytest.mark.parametrize("replicas", [1, 8])
def test_real_npu_mixed_segment_one_output_matches_fallback(in_dim, out_dim, backward, replicas):
    import torch_npu

    from veomni.ops.kernels.moe_experts.standard.npu import npu_group_gemm
    from veomni.ops.kernels.moe_experts.standard.npu_ep_memory import segmented_group_gemm

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    get_torch_device().set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    torch.manual_seed(761)
    local_counts = [0 if j % 11 == 0 else 400 + 5 * j for j in range(32)]
    replica_counts = [117] if replicas == 1 else [0, 700, 210, 0, 380, 0, 90, 1]
    shape = (in_dim, out_dim) if backward else (out_dim, in_dim)
    local = torch.randn(32, *shape, dtype=torch.bfloat16).to(device)
    replica = torch.randn(replicas, *shape, dtype=torch.bfloat16).to(device)
    if backward:
        local, replica = local.transpose(1, 2), replica.transpose(1, 2)
    tokens = torch.randn(sum(local_counts + replica_counts), in_dim, dtype=torch.bfloat16).to(device)
    counts = torch.tensor(local_counts + replica_counts, dtype=torch.long, device=device)
    local_tokens = sum(local_counts)
    with torch.no_grad():
        expected = torch.cat(
            (
                npu_group_gemm(tokens[:local_tokens], local.transpose(1, 2), counts[:32]),
                npu_group_gemm(tokens[local_tokens:], replica.transpose(1, 2), counts[32:]),
            )
        )
        with patch.object(torch_npu, "npu_grouped_matmul", wraps=torch_npu.npu_grouped_matmul) as calls:
            actual = segmented_group_gemm(tokens, local, replica, counts, local_tokens, npu_group_gemm)
            calls.assert_called_once()
            assert all(weight.ndim == 2 for weight in calls.call_args.args[1])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.stride() == expected.stride()
        assert torch_npu.get_npu_format(actual) == torch_npu.get_npu_format(expected)
