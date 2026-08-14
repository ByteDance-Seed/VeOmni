# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Two-NPU HCCL integration gate for lossless GDN context parallelism."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from veomni.distributed.context_parallel.gdn_kcp import (
    local_affine_summary,
    resolve_kcp_initial_state,
    unpack_affine_hm,
)
from veomni.distributed.context_parallel.gdn_lossless import (
    attach_state_dependency,
    compile_gdn_lossless_runtime_plan,
    make_state_participation,
    owned_to_physical,
    physical_to_owned,
    receive_initial_state,
    send_final_state,
)
from veomni.distributed.context_parallel.gdn_runtime import GdnCpOperation, make_gdn_cp_runtime_observer
from veomni.ops.kernels.gated_delta_rule.normalization import producer_dtype_l2norm
from veomni.ops.kernels.gdn_kcp_affine_ttx import ttx_bc8_m1_torch_reference
from veomni.utils.device import IS_NPU_AVAILABLE, get_torch_device

from ...tools.launch_utils import torchrun


_TOKENS = 128


def _has_two_npus() -> bool:
    if not IS_NPU_AVAILABLE:
        return False
    try:
        return bool(get_torch_device().is_available() and get_torch_device().device_count() >= 2)
    except (AttributeError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _has_two_npus(), reason="requires at least two Ascend NPUs")


def _physical_global_indices(plan) -> list[int]:
    """Map this rank's public zigzag layout back to valid global token ordinals."""
    indices: list[int] = []
    valid_offset = 0
    for valid_length, ring_length in zip(plan.global_plan.valid_lengths, plan.global_plan.ring_physical_lengths):
        padded = list(range(valid_offset, valid_offset + valid_length))
        padded.extend([-1] * (ring_length - valid_length))
        half = ring_length // (2 * plan.cp_size)
        first = plan.cp_rank
        second = 2 * plan.cp_size - 1 - plan.cp_rank
        indices.extend(padded[first * half : (first + 1) * half])
        indices.extend(padded[second * half : (second + 1) * half])
        valid_offset += valid_length
    if len(indices) != plan.local.source_token_count:
        raise AssertionError("physical index count does not match the compiled lossless plan")
    if any(index < 0 for index in indices):
        raise AssertionError("the aligned CP2 smoke must not contain physical padding")
    return indices


def _physical_shard(tensor: torch.Tensor, plan) -> torch.Tensor:
    index = torch.tensor(_physical_global_indices(plan), device=tensor.device, dtype=torch.long)
    return tensor.index_select(1, index).contiguous()


def _state_scan(tokens: torch.Tensor, initial_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state[0]
    outputs = []
    for token in tokens[0].unbind(dim=0):
        state = state * 0.875 + token
        outputs.append(state)
    return torch.stack(outputs, dim=0).unsqueeze(0), state.unsqueeze(0)


def _shared_state_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(2026081401)
    tokens = torch.randn(1, _TOKENS, 4, generator=generator, dtype=torch.float32) * 0.05
    weights = torch.randn(1, _TOKENS, 4, generator=generator, dtype=torch.float32) * 0.1
    return tokens.to(device), weights.to(device)


def _assert_forward_backward_observed(snapshot, operation: GdnCpOperation) -> None:
    phases = {
        event.phase
        for event in snapshot.events
        if event.operation == operation.value and event.enter > 0 and event.enter == event.exit and event.error == 0
    }
    assert phases == {"forward", "backward"}


def _run_state_passing_full_grad(plan, device: torch.device) -> None:
    observer = make_gdn_cp_runtime_observer("state_passing_lossless", plan=plan)
    global_input, global_weight = _shared_state_tensors(device)
    physical_input = _physical_shard(global_input, plan).detach().requires_grad_(True)
    physical_weight = _physical_shard(global_weight, plan)

    owned_input = physical_to_owned(
        physical_input,
        plan=plan,
        cp_group=dist.group.WORLD,
        observer=observer,
    )
    initial_state = receive_initial_state(
        plan=plan,
        cp_group=dist.group.WORLD,
        state_template=torch.zeros(1, 4, device=device, dtype=torch.float32),
        participation=make_state_participation(owned_input),
        observer=observer,
    )
    owned_output, final_state = _state_scan(owned_input, initial_state)
    sent_state = send_final_state(
        final_state,
        plan=plan,
        cp_group=dist.group.WORLD,
        observer=observer,
    )
    physical_output = owned_to_physical(
        owned_output,
        plan=plan,
        cp_group=dist.group.WORLD,
        observer=observer,
    )
    loss = attach_state_dependency((physical_output * physical_weight).sum(), sent_state)
    loss.backward()

    oracle_input = global_input.detach().clone().requires_grad_(True)
    oracle_output, _ = _state_scan(oracle_input, torch.zeros(1, 4, device=device, dtype=torch.float32))
    (oracle_output * global_weight).sum().backward()

    torch.testing.assert_close(physical_output, _physical_shard(oracle_output, plan), rtol=0, atol=1e-6)
    assert physical_input.grad is not None and oracle_input.grad is not None
    torch.testing.assert_close(physical_input.grad, _physical_shard(oracle_input.grad, plan), rtol=0, atol=1e-6)

    snapshot = observer.snapshot()
    operations = {event.operation for event in snapshot.events}
    assert GdnCpOperation.OWNERSHIP_A2A.value in operations
    state_operation = GdnCpOperation.STATE_P2P_SEND if plan.cp_rank == 0 else GdnCpOperation.STATE_P2P_RECV
    assert state_operation.value in operations
    _assert_forward_backward_observed(snapshot, GdnCpOperation.OWNERSHIP_A2A)
    _assert_forward_backward_observed(snapshot, state_operation)
    assert snapshot.balanced


def _shared_kcp_tensors(device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(2026081402)
    key = (torch.randn(1, _TOKENS, 1, 32, generator=generator) * 0.05).to(torch.bfloat16)
    value = (torch.randn(1, _TOKENS, 1, 32, generator=generator) * 0.05).to(torch.bfloat16)
    g = -torch.rand(1, _TOKENS, 1, generator=generator, dtype=torch.float32) * 0.05
    beta = torch.sigmoid(torch.randn(1, _TOKENS, 1, generator=generator)).to(torch.bfloat16)
    return tuple(tensor.to(device) for tensor in (key, value, g, beta))


def _run_kcp_full_grad(plan, device: torch.device) -> None:
    observer = make_gdn_cp_runtime_observer("kcp", plan=plan)
    global_inputs = _shared_kcp_tensors(device)
    physical_inputs = tuple(_physical_shard(tensor, plan).detach().requires_grad_(True) for tensor in global_inputs)
    owned_raw = tuple(
        physical_to_owned(
            tensor,
            plan=plan,
            cp_group=dist.group.WORLD,
            observer=observer,
        )
        for tensor in physical_inputs
    )
    owned_key = producer_dtype_l2norm(owned_raw[0])
    owned = (owned_key, *owned_raw[1:])
    cu_seqlens = torch.tensor(plan.local.owned_cu_seqlens, device=device, dtype=torch.int32)
    initial_state = resolve_kcp_initial_state(
        *owned,
        plan=plan,
        cp_group=dist.group.WORLD,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm=False,
        affine_impl="ttx_bc8_m1",
        observer=observer,
    )
    local_hm = local_affine_summary(
        *owned,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm=False,
        impl="ttx_bc8_m1",
    )
    local_he, local_matrix = unpack_affine_hm(local_hm, v_dim=32)
    final_state = torch.einsum("nhki,nhiv->nhkv", local_matrix, initial_state) + local_he
    terminal_owner = plan.local.successor_rank is None
    # Every CP rank must execute backward. The non-terminal zero objective adds
    # no local loss, but keeps its AG VJP ordinal live so the terminal rank's
    # gradient is routed back by deterministic reduce-scatter.
    loss = final_state.square().mean() if terminal_owner else final_state.square().mean() * 0
    loss.backward()

    oracle_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in global_inputs)
    oracle_key = producer_dtype_l2norm(oracle_inputs[0])
    oracle_hm = ttx_bc8_m1_torch_reference(
        oracle_key,
        *oracle_inputs[1:],
        use_qk_l2norm=False,
    )
    oracle_he, _oracle_matrix = unpack_affine_hm(oracle_hm, v_dim=32)
    # CP1 starts at S_init=0, hence its monolithic affine summary has
    # S_final = M @ 0 + he = he.
    oracle_he.square().mean().backward()

    if terminal_owner:
        torch.testing.assert_close(final_state, oracle_he, rtol=2e-3, atol=5e-4)
    for actual, expected in zip(physical_inputs, oracle_inputs):
        assert actual.grad is not None and expected.grad is not None
        torch.testing.assert_close(actual.grad, _physical_shard(expected.grad, plan), rtol=5e-2, atol=5e-3)

    snapshot = observer.snapshot()
    operations = {event.operation for event in snapshot.events}
    assert GdnCpOperation.OWNERSHIP_A2A.value in operations
    assert GdnCpOperation.KCP_AFFINE_AG.value in operations
    _assert_forward_backward_observed(snapshot, GdnCpOperation.OWNERSHIP_A2A)
    _assert_forward_backward_observed(snapshot, GdnCpOperation.KCP_AFFINE_AG)
    assert snapshot.observed_cp_ranks == (0, 1)
    assert snapshot.balanced


def _run_cp2_hccl_smoke() -> None:
    rank = dist.get_rank()
    device = torch.device("npu", rank)
    plan = compile_gdn_lossless_runtime_plan([_TOKENS], cp_group=dist.group.WORLD, ulysses_size=1)
    assert dist.get_backend() == "hccl"
    assert plan.cp_size == 2

    _run_state_passing_full_grad(plan, device)
    dist.barrier()
    _run_kcp_full_grad(plan, device)
    get_torch_device().synchronize()
    dist.barrier()


def test_cp2_hccl_lossless_state_and_kcp_match_shared_input_full_grad():
    """Exercise production ownership/state/KCP collectives against CP1 math."""
    torchrun(_run_cp2_hccl_smoke, world_size=2)
