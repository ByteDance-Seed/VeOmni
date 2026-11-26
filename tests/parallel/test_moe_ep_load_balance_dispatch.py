"""Real Gloo dispatch, replica exchange and validation behind both fused rows."""

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.distributed.moe.ep_buffer import ReplicaBufferPool
from veomni.distributed.moe.ep_load_balance import _Qwen3_5MoEEPLoadBalancer, cat_local_and_replica_weights
from veomni.distributed.moe.moe_utils import _OrderedAliasGather, alias_permute
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state
from veomni.ops.kernels.moe_experts.standard import npu, triton


_CPU_2_RANKS = LaunchSpec(nproc=2, device="cpu")


def test_ordered_alias_gather_preserves_duplicate_token_gradient_order():
    generator = torch.Generator().manual_seed(372)
    tokens = torch.randn(64, 12, generator=generator, dtype=torch.bfloat16).T.requires_grad_()
    reference_tokens = tokens.detach().clone().requires_grad_()
    logical = torch.randint(12, (12, 8), generator=generator)
    physical = torch.randint(16, logical.shape, generator=generator)
    reference, (logical_order, logical_to_physical) = alias_permute(reference_tokens, physical, logical)
    physical_order = torch.argsort(physical.flatten(), stable=True)
    actual = _OrderedAliasGather.apply(tokens, logical_order // 8, physical_order // 8, logical_to_physical)
    gradient = torch.randn(actual.shape, generator=generator, dtype=torch.bfloat16)
    actual.backward(gradient)
    reference.backward(gradient)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(tokens.grad, reference_tokens.grad, rtol=0, atol=0)


def _reference_mlp(tokens, gates, downs):
    projected = torch.bmm(gates, tokens.unsqueeze(-1)).squeeze(-1)
    gate, up = projected.chunk(2, dim=-1)
    return torch.bmm(downs, (F.silu(gate) * up).unsqueeze(-1)).squeeze(-1)


def _cpu_experts(tokens, gate, down, plan, group, gate_transfer, down_transfer, memory, limit, activation):
    # Keep Gloo exchange and the actual replica ownership path. Only substitute
    # the accelerator-specific GEMM for a reference tensor implementation.
    assert plan.rank_loads_before == (16, 0) and plan.rank_loads_after == (8, 8)
    assert limit is None and activation == "silu"
    assert gate_transfer is not None and down_transfer is not None
    gates = cat_local_and_replica_weights(gate, gate_transfer.wait(), plan, group)
    downs = cat_local_and_replica_weights(down, down_transfer.wait(), plan, group)
    counts = plan.tokens_per_local_physical_expert.sum(dim=0)
    result = _reference_mlp(tokens, gates.repeat_interleave(counts, dim=0), downs.repeat_interleave(counts, dim=0))
    if memory:
        gate_transfer.record_stream_use()
        down_transfer.record_stream_use()
    return result


def _worker(rank, rendezvous, backend_name, memory):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        state = _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu", extra_parallel_sizes=(2,))
        backend = {"npu": npu, "triton": triton}[backend_name]
        with use_parallel_state(state), patch.object(backend, "_balanced_experts", _cpu_experts):
            controller = _Qwen3_5MoEEPLoadBalancer(
                ep_group=state.ep_group,
                num_experts=4,
                max_replicas_per_rank=1,
                layer_index=0,
                buffer_pool=ReplicaBufferPool() if memory else None,
            )
            all_tokens = torch.arange(32, dtype=torch.float32).reshape(16, 2).mul_(0.01).requires_grad_()
            all_gate = torch.arange(32, dtype=torch.float32).reshape(4, 4, 2).mul_(0.01).requires_grad_()
            all_down = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2).mul_(0.01).requires_grad_()
            all_routing = torch.full((16, 1), 0.75, requires_grad=True)
            token_slice = slice(8 * rank, 8 * (rank + 1))
            expert_slice = slice(2 * rank, 2 * (rank + 1))
            tokens, routing = [t[token_slice].detach().clone().requires_grad_() for t in (all_tokens, all_routing)]
            gate, down = [t[expert_slice].detach().clone().requires_grad_() for t in (all_gate, all_down)]
            ids = torch.zeros((8, 1), dtype=torch.long)
            empty = torch.empty(0)

            def actual(down_weight):
                return backend.wrapper(
                    tokens, routing, ids, empty, empty, down_weight, gate, num_experts=4, load_balancer=controller
                )

            # Invalid local weights propagate through the same histogram gather
            # on both public rows, before either rank can start replica sends.
            with pytest.raises(ValueError, match="same expert-row count|counts must be non-negative"):
                actual(down[:1] if rank else down)

            result = actual(down)
            result.sum().backward()
            expected = (
                _reference_mlp(all_tokens, all_gate[0].expand(16, -1, -1), all_down[0].expand(16, -1, -1))
                * all_routing
            )
            expected.sum().backward()
            torch.testing.assert_close(result, expected[token_slice])
            torch.testing.assert_close(tokens.grad, all_tokens.grad[token_slice])
            torch.testing.assert_close(routing.grad, all_routing.grad[token_slice])
            torch.testing.assert_close(gate.grad, all_gate.grad[expert_slice])
            torch.testing.assert_close(down.grad, all_down.grad[expert_slice])
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.parametrize("backend", ["npu", "triton"])
@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.launch_plan(provider=_CPU_2_RANKS)
def test_fused_row_dispatches_replicas_and_recovers_from_rank_local_validation(tmp_path, backend, memory):
    mp.spawn(
        _worker,
        args=((tmp_path / "balanced-dispatch").as_uri(), backend, memory),
        nprocs=_CPU_2_RANKS.nproc,
        join=True,
    )
