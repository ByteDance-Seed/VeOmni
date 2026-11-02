"""Dense-reference equality for occurrence routing and real Gloo replica flows."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


_PATH = Path(__file__).parents[2] / "veomni/distributed/moe/moe_utils.py"
_SPEC = importlib.util.spec_from_file_location("compact_moe_utils", _PATH)
utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(utils)


@pytest.mark.parametrize("size", [0, 1, 7])
@pytest.mark.parametrize("topk", [1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_compact_helpers_preserve_dense_order_values_and_gradients(size, topk, dtype):
    torch.manual_seed(19)
    experts, hidden = 8, 5
    ids = (
        torch.stack([torch.randperm(experts)[:topk] for _ in range(size)])
        if size
        else torch.empty(0, topk, dtype=torch.long)
    )
    x = torch.randn(size, hidden).to(dtype).requires_grad_()
    weights = torch.randn(size, topk).to(dtype).requires_grad_()
    scale = torch.randn(experts, hidden).to(dtype).requires_grad_()
    mask = F.one_hot(ids, experts).permute(2, 1, 0).sum(1)
    dense_tokens, dense_order = utils.permute(x, mask)
    compact_tokens, order = utils.compact_permute(x, ids)
    assert torch.equal(compact_tokens, dense_tokens)
    assert torch.equal(order // topk, dense_order)
    sorted_experts = ids.reshape(-1).index_select(0, order)
    dense_expert_output = dense_tokens * scale.index_select(0, sorted_experts)
    compact_expert_output = compact_tokens * scale.index_select(0, sorted_experts)
    dense = utils.unpermute(
        dense_expert_output, utils.generate_weights_idx(weights, ids, experts), x.shape, dense_order, mask
    )
    compact = utils.compact_unpermute(compact_expert_output, weights, x.shape, order)
    assert torch.equal(dense, compact)
    grad = torch.randn_like(dense)
    dense_grads = torch.autograd.grad(dense, (x, weights, scale), grad, retain_graph=True)
    compact_grads = torch.autograd.grad(compact, (x, weights, scale), grad)
    assert all(torch.equal(a, b) for a, b in zip(dense_grads, compact_grads, strict=True))


def test_compact_combine_preserves_expert_order_not_topk_slot_order():
    ids = torch.tensor([[2, 0, 1]])
    order = torch.argsort(ids.reshape(-1), stable=True)
    values = torch.tensor([[-100139008.0], [1.0], [100139008.0]], dtype=torch.bfloat16)
    actual = utils.compact_unpermute(values, torch.ones(1, 3, dtype=torch.bfloat16), (1, 1), order)
    assert actual.item() == 0.0
    inverse = torch.argsort(order)
    assert values[inverse].float().sum().item() == 1.0


def _compact_worker(rank, world_size, rendezvous, dtype, duplicate_logical):
    # Reuse the established CPU-only import seam; production planner, token
    # collectives and replica autograd are retained unchanged.
    from tests.parallel.test_moe_ep_load_balance_dispatch import ep_dispatch
    from tests.parallel.test_moe_ep_load_balance_dispatch import ep_load_balance as ep
    from tests.parallel.test_moe_ep_load_balance_dispatch import moe_layer as moe

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        experts = 4 if duplicate_logical else 8
        rows = (1 if rank == 0 else 0) if duplicate_logical else (0, 7, 4, 3)[rank]
        selected = torch.zeros(rows, 2, dtype=torch.long)
        if not duplicate_logical:
            selected[:, 1] = 1
            selected[::2] = selected[::2].flip(-1)
        torch.manual_seed(101 + rank)
        original_x = torch.randn(rows, 3, dtype=dtype)
        original_weights = torch.randn(rows, 2, dtype=dtype)
        original_gate = torch.randn(experts // world_size, 4, 3, dtype=dtype)
        original_down = torch.randn(experts // world_size, 3, 2, dtype=dtype)
        balancer = SimpleNamespace(ep_group=dist.group.WORLD, layer_index=0)
        balancer.build_plan = lambda selected: ep.build_ep_balance_plan(
            selected, experts, dist.group.WORLD, max_replicas_per_rank=1 if duplicate_logical else 2
        )
        original_pre = moe.token_pre_all2all
        original_apply = moe.EPMergedFc1GroupGemm.apply
        original_state = ep_dispatch.get_parallel_state

        def eager_experts(tokens, cumsum, gate_up, down, limit):
            assert limit is None
            outputs = []
            start = 0
            for index, end in enumerate(cumsum.tolist()):
                gate, up = F.linear(tokens[start:end], gate_up[index]).chunk(2, dim=-1)
                outputs.append(F.linear(F.silu(gate) * up, down[index]))
                start = end
            return torch.cat(outputs, dim=0)

        def dense_pre(**kwargs):
            ids = kwargs.pop("selected_experts")
            kwargs["expert_mask"] = F.one_hot(ids, kwargs["num_experts"]).permute(2, 1, 0)
            return original_pre(**kwargs)

        moe.EPMergedFc1GroupGemm.apply = staticmethod(eager_experts)
        ep_dispatch.get_parallel_state = lambda: SimpleNamespace(ep_group=dist.group.WORLD)
        results = []
        try:
            for compact in (False, True):
                x, weights, gate, down = [
                    t.detach().clone().requires_grad_()
                    for t in (original_x, original_weights, original_gate, original_down)
                ]
                moe.token_pre_all2all = original_pre if compact else dense_pre
                output = moe.dispatch_to_ep_class(
                    moe.EPMergedFc1GroupGemm, experts, weights, selected, x, gate, down, None, load_balancer=balancer
                )
                output.float().square().sum().backward()
                results.append((output.detach(), *(t.grad for t in (x, weights, gate, down))))
                dist.barrier()
            assert all(torch.equal(a, b) for a, b in zip(*results, strict=True))
        finally:
            moe.token_pre_all2all = original_pre
            moe.EPMergedFc1GroupGemm.apply = original_apply
            ep_dispatch.get_parallel_state = original_state
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("duplicate_logical", [False, True])
def test_compact_replica_dispatch_preserves_dense_values_and_gradients(tmp_path, dtype, duplicate_logical):
    mp.spawn(
        _compact_worker, args=(4, f"file://{tmp_path / 'compact'}", dtype, duplicate_logical), nprocs=4, join=True
    )
