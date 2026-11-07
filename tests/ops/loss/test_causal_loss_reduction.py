"""Real two-rank checks for the model wrapper's explicit CE denominator."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec


def _worker(rank, rendezvous, count, explicit_group, empty_rank):
    from veomni.models.loss_utils import cross_entropy_loss as loss_utils
    from veomni.ops import VeomniOp

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        loss_utils.get_parallel_state = lambda: type("State", (), {"sp_enabled": not explicit_group})()
        loss_utils.get_unified_sequence_parallel_group = lambda: dist.group.WORLD
        torch.manual_seed(127)
        global_hidden = torch.randn(2, 5, 3, requires_grad=True)
        global_weight = torch.randn(7, 3, requires_grad=True)
        targets = torch.tensor([[0, -100, -100, -100, 1], [2, 3, 4, 5, 6]])
        if empty_rank:
            targets[0].fill_(-100)
        denominator = (targets != -100).sum() if count is None else (1 if count == 0 else count)
        reference = (
            F.cross_entropy(
                F.linear(global_hidden, global_weight).reshape(-1, 7), targets.reshape(-1), reduction="sum"
            )
            / denominator
        )
        expected_hidden, expected_weight = torch.autograd.grad(reference * 2.75, (global_hidden, global_weight))
        hidden = global_hidden[rank : rank + 1].detach().clone().requires_grad_()
        weight = global_weight.detach().clone().requires_grad_()
        local_targets = targets[rank : rank + 1]
        actual, _, _ = loss_utils.ForCausalLMLoss(
            labels=torch.full_like(local_targets, -100),
            shift_labels=local_targets,
            hidden_states=hidden,
            weights=weight,
            num_items_in_batch=count,
            op=VeomniOp("cross_entropy_loss", "standard", "chunk_loss"),
            loss_reduction_group=dist.group.WORLD if explicit_group else None,
            chunk_size=2,
        )
        grad_hidden, grad_weight = torch.autograd.grad(actual * 2.75, (hidden, weight))
        dist.all_reduce(grad_weight)
        grad_weight /= 2
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(grad_weight, expected_weight)
        torch.testing.assert_close(grad_hidden, expected_hidden[rank : rank + 1] * 2)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("count", [None, 13, 0])
@pytest.mark.parametrize("explicit_group", [False, True])
@pytest.mark.parametrize("empty_rank", [False, True])
def test_causal_loss_shared_denominator_preserves_loss_and_gradients(tmp_path, count, explicit_group, empty_rank):
    mp.spawn(_worker, args=(str(tmp_path / "gloo_init"), count, explicit_group, empty_rank), nprocs=2, join=True)
