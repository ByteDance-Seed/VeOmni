from __future__ import annotations

import importlib.util
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


_EAGER_PATH = Path(__file__).resolve().parents[2] / "veomni" / "ops" / "kernels" / "load_balancing_loss" / "eager.py"
_SPEC = importlib.util.spec_from_file_location("_veomni_load_balancing_loss_eager_test", _EAGER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_EAGER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EAGER)
load_balancing_loss_pytorch = _EAGER.load_balancing_loss_pytorch


def _independent_reference(
    gate_logits: tuple[torch.Tensor, ...],
    num_experts: int,
    top_k: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    logits = torch.cat(gate_logits, dim=0)
    probs = torch.softmax(logits.float(), dim=-1)
    selected = torch.topk(probs, top_k, dim=-1).indices
    mask = attention_mask.reshape(-1).to(dtype=probs.dtype)
    expert_count = torch.nn.functional.one_hot(selected, num_experts).to(probs.dtype)
    expert_count = (expert_count * mask[:, None, None]).sum(dim=(0, 1))
    router_prob_sum = (probs * mask[:, None]).sum(dim=0)
    total_weight = mask.sum()
    return torch.dot(expert_count, router_prob_sum) * num_experts / total_weight.square()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _case_inputs(case: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    features = [
        torch.tensor([[2.0, 0.0, 0.1], [1.5, 0.2, -0.1], [2.2, -0.1, 0.0]], dtype=torch.float64),
        torch.tensor([[-0.2, 0.1, 2.0], [0.0, 0.0, 1.0], [0.3, 0.0, 0.8]], dtype=torch.float64),
    ]
    if case == "asymmetric":
        masks = [torch.tensor([[1, 1, 1]]), torch.tensor([[1, 0, 0]])]
    elif case == "empty_rank":
        masks = [torch.tensor([[0, 0, 0]]), torch.tensor([[1, 1, 0]])]
    else:
        raise AssertionError(f"unknown case {case}")
    return features, masks


def _worker(rank: int, world_size: int, port: int, case: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        features, masks = _case_inputs(case)
        initial_weight = torch.tensor(
            [[1.3, -0.5, 0.2, -0.8], [0.1, 0.7, -0.6, -0.2], [-0.7, 0.0, 1.4, 0.3]],
            dtype=torch.float64,
        )

        weight = initial_weight.clone().requires_grad_(True)
        local_logits = features[rank] @ weight
        distributed_loss = load_balancing_loss_pytorch(
            (local_logits,),
            num_experts=4,
            top_k=2,
            attention_mask=masks[rank],
            group=dist.group.WORLD,
        )

        oracle_weight = initial_weight.clone().requires_grad_(True)
        oracle_logits = torch.cat(features, dim=0) @ oracle_weight
        oracle_mask = torch.cat(masks, dim=1)
        oracle_loss = load_balancing_loss_pytorch(
            (oracle_logits,),
            num_experts=4,
            top_k=2,
            attention_mask=oracle_mask,
        )
        torch.testing.assert_close(distributed_loss, oracle_loss, rtol=0, atol=1e-12)

        distributed_loss.backward()
        oracle_loss.backward()
        assert weight.grad is not None
        assert oracle_weight.grad is not None
        fsdp_averaged_grad = weight.grad.clone()
        dist.all_reduce(fsdp_averaged_grad)
        fsdp_averaged_grad /= world_size
        torch.testing.assert_close(fsdp_averaged_grad, oracle_weight.grad, rtol=0, atol=1e-12)

        if case == "asymmetric":
            naive_local_loss = load_balancing_loss_pytorch(
                (local_logits.detach(),),
                num_experts=4,
                top_k=2,
                attention_mask=masks[rank],
            )
            naive_average = naive_local_loss.detach().clone()
            dist.all_reduce(naive_average)
            naive_average /= world_size
            assert not torch.isclose(naive_average, oracle_loss.detach(), rtol=1e-6, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["asymmetric", "empty_rank"])
def test_unified_sp_aux_loss_matches_monolithic_value_and_shared_weight_grad(case: str):
    mp.spawn(_worker, args=(2, _free_port(), case), nprocs=2, join=True)


def test_group_none_preserves_hf_single_rank_value_and_grad():
    torch.manual_seed(17)
    logits = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1]])

    actual = load_balancing_loss_pytorch((logits,), 4, 2, mask, group=None)
    expected = _independent_reference((reference_logits,), 4, 2, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=5e-7)

    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-6, atol=5e-7)
