from types import SimpleNamespace

import pytest
import torch

from veomni.distributed.fsdp2.clip_grad_norm import _fsdp_grad_norm_reduce_groups, _local_pth_sum


def _parallel_state(
    *,
    dp_mode: str = "fsdp2",
    dp_replicate_enabled: bool = False,
    dp_shard_size: int = 1,
    dp_shard_sp_enabled: bool = False,
):
    return SimpleNamespace(
        dp_mode=dp_mode,
        dp_replicate_enabled=dp_replicate_enabled,
        dp_shard_size=dp_shard_size,
        dp_shard_sp_enabled=dp_shard_sp_enabled,
        dp_shard_group=object(),
        dp_shard_sp_group=object(),
        fsdp_group=object(),
    )


def test_fsdp_grad_norm_reduce_groups_use_fsdp_group_for_plain_fsdp2():
    ps = _parallel_state()

    assert _fsdp_grad_norm_reduce_groups(ps) == [("fsdp", ps.fsdp_group)]


def test_fsdp_grad_norm_reduce_groups_use_shard_group_for_hsdp():
    ps = _parallel_state(dp_replicate_enabled=True, dp_shard_size=4)

    assert _fsdp_grad_norm_reduce_groups(ps) == [("fsdp_shard", ps.dp_shard_group)]


def test_fsdp_grad_norm_reduce_groups_include_sp_for_hsdp_sp():
    ps = _parallel_state(dp_replicate_enabled=True, dp_shard_size=4, dp_shard_sp_enabled=True)

    assert _fsdp_grad_norm_reduce_groups(ps) == [("fsdp_shard_sp", ps.dp_shard_sp_group)]


def test_fsdp_grad_norm_reduce_groups_include_sp_for_replicated_sp_only_sharding():
    ps = _parallel_state(dp_replicate_enabled=True, dp_shard_size=1, dp_shard_sp_enabled=True)

    assert _fsdp_grad_norm_reduce_groups(ps) == [("fsdp_shard_sp", ps.dp_shard_sp_group)]


def test_fsdp_grad_norm_reduce_groups_skip_replicated_unsharded_fsdp2():
    ps = _parallel_state(dp_replicate_enabled=True, dp_shard_size=1)

    assert _fsdp_grad_norm_reduce_groups(ps) == []


def test_fsdp_grad_norm_reduce_groups_skip_non_fsdp2_modes():
    ps = _parallel_state(dp_mode="ddp")

    assert _fsdp_grad_norm_reduce_groups(ps) == []


def test_local_pth_sum_skips_missing_grads_and_accumulates_in_fp32():
    p1 = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    p2 = torch.nn.Parameter(torch.tensor([3.0]))
    p3 = torch.nn.Parameter(torch.tensor([5.0]))
    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = None
    p3.grad = torch.tensor([12.0])

    actual = _local_pth_sum([p1, p2, p3], p=2.0)

    assert actual.dtype == torch.float32
    assert torch.equal(actual.cpu(), torch.tensor(169.0))


def test_local_pth_sum_accumulates_bfloat16_gradients_in_fp32():
    p1 = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    p2 = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.bfloat16))
    p1.grad = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    p2.grad = torch.tensor([12.0], dtype=torch.bfloat16)

    actual = _local_pth_sum([p1, p2], p=2.0)

    assert actual.dtype == torch.float32
    assert torch.equal(actual.cpu(), torch.tensor(169.0))


def test_local_pth_sum_supports_float64_gradients():
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    param.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)

    actual = _local_pth_sum([param], p=2.0)

    assert actual.dtype == torch.float32
    assert torch.equal(actual.cpu(), torch.tensor(25.0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("p", [1.0, 2.0, 3.5])
def test_local_pth_sum_matches_previous_fp32_conversion(dtype, p):
    params = [
        torch.nn.Parameter(torch.zeros(4, dtype=dtype)),
        torch.nn.Parameter(torch.zeros(3, dtype=dtype)),
    ]
    params[0].grad = torch.tensor([0.25, -1.5, 2.0, -0.75], dtype=dtype)
    params[1].grad = torch.tensor([3.0, -0.5, 1.25], dtype=dtype)

    actual = _local_pth_sum(params, p)
    expected = sum(torch.linalg.vector_norm(param.grad.to(torch.float32), ord=p).pow(p) for param in params)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected.cpu())
