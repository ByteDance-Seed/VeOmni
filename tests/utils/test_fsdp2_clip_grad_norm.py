import importlib
from types import SimpleNamespace

import torch

from veomni.distributed.fsdp2.clip_grad_norm import _fsdp_grad_norm_reduce_groups, _local_pth_sum


clip_grad_norm_module = importlib.import_module("veomni.distributed.fsdp2.clip_grad_norm")


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


def test_local_pth_sum_skips_missing_grads_and_accumulates_in_fp32(monkeypatch):
    monkeypatch.setattr(clip_grad_norm_module, "_LOCAL_NORM_CHUNK_SIZE", 2)
    p1 = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    p2 = torch.nn.Parameter(torch.tensor([3.0]))
    p3 = torch.nn.Parameter(torch.tensor([5.0]))
    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = None
    p3.grad = torch.tensor([12.0])

    actual = _local_pth_sum([p1, p2, p3], p=2.0)

    assert actual.dtype == torch.float32
    assert torch.equal(actual.cpu(), torch.tensor(169.0))


def test_local_pth_sum_falls_back_without_foreach_support(monkeypatch):
    monkeypatch.setattr(clip_grad_norm_module, "_has_foreach_support", lambda tensors, device: False)
    monkeypatch.setattr(clip_grad_norm_module, "_device_has_foreach_support", lambda device: False)

    def fail_foreach_norm(*args, **kwargs):
        raise AssertionError("foreach path should not be used")

    monkeypatch.setattr(torch, "_foreach_norm", fail_foreach_norm)
    p1 = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    p2 = torch.nn.Parameter(torch.tensor([3.0]))
    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = torch.tensor([12.0])

    actual = _local_pth_sum([p1, p2], p=2.0)

    assert actual.dtype == torch.float32
    assert torch.equal(actual.cpu(), torch.tensor(169.0))
