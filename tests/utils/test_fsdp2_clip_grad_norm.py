from types import SimpleNamespace

from veomni.distributed.fsdp2.clip_grad_norm import _fsdp_grad_norm_reduce_groups


def _parallel_state(*, dp_mode: str = "fsdp2", dp_replicate_enabled: bool = False, dp_shard_size: int = 1):
    return SimpleNamespace(
        dp_mode=dp_mode,
        dp_replicate_enabled=dp_replicate_enabled,
        dp_shard_size=dp_shard_size,
        dp_shard_group=object(),
        fsdp_group=object(),
    )


def test_fsdp_grad_norm_reduce_groups_use_fsdp_group_for_plain_fsdp2():
    ps = _parallel_state()

    assert _fsdp_grad_norm_reduce_groups(ps) == [("fsdp", ps.fsdp_group)]


def test_fsdp_grad_norm_reduce_groups_use_shard_group_for_hsdp():
    ps = _parallel_state(dp_replicate_enabled=True, dp_shard_size=4)

    assert _fsdp_grad_norm_reduce_groups(ps) == [("fsdp_shard", ps.dp_shard_group)]


def test_fsdp_grad_norm_reduce_groups_skip_non_fsdp2_modes():
    ps = _parallel_state(dp_mode="ddp")

    assert _fsdp_grad_norm_reduce_groups(ps) == []
