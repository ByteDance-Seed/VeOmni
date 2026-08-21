import pytest
import torch.nn as nn

from veomni.distributed.torch_parallelize import (
    _extra_parallel_fsdp_shard_size,
    _validate_fsdp_shard_divisibility,
)


class _FakeMeshDimension:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _FakeHSDPExtraParallelMesh:
    def __getitem__(self, dim_name: str) -> _FakeMeshDimension:
        assert dim_name == "ep_fsdp"
        return _FakeMeshDimension(2)


class _FakeParallelState:
    extra_parallel_fsdp_device_mesh = {"ep": _FakeHSDPExtraParallelMesh()}


def test_shard_size_excludes_hsdp_replicate_dimension() -> None:
    assert _extra_parallel_fsdp_shard_size(_FakeParallelState(), "ep") == 2


def test_nonzero_shard_rejects_indivisible_parameter_with_fqn() -> None:
    experts = nn.Linear(7, 16, bias=False)

    with pytest.raises(ValueError, match=r"decoder\.moe.*weight.*dim 1.*size 7.*4"):
        _validate_fsdp_shard_divisibility(
            experts,
            module_fqn="decoder.moe",
            shard_dim=1,
            shard_size=4,
        )


def test_nonzero_shard_accepts_even_parameter() -> None:
    experts = nn.Linear(8, 16, bias=False)

    _validate_fsdp_shard_divisibility(
        experts,
        module_fqn="decoder.moe",
        shard_dim=1,
        shard_size=4,
    )


def test_dim_zero_allows_uneven_parameter() -> None:
    experts = nn.Linear(8, 15, bias=False)

    _validate_fsdp_shard_divisibility(
        experts,
        module_fqn="decoder.moe",
        shard_dim=0,
        shard_size=4,
    )


def test_nonzero_shard_rejects_parameter_without_that_dimension() -> None:
    experts = nn.LayerNorm(8)

    with pytest.raises(ValueError, match=r"decoder\.moe.*weight.*has 1 dimensions.*dim 1"):
        _validate_fsdp_shard_divisibility(
            experts,
            module_fqn="decoder.moe",
            shard_dim=1,
            shard_size=4,
        )
