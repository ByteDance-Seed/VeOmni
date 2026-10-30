"""Ensure error propagation follows the backend of the actual process group."""

from unittest.mock import patch

import pytest
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.utils.dist_utils import any_rank_failed, raise_if_any_rank_failed


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))


def _default_gloo_worker(rank, world_size, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=world_size)
    try:
        # A host may also expose CUDA/NPU devices while the default group is
        # Gloo; selecting a device from hardware availability would be wrong.
        with patch("veomni.utils.dist_utils.get_device_type", return_value="cuda"):
            assert any_rank_failed(rank == 0)
            assert not any_rank_failed(False)
            expected = ValueError if rank == 0 else RuntimeError
            message = "rank 0" if rank == 0 else "checking a distributed action"
            with pytest.raises(expected, match=message):
                raise_if_any_rank_failed(ValueError("rank 0") if rank == 0 else None, "checking a distributed action")
            group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
            assert any_rank_failed(rank == 1, group=group)
    finally:
        dist.destroy_process_group()


def test_default_gloo_group_uses_cpu_even_when_accelerator_is_available():
    spawn_with_rendezvous(_default_gloo_worker, args=(2,), nprocs=2)


def test_any_rank_failed_without_process_group():
    assert any_rank_failed(True)
    assert not any_rank_failed(False)
