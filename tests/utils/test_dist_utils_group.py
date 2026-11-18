"""Ensure error propagation follows the backend of the actual process group."""

import os
from unittest.mock import patch

import pytest
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.dist_utils import all_reduce, any_rank_failed, raise_if_any_rank_failed


def _all_reduce_worker(rank, world_size, endpoint):
    """`all_reduce` must allocate for the backend that runs it, not for the host.

    Two properties, both learned from a failure rather than guessed at:

    * A Gloo group on an accelerator host reduces on CPU. Allocating on
      `get_device_type()` there raises `No backend type associated with device
      type npu` on Ascend (and its CUDA equivalent) because Gloo has no
      counterpart for an accelerator tensor -- so the default-Gloo test above and
      every single-process unit test that needs a real reduction hit it.
    * `mean` cannot take an integer tensor. The reduction is `SUM` followed by an
      in-place division, which cannot store a fraction in an integer tensor, so
      an integer count under `op="mean"` used to raise
      `result type Float can't be cast to the desired output type Long`. Integer
      counts still travel as int64 for `max`/`min`/`sum`, which is what the HCCL
      MAX/MIN path needs.
    """
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=world_size)
    try:
        with patch("veomni.utils.dist_utils.get_device_type", return_value="cuda"):
            # The host claims an accelerator; the group reduces on CPU. If the
            # allocation followed the host this would raise instead of returning.
            assert all_reduce(rank + 1, op="sum") == world_size * (world_size + 1) // 2
            assert all_reduce(rank + 1, op="max") == world_size
            assert all_reduce([rank + 1, rank + 2], op="sum") == [
                world_size * (world_size + 1) // 2,
                world_size * (world_size + 1) // 2 + world_size,
            ]

            # Integer input under `mean` keeps its value and stays a fraction.
            mean = (world_size + 1) / 2
            assert all_reduce(rank + 1, op="mean") == pytest.approx(mean)
            # A single-element list collapses to a scalar, as every op already did.
            assert all_reduce([rank + 1], op="mean") == pytest.approx(mean)
            assert all_reduce([rank + 1, rank + 1], op="mean") == [pytest.approx(mean)] * 2
            # A float input keeps the old path untouched.
            assert all_reduce(float(rank + 1), op="mean") == pytest.approx((world_size + 1) / 2)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_all_reduce_follows_the_group_backend_and_mean_accepts_integer_counts():
    spawn_with_rendezvous(_all_reduce_worker, args=(2,), nprocs=2)


def _default_gloo_worker(rank, world_size, endpoint):
    """Check failure propagation uses CPU for default and explicit Gloo groups."""
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


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_default_gloo_group_uses_cpu_even_when_accelerator_is_available():
    spawn_with_rendezvous(_default_gloo_worker, args=(2,), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_any_rank_failed_without_process_group():
    assert any_rank_failed(True)
    assert not any_rank_failed(False)


def _accelerator_group_worker(rank, world_size, endpoint):
    """Check failure propagation uses the assigned accelerator backend."""
    device = get_torch_device()
    device.set_device(rank)
    backend = get_dist_comm_backend()
    dist.init_process_group(backend, store=endpoint.connect(), rank=rank, world_size=world_size)
    try:
        assert dist.get_backend() == backend
        assert any_rank_failed(rank == 0)
        assert not any_rank_failed(False)
    finally:
        dist.destroy_process_group()


def _require_two_accelerators():
    """Allow optional local skips, but fail when accelerator CI is misconfigured."""
    capabilities = runner_capabilities() if os.environ.get("VEOMNI_CI_CAPABILITIES") else None
    assigned = capabilities is not None and capabilities.backend != "cpu"
    device_type = get_device_type()
    if device_type == "cpu" or not get_torch_device().is_available() or get_torch_device().device_count() < 2:
        if assigned:
            pytest.fail("Assigned accelerator runtime must provide two usable devices")
        pytest.skip("Requires two accelerator devices")
    if assigned and device_type != capabilities.backend:
        pytest.fail("Assigned accelerator backend does not match the runtime")


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="accelerator"))
def test_default_accelerator_group_failure_propagation():
    _require_two_accelerators()
    spawn_with_rendezvous(_accelerator_group_worker, args=(2,), nprocs=2)
