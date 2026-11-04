# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""TP expert checkpoint errors must precede writes and reach every rank."""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.distributed.parallel_plan import SpecInfo
from veomni.distributed.parallel_state import ParallelState
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


def _model(kind):
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(2, 2, bias=False)
    if kind == "dense":
        return model
    if kind == "registered_ep":
        from torch.distributed.tensor import Shard

        model._fqn2spec_info = {"dense.weight": SpecInfo("ep", Shard(0), "dense.weight", None)}
        return model
    model.experts = torch.nn.Module()
    if kind == "buffer":
        model.experts.register_buffer("weight", torch.ones(2))
    elif kind == "alias":
        model.experts.weight = model.dense.weight
        model.dense.weight.requires_grad_(False)
    else:
        model.experts.weight = torch.nn.Parameter(torch.ones(2), requires_grad=kind != "frozen")
    return model


@pytest.mark.parametrize("kind", ["parameter", "frozen", "alias", "buffer", "registered_ep"])
@pytest.mark.parametrize("mode", ["sync", "async", "stage"])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_tp_expert_save_rejected_before_writes(tmp_path, kind, mode):
    checkpoint = tmp_path / "checkpoint"
    marker = checkpoint / "model" / "ckpt" / ".metadata"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"preserve")
    stage = tmp_path / "stage" if mode == "stage" else None
    with (
        patch.object(DistributedCheckpointer, "_remove_dcp_markers", side_effect=AssertionError("writer started")),
        patch("veomni.checkpoint.dcp_checkpointer._prepare_stage_dir", side_effect=AssertionError("staging started")),
    ):
        with pytest.raises(RuntimeError, match="tensor.parallel.*expert"):
            DistributedCheckpointer.save(
                str(checkpoint),
                {"model": _model(kind)},
                save_async=mode == "async",
                trainable_only=kind in ("frozen", "alias"),
                stage_dir=str(stage) if stage else None,
                parallel_state=SimpleNamespace(tp_size=2, dp_mode="ddp"),
            )
    assert marker.read_bytes() == b"preserve"
    assert stage is None or not stage.exists()


@pytest.mark.parametrize("replicated_ep", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_tp_dense_model_still_saves_and_loads(tmp_path, replicated_ep):
    parallel = SimpleNamespace(tp_size=2, dp_mode="ddp")
    model = _model("dense")
    if replicated_ep:
        from torch.distributed.tensor import Replicate

        model._fqn2spec_info = {"dense.weight": SpecInfo("ep", Replicate(), "dense.weight", None)}
    path = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(path, {"model": model}, parallel_state=parallel)
    restored = _model("dense")
    DistributedCheckpointer.load(path, {"model": restored}, parallel_state=parallel)
    torch.testing.assert_close(restored.dense.weight, model.dense.weight, rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_tp_expert_target_can_load_canonical_checkpoint(tmp_path):
    model = _model("parameter")
    path = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(path, {"model": model}, parallel_state=ParallelState(dp_mode="ddp"))
    restored = _model("parameter")
    DistributedCheckpointer.load(path, {"model": restored}, parallel_state=SimpleNamespace(tp_size=2, dp_mode="ddp"))
    torch.testing.assert_close(restored.experts.weight, model.experts.weight, rtol=0, atol=0)


def _distributed_worker(rank, root, accelerator, endpoint):
    if accelerator:
        device = get_device_type()
        get_torch_device().set_device(f"{device}:{rank}")
        backend = get_dist_comm_backend()
    else:
        device, backend = "cpu", "gloo"
    dist.init_process_group(backend, store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        model = _model("parameter" if rank == 0 else "dense")
        expected = "tensor.parallel.*expert" if rank == 0 else "checking canonical ExtraParallel optimizer state"
        with patch.object(
            DistributedCheckpointer, "_remove_dcp_markers", side_effect=AssertionError("writer started")
        ):
            with pytest.raises(RuntimeError, match=expected):
                DistributedCheckpointer.save(
                    root, {"model": model}, parallel_state=SimpleNamespace(tp_size=2, dp_mode="ddp")
                )
        done = torch.ones(1, device=device)
        dist.all_reduce(done)
        assert done.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_tp_expert_failure_is_coordinated_before_writes(tmp_path):
    path = tmp_path / "checkpoint"
    spawn_with_rendezvous(_distributed_worker, args=(str(path), False), nprocs=2)
    assert not path.exists()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="accelerator"))
def test_native_accelerator_tp_expert_checkpoint_preflight(tmp_path):
    device = get_device_type()
    available = device != "cpu" and get_torch_device().is_available() and get_torch_device().device_count() >= 2
    require_npu = os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1" or (
        "VEOMNI_CI_CAPABILITIES" in os.environ and runner_capabilities().backend == "npu"
    )
    if not available or (require_npu and device != "npu"):
        if require_npu:
            pytest.fail("The NPU checkpoint preflight requires two NPU devices")
        pytest.skip("requires two accelerators")
    path = tmp_path / "checkpoint"
    spawn_with_rendezvous(_distributed_worker, args=(str(path), True), nprocs=2)
    assert not path.exists()
