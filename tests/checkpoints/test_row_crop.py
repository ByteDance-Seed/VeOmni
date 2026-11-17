# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Logical-row checkpoint plans preserve padded runtime tensors."""

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.row_crop import RowCropLoadPlanner, RowCropSavePlanner


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _row_metadata_failure_worker(rank, directory, operation, endpoint):
    from datetime import timedelta
    from types import SimpleNamespace
    from unittest.mock import patch

    from veomni.checkpoint import dcp_checkpointer
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer, ModelState

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model = torch.nn.Linear(2, 2)
        failure = ValueError("injected row metadata error") if rank == 1 else None
        with (
            patch.object(dcp_checkpointer, "checkpoint_row_counts", side_effect=failure, return_value={}),
            patch.object(dcp, "save") as save,
            patch.object(dcp, "async_save") as async_save,
            patch.object(dcp, "load") as load,
        ):
            expected = ValueError if rank == 1 else RuntimeError
            with pytest.raises(expected, match="injected row metadata error|failed on another rank"):
                if operation == "load":
                    DistributedCheckpointer.load(directory, {"model": model}, process_group=dist.group.WORLD)
                else:
                    DistributedCheckpointer.execute_save(
                        {"model": ModelState(model, parallel_state=SimpleNamespace(dp_mode="ddp"))},
                        dcp.FileSystemWriter(directory),
                        save_async=operation == "async",
                    )
            save.assert_not_called()
            async_save.assert_not_called()
            load.assert_not_called()
        value = torch.ones(1)
        dist.all_reduce(value)
        assert value.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("operation", ["load", "save", "async"])
def test_row_metadata_failure_is_coordinated_before_dcp(tmp_path, operation):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_row_metadata_failure_worker, args=(str(tmp_path), operation), nprocs=2)


def test_logical_rows_roundtrip(tmp_path):
    """Store logical prefixes and zero the unsaved runtime tail."""
    weight = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    original = weight.clone()
    counts = {"model.weight": (11, 16)}
    planner = RowCropSavePlanner(counts)
    dcp.save({"model": {"weight": weight}}, checkpoint_id=tmp_path, planner=planner)
    metadata = dcp.FileSystemReader(tmp_path).read_metadata()
    stored = metadata.state_dict_metadata["model.weight"]
    assert stored.size == (11, 8)
    assert len(stored.chunks) == 1
    assert stored.chunks[0].offsets == (0, 0)
    assert stored.chunks[0].sizes == (11, 8)
    torch.testing.assert_close(weight, original)
    target = torch.full_like(weight, -1)
    dcp.load(
        {"model": {"weight": target}},
        checkpoint_id=tmp_path,
        planner=RowCropLoadPlanner(strict_model=True, rows_by_name=counts),
    )
    torch.testing.assert_close(target[:11], weight[:11])
    assert torch.count_nonzero(target[11:]) == 0


@pytest.mark.parametrize("saved_rows", [10, 17])
def test_invalid_saved_rows_are_rejected(tmp_path, saved_rows):
    """Reject checkpoint shapes outside the logical/runtime row contract."""
    dcp.save({"model": {"weight": torch.zeros(saved_rows, 8)}}, checkpoint_id=tmp_path)
    with pytest.raises(dcp.CheckpointException, match="incompatible"):
        dcp.load(
            {"model": {"weight": torch.zeros(16, 8)}},
            checkpoint_id=tmp_path,
            planner=RowCropLoadPlanner(strict_model=True, rows_by_name={"model.weight": (11, 16)}),
        )


def test_mixed_row_formats_are_rejected(tmp_path):
    """Require cropped or padded format consistently across selected weights."""
    dcp.save({"model": {"a": torch.zeros(11, 8), "b": torch.zeros(16, 8)}}, checkpoint_id=tmp_path)
    with pytest.raises(dcp.CheckpointException, match="mixes logical and padded"):
        dcp.load(
            {"model": {"a": torch.zeros(16, 8), "b": torch.zeros(16, 8)}},
            checkpoint_id=tmp_path,
            planner=RowCropLoadPlanner(strict_model=True, rows_by_name={"model.a": (11, 16), "model.b": (11, 16)}),
        )
