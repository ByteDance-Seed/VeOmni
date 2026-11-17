# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Logical-row checkpoint plans preserve padded runtime tensors."""

import pytest
import torch
import torch.distributed.checkpoint as dcp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.row_crop import RowCropLoadPlanner, RowCropSavePlanner


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


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
