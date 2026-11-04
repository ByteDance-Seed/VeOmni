# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unsupported TP expert saves must fail before checkpoint writes."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.distributed.parallel_plan import SpecInfo
from veomni.distributed.parallel_state import ParallelState
from veomni.utils.dist_utils import raise_if_any_rank_failed


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
        patch("veomni.checkpoint.dcp_checkpointer.raise_if_any_rank_failed", wraps=raise_if_any_rank_failed) as agree,
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
    assert isinstance(agree.call_args.args[0], RuntimeError)
    assert agree.call_args.args[1] == "checking canonical ExtraParallel optimizer state"
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
    with patch("veomni.checkpoint.dcp_checkpointer.raise_if_any_rank_failed", wraps=raise_if_any_rank_failed) as agree:
        DistributedCheckpointer.save(path, {"model": model}, parallel_state=parallel)
    assert any(
        call.args[0] is None and call.args[1] == "checking canonical ExtraParallel optimizer state"
        for call in agree.call_args_list
    )
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
