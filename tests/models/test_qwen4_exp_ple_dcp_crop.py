# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DCP save stores logical PLE rows and reloads them without changing a value."""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.checkpoint.layout import optimizer_dir, step_dir, weights_dir
from veomni.checkpoint.ple_rows import _resize_dtensor, pad_ple_state_dict_from_dcp, ple_checkpoint_is_cropped
from veomni.models.transformers.qwen4_exp.checkpoint_tensor_converter import (
    create_qwen4_exp_checkpoint_tensor_converter,
)
from veomni.utils.save_safetensor_utils import crop_state_dict_for_hf_export, get_model_save_state


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))


PADDED = 16
LOGICAL = 11
COLS = 8
SENTINEL = -123.0


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.ngram_embedding = nn.ModuleDict({"shard_0": nn.Embedding(PADDED, COLS)})
        self.other = nn.Linear(4, 4, bias=False)
        self.block.rows_per_checkpoint_shard = LOGICAL
        self.block.padded_rows_per_shard = PADDED
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(split_ngram_parts=4, make_ngram_vocab_size_divisible_by=8)
        )
        self._create_checkpoint_tensor_converter = create_qwen4_exp_checkpoint_tensor_converter


def _parallel_state():
    return SimpleNamespace(dp_mode="fsdp2")


def _reference():
    rows = torch.arange(PADDED * COLS, dtype=torch.float32).reshape(PADDED, COLS)
    rows[LOGICAL:] = SENTINEL
    other = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 0.5
    return rows, other


def _install(model, mesh, rows, other):
    weight = distribute_tensor(rows, mesh, [Shard(1), Shard(0)])
    model.block.ngram_embedding["shard_0"].weight = nn.Parameter(weight)
    with torch.no_grad():
        model.other.weight.copy_(other)


def _full(weight):
    tensor = weight.full_tensor() if hasattr(weight, "full_tensor") else weight
    return tensor.detach().cpu()


def _optimizer(model):
    # Disable weight decay on PLE so the sentinel padding cannot affect the
    # uninterrupted reference. Real gradients touch logical vocabulary only.
    return torch.optim.AdamW(
        [
            {"params": list(model.block.parameters()), "weight_decay": 0.0},
            {"params": list(model.other.parameters())},
        ],
        lr=0.01,
        foreach=False,
    )


def _train_step(model, optimizer, step):
    optimizer.zero_grad(set_to_none=True)
    weight = model.block.ngram_embedding["shard_0"].weight
    local = weight.to_local()
    mesh = weight.device_mesh
    row_axis = mesh.mesh_dim_names.index("ple")
    row_start = mesh.get_local_rank(row_axis) * (PADDED // mesh.size(row_axis))
    valid_rows = max(0, min(local.shape[0], LOGICAL - row_start))
    # to_local() remains differentiable: backward populates the real DTensor
    # parameter gradient, and AdamW creates padded DTensor moments naturally.
    loss = local[:valid_rows].square().sum() * (0.001 * (step + 1))
    loss = loss + model.other.weight.square().sum() * (0.01 * (step + 1))
    loss.backward()
    optimizer.step()


def _check_optimizer(source, restored, source_opt, restored_opt):
    """Compare optimizer tensor state for the original and restored parameters."""
    restored_parameters = dict(restored.named_parameters())
    for name, parameter in source.named_parameters():
        loaded_parameter = restored_parameters[name]
        expected_state = source_opt.state[parameter]
        loaded_state = restored_opt.state[loaded_parameter]
        for key in ("exp_avg", "exp_avg_sq"):
            expected, actual = expected_state[key], loaded_state[key]
            assert actual.shape == expected.shape == parameter.shape, (name, key)
            if "ngram_embedding" in name:
                assert isinstance(actual, DTensor)
                assert actual.placements == expected.placements == parameter.placements
                assert actual.device_mesh == parameter.device_mesh
                assert actual.to_local().shape == parameter.to_local().shape
                assert torch.count_nonzero(_full(expected)[:LOGICAL]) > 0, (name, key)
            torch.testing.assert_close(_full(actual), _full(expected), rtol=0, atol=0, msg=f"{name}: {key}")
        torch.testing.assert_close(loaded_state["step"], expected_state["step"], rtol=0, atol=0)


def _max_abs(left, right):
    return (left.detach().cpu() - right.detach().cpu()).abs().max().item()


def _check_resume(model, trained_rows, trained_other, source, source_opt, opt):
    """Check restored weights and optimizer state before the next update."""
    loaded = _full(model.block.ngram_embedding["shard_0"].weight)
    logical_diff = _max_abs(loaded[:LOGICAL], trained_rows[:LOGICAL])
    tail_diff = _max_abs(loaded[LOGICAL:], torch.zeros_like(loaded[LOGICAL:]))
    other_diff = _max_abs(model.other.weight, trained_other)
    assert tuple(loaded.shape) == (PADDED, COLS)
    assert logical_diff == 0.0, logical_diff
    assert tail_diff == 0.0, tail_diff
    assert SENTINEL not in set(loaded.reshape(-1).tolist())
    assert other_diff == 0.0, other_diff
    _check_optimizer(source, model, source_opt, opt)
    return logical_diff, tail_diff, other_diff


def _dcp_save(*args, **kwargs):
    """Require DCP to save without constructing resized PLE DTensors."""
    with patch("veomni.checkpoint.ple_rows._resize_dtensor", side_effect=AssertionError("DCP resharded PLE")):
        return DistributedCheckpointer.save(*args, **kwargs)


def _dcp_load(*args, **kwargs):
    """Require DCP to read directly into the runtime PLE layout."""
    with patch("veomni.checkpoint.ple_rows._resize_dtensor", side_effect=AssertionError("DCP resharded PLE")):
        return DistributedCheckpointer.load(*args, **kwargs)


def main():
    # The container's training backend is NPU, but these ranks use gloo so the
    # checkpointer's agreement collectives must stay on CPU.
    """Exercise distributed PLE checkpoint save and resume."""
    import veomni.utils.dist_utils as dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ple_fsdp", "ple"))
    rows, other = _reference()
    # Changed chunk boundaries, empty peers, and expansion must never gather.
    for placements in ([Shard(1), Shard(0)], [Replicate(), Shard(0)], [Shard(0), Replicate()]):
        source = distribute_tensor(rows.clone(), mesh, placements)
        for logical in (15, 11, 1):
            with patch.object(DTensor, "full_tensor", side_effect=AssertionError("unexpected full-table gather")):
                cropped = _resize_dtensor(source, logical)
                restored = _resize_dtensor(cropped, PADDED)
                with patch.object(dist, "all_to_all_single", side_effect=AssertionError("load target communicated")):
                    target = _resize_dtensor(source, logical, empty=True)
            torch.testing.assert_close(_full(cropped), rows[:logical], rtol=0, atol=0)
            expected = rows.clone()
            expected[logical:] = 0
            torch.testing.assert_close(_full(restored), expected, rtol=0, atol=0)
            assert target.shape == cropped.shape
            assert target.to_local().shape == cropped.to_local().shape
    for placements in ([Shard(1), Replicate()], [Replicate(), Replicate()]):
        name = "block.ngram_embedding.shard_0.weight"
        column_state = {name: distribute_tensor(rows.clone(), mesh, placements)}
        cropped = crop_state_dict_for_hf_export(_Model(), column_state)
        restored = pad_ple_state_dict_from_dcp(_Model(), cropped)
        torch.testing.assert_close(_full(cropped[name]), rows[:LOGICAL], rtol=0, atol=0)
        expected = rows.clone()
        expected[LOGICAL:] = 0
        torch.testing.assert_close(_full(restored[name]), expected, rtol=0, atol=0)

    source = _Model()
    _install(source, mesh, rows.clone(), other.clone())
    source_opt = _optimizer(source)
    for step in range(2):
        _train_step(source, source_opt, step)
    trained_rows = _full(source.block.ngram_embedding["shard_0"].weight).clone()
    trained_other = source.other.weight.detach().cpu().clone()

    directory = None
    if rank == 0:
        directory = tempfile.mkdtemp(prefix="ple-dcp-")
    holder = [directory]
    dist.broadcast_object_list(holder, src=0)
    directory = holder[0]
    parallel_state = _parallel_state()

    try:
        _dcp_save(
            directory,
            {"model": source, "optimizer": source_opt, "extra_state": {"lr_scheduler": None}},
            global_steps=1,
            parallel_state=parallel_state,
        )
        checkpoint = step_dir(directory, 1)
        weight_dir = weights_dir(checkpoint)
        assert ple_checkpoint_is_cropped(weight_dir, source)
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(weight_dir).read_metadata()
        ple_key = next(key for key in metadata.state_dict_metadata if "ngram_embedding.shard_0.weight" in key)
        saved_rows = int(metadata.state_dict_metadata[ple_key].size[0])
        assert saved_rows == LOGICAL, saved_rows
        ple_metadata = metadata.state_dict_metadata[ple_key]
        row_intervals = {(int(chunk.offsets[0]), int(chunk.sizes[0])) for chunk in ple_metadata.chunks}
        assert row_intervals == {(0, 8), (8, 3)}, row_intervals
        # The ordinary weight must keep its original complete shape.
        assert tuple(metadata.state_dict_metadata["model.other.weight"].size) == (4, 4)

        optimizer_metadata = FileSystemReader(optimizer_dir(checkpoint)).read_metadata()
        for moment in ("exp_avg", "exp_avg_sq"):
            entries = [
                value
                for key, value in optimizer_metadata.state_dict_metadata.items()
                if "ngram_embedding.shard_0.weight" in key and key.endswith(moment)
            ]
            assert len(entries) == 1, (moment, optimizer_metadata.state_dict_metadata.keys())
            assert tuple(entries[0].size) == (PADDED, COLS), moment

        dense = {
            "block.ngram_embedding.shard_0.weight": trained_rows,
            "other.weight": trained_other,
        }
        exported = crop_state_dict_for_hf_export(source, dense)
        hf_ple = exported["block.ngram_embedding.shard_0.weight"]
        assert tuple(hf_ple.shape) == (LOGICAL, COLS)
        assert _max_abs(hf_ple, trained_rows[:LOGICAL]) == 0.0
        assert torch.equal(exported["other.weight"], trained_other)

        hf_state = get_model_save_state(source, None, parallel_state=parallel_state)
        hf_saved = _full(hf_state["block.ngram_embedding.shard_0.weight"])
        assert tuple(hf_saved.shape) == (LOGICAL, COLS)
        assert hf_saved.dtype == torch.bfloat16
        assert _max_abs(hf_saved, trained_rows[:LOGICAL].to(torch.bfloat16)) == 0.0
        assert _max_abs(hf_state["other.weight"], trained_other.to(torch.bfloat16)) == 0.0

        restored = _Model()
        _install(restored, mesh, torch.zeros(PADDED, COLS), torch.zeros(4, 4))
        restored_opt = _optimizer(restored)
        _dcp_load(
            checkpoint,
            {"model": restored, "optimizer": restored_opt, "extra_state": {}},
            parallel_state=parallel_state,
        )
        logical_diff, tail_diff, other_diff = _check_resume(
            restored, trained_rows, trained_other, source, source_opt, restored_opt
        )

        _dcp_save(
            directory,
            {"model": restored, "optimizer": restored_opt, "extra_state": {"lr_scheduler": None}},
            global_steps=2,
            parallel_state=parallel_state,
        )
        second = _Model()
        _install(second, mesh, torch.full((PADDED, COLS), 9.0), torch.full((4, 4), 9.0))
        second_opt = _optimizer(second)
        _dcp_load(
            step_dir(directory, 2),
            {"model": second, "optimizer": second_opt, "extra_state": {}},
            parallel_state=parallel_state,
        )
        _check_resume(second, trained_rows, trained_other, source, source_opt, second_opt)
        assert ple_checkpoint_is_cropped(weights_dir(step_dir(directory, 2)), second)

        # Older padded checkpoints must retain their stored tail, not zero it.
        from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

        with patch(
            "veomni.checkpoint.dcp_checkpointer.PleSavePlanner",
            side_effect=lambda rows, **kwargs: DefaultSavePlanner(**kwargs),
        ):
            _dcp_save(
                directory,
                {"model": source, "optimizer": source_opt, "extra_state": {"lr_scheduler": None}},
                global_steps=3,
                parallel_state=parallel_state,
            )
        legacy = _Model()
        _install(legacy, mesh, torch.zeros(PADDED, COLS), torch.zeros(4, 4))
        legacy_opt = _optimizer(legacy)
        _dcp_load(
            step_dir(directory, 3),
            {"model": legacy, "optimizer": legacy_opt, "extra_state": {}},
            parallel_state=parallel_state,
        )
        torch.testing.assert_close(_full(legacy.block.ngram_embedding["shard_0"].weight), trained_rows, rtol=0, atol=0)
        torch.testing.assert_close(legacy.other.weight, trained_other, rtol=0, atol=0)
        _check_optimizer(source, legacy, source_opt, legacy_opt)

        # Compare the next real optimizer update with uninterrupted training.
        _train_step(source, source_opt, 2)
        _train_step(second, second_opt, 2)
        next_reference = _full(source.block.ngram_embedding["shard_0"].weight)
        next_restored = _full(second.block.ngram_embedding["shard_0"].weight)
        torch.testing.assert_close(next_restored[:LOGICAL], next_reference[:LOGICAL], rtol=0, atol=0)
        torch.testing.assert_close(next_restored[LOGICAL:], torch.zeros_like(next_restored[LOGICAL:]), rtol=0, atol=0)
        torch.testing.assert_close(second.other.weight, source.other.weight, rtol=0, atol=0)
        _check_optimizer(source, second, source_opt, second_opt)

        if rank == 0:
            print(
                f"PASS saved_rows={saved_rows} logical_max_abs={logical_diff} "
                f"tail_max_abs={tail_diff} other_max_abs={other_diff} "
                f"hf_bf16_max_abs=0.0 resume_cycles=2 ple_optimizer_padded=True next_update_match=True",
                flush=True,
            )
    finally:
        if rank == 0 and directory:
            shutil.rmtree(directory, ignore_errors=True)


def test_distributed_ple_checkpoint_roundtrip():
    """Launch the PLE checkpoint round trip with the configured accelerator mesh."""
    from tests.tools.launch_utils import run_bounded_process_group

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", ASCEND_RT_VISIBLE_DEVICES="", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    run_bounded_process_group(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            str(Path(__file__).resolve()),
        ],
        env=env,
        timeout=300,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
