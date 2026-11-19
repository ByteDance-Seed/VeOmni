# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Exercise v2 EP source provenance together with persistent, row-cropped PLE."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.test_qwen4_exp_ple_dcp_crop import _Model, _reference
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import (
    _EXPERT_LAYOUTS_KEY,
    ModelState,
    _CanonicalRowCropLoadPlanner,
    _CanonicalRowCropSavePlanner,
    _checkpoint_rows_by_name,
)
from veomni.distributed.parallel_plan import SpecInfo


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _canonical_ep_ple_worker(rank, path, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        ep = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        ple = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ple_fsdp", "ple"))
        rows, other = _reference()
        ple_name = "block.ngram_embedding.shard_0.weight"
        expert_name = "experts.weight"
        expert = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        local_expert = expert.chunk(2, dim=0)[ep.get_coordinate()[1]].contiguous()
        ps = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep", "ple"), extra_parallel_fsdp_device_mesh={"ep": ep, "ple": ple}
        )

        def model(fill):
            result = _Model()
            result.experts = torch.nn.Module()
            result.block.ngram_embedding["shard_0"].weight = torch.nn.Parameter(
                distribute_tensor(torch.zeros_like(rows) if fill else rows, ple, [Shard(1), Shard(0)])
            )
            result.experts.weight = torch.nn.Parameter(
                distribute_tensor(torch.zeros_like(local_expert) if fill else local_expert, ep["ep_fsdp"], [Shard(0)])
            )
            with torch.no_grad():
                result.other.weight.copy_(torch.zeros_like(other) if fill else other)
            result._fqn2spec_info = {
                ple_name: SpecInfo("ple", Shard(0), ple_name, ple, 1, 1, tuple(rows.shape), False),
                expert_name: SpecInfo("ep", Shard(0), expert_name, ep, 0, None, tuple(expert.shape), True),
                "other.weight": SpecInfo("ep", Replicate(), "other.weight", ep),
            }
            return result

        source = model(False)
        from veomni.utils.save_safetensor_utils import _save_hf_safetensor_distributed

        hf_dir = Path(path) / "hf"
        with patch("veomni.utils.save_safetensor_utils.synchronize"):
            _save_hf_safetensor_distributed(source, str(hf_dir), None, None, parallel_state=ps)
        if rank == 0:
            from safetensors.torch import load_file

            exported = {}
            for file in hf_dir.glob("*.safetensors"):
                exported.update(load_file(str(file)))
            torch.testing.assert_close(exported[ple_name], rows[:11].to(torch.bfloat16), rtol=0, atol=0)
            torch.testing.assert_close(exported[expert_name], expert.to(torch.bfloat16), rtol=0, atol=0)
        dist.barrier()
        dcp.save(
            {"model": ModelState(source, parallel_state=ps)},
            checkpoint_id=path,
            planner=_CanonicalRowCropSavePlanner(
                _checkpoint_rows_by_name(source), parallel_state=ps, source_model=source
            ),
        )
        metadata = dcp.FileSystemReader(path).read_metadata()
        ple_keys = [key for key in metadata.state_dict_metadata if key.endswith(ple_name)]
        expert_keys = [key for key in metadata.state_dict_metadata if key.endswith(expert_name)]
        assert len(ple_keys) == 1 and metadata.state_dict_metadata[ple_keys[0]].size == (11, 8)
        assert len(expert_keys) == 1
        assert metadata.planner_data[_EXPERT_LAYOUTS_KEY]["tensors"][expert_keys[0]]["expert_order"] == "canonical"
        target = model(True)
        dcp.load(
            {"model": ModelState(target, parallel_state=ps)},
            checkpoint_id=path,
            planner=_CanonicalRowCropLoadPlanner(True, _checkpoint_rows_by_name(target)),
        )
        torch.testing.assert_close(
            target.block.ngram_embedding["shard_0"].weight.to_local(),
            source.block.ngram_embedding["shard_0"].weight.to_local(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(target.experts.weight.to_local(), source.experts.weight.to_local(), rtol=0, atol=0)
        torch.testing.assert_close(target.other.weight, other, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_canonical_ep_and_persistent_ple_logical_rows_roundtrip(tmp_path):
    spawn_with_rendezvous(_canonical_ep_ple_worker, args=(str(tmp_path),), nprocs=4)


def _canonical_ep_ple_lora_worker(rank, path, endpoint):
    from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        ep = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        ple = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ple_fsdp", "ple"))
        rows, _ = _reference()
        ple_name = "base_model.model.block.ngram_embedding.shard_0.weight"
        expert_name = "base_model.model.experts.weight"
        expert = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        local_expert = expert.chunk(2, dim=0)[ep.get_coordinate()[1]].contiguous()
        ps = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep", "ple"), extra_parallel_fsdp_device_mesh={"ep": ep, "ple": ple}
        )

        def model(target, train_ple):
            base = _Model()
            base.experts = torch.nn.Module()
            base.experts.weight = torch.nn.Parameter(torch.ones(4, 4))
            result = VeOmniLoraModel(base, VeOmniLoraConfig(r=2, lora_alpha=4, target_modules=["other"]))
            ple_values = torch.full_like(rows, 0.5) if target else rows
            result.base_model.model.block.ngram_embedding["shard_0"].weight = torch.nn.Parameter(
                distribute_tensor(ple_values, ple, [Shard(1), Shard(0)]), requires_grad=train_ple
            )
            expert_values = torch.full_like(local_expert, 0.5) if target else local_expert
            result.base_model.model.experts.weight = torch.nn.Parameter(
                distribute_tensor(expert_values, ep["ep_fsdp"], [Shard(0)]), requires_grad=False
            )
            with torch.no_grad():
                for name, value in result.named_parameters():
                    if ".lora_" in name:
                        value.fill_(0.0 if target else 0.25)
            result._fqn2spec_info = {
                name: SpecInfo("ep", Replicate(), name, ep) for name, _ in result.named_parameters()
            }
            result._fqn2spec_info[ple_name] = SpecInfo("ple", Shard(0), ple_name, ple, 1, 1, tuple(rows.shape), False)
            result._fqn2spec_info[expert_name] = SpecInfo(
                "ep", Shard(0), expert_name, ep, 0, None, tuple(expert.shape), True
            )
            return result

        for train_ple in (False, True):
            source, target = model(False, train_ple), model(True, train_ple)
            checkpoint = str(Path(path) / f"train-ple-{train_ple}")
            dcp.save(
                {"model": ModelState(source, trainable_only=True, parallel_state=ps)},
                checkpoint_id=checkpoint,
                planner=_CanonicalRowCropSavePlanner(
                    _checkpoint_rows_by_name(source), parallel_state=ps, source_model=source
                ),
            )
            metadata = dcp.FileSystemReader(checkpoint).read_metadata()
            keys = [key for key in metadata.state_dict_metadata if key.endswith(ple_name)]
            assert len(keys) == int(train_ple)
            if keys:
                assert metadata.state_dict_metadata[keys[0]].size == (11, 8)
            old_ple = target.base_model.model.block.ngram_embedding["shard_0"].weight.to_local().clone()
            old_expert = target.base_model.model.experts.weight.to_local().clone()
            dcp.load(
                {"model": ModelState(target, trainable_only=True, parallel_state=ps)},
                checkpoint_id=checkpoint,
                planner=_CanonicalRowCropLoadPlanner(False, _checkpoint_rows_by_name(target)),
            )
            source_params, target_params = dict(source.named_parameters()), dict(target.named_parameters())
            for name in source_params:
                if ".lora_" in name:
                    torch.testing.assert_close(target_params[name], source_params[name], rtol=0, atol=0)
            restored_ple = target.base_model.model.block.ngram_embedding["shard_0"].weight.to_local()
            expected_ple = (
                source.base_model.model.block.ngram_embedding["shard_0"].weight.to_local() if train_ple else old_ple
            )
            torch.testing.assert_close(restored_ple, expected_ple, rtol=0, atol=0)
            torch.testing.assert_close(target.base_model.model.experts.weight.to_local(), old_expert, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_canonical_ep_and_ple_trainable_only_lora_roundtrip(tmp_path):
    spawn_with_rendezvous(_canonical_ep_ple_lora_worker, args=(str(tmp_path),), nprocs=4)


def test_proven_multi_optimizer_restores_cropped_ple_and_next_update(tmp_path):
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.checkpoint.layout import weights_dir
    from veomni.optim.optimizer import MultiOptimizer

    def build():
        value = _Model()
        rows, other = _reference()
        with torch.no_grad():
            value.block.ngram_embedding["shard_0"].weight.copy_(rows)
            value.other.weight.copy_(other)
        ple = torch.optim.AdamW([value.block.ngram_embedding["shard_0"].weight], lr=0.001)
        dense = torch.optim.AdamW([value.other.weight], lr=0.002)
        return value, MultiOptimizer(value, {"ple": ple, "dense": dense}, ["ple", "dense"])

    def step(model, optimizer):
        for param in model.parameters():
            param.grad = torch.full_like(param, 0.25)
        model.block.ngram_embedding["shard_0"].weight.grad[11:] = 0
        optimizer.step()
        optimizer.zero_grad()

    source, optimizer = build()
    step(source, optimizer)
    parallel = SimpleNamespace(dp_mode="ddp")
    state = {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None}
    DistributedCheckpointer.save(str(tmp_path), state, parallel_state=parallel)
    metadata = dcp.FileSystemReader(weights_dir(str(tmp_path))).read_metadata()
    name = "model.block.ngram_embedding.shard_0.weight"
    assert metadata.state_dict_metadata[name].size == (11, 8)
    target, restored = build()
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(restored.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
    step(source, optimizer)
    step(target, restored)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(restored.state_dict(), optimizer.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["rows", "columns", "dtype"])
def test_proven_multi_optimizer_rejects_incompatible_ple_before_model_load(tmp_path, fault):
    import pickle
    from dataclasses import replace

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.checkpoint.layout import weights_dir
    from veomni.optim.optimizer import MultiOptimizer

    def build():
        model = _Model()
        leaf = torch.optim.AdamW(model.parameters(), lr=0.001)
        return model, MultiOptimizer(model, {"all": leaf}, ["all"])

    source, optimizer = build()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    metadata_path = Path(weights_dir(str(tmp_path))) / ".metadata"
    with metadata_path.open("rb") as file:
        metadata = pickle.load(file)
    name = "model.block.ngram_embedding.shard_0.weight"
    entry = metadata.state_dict_metadata[name]
    if fault == "rows":
        metadata.state_dict_metadata[name] = replace(entry, size=torch.Size((10, 8)))
    elif fault == "columns":
        metadata.state_dict_metadata[name] = replace(entry, size=torch.Size((11, 7)))
    else:
        metadata.state_dict_metadata[name] = replace(entry, properties=replace(entry.properties, dtype=torch.bfloat16))
    with metadata_path.open("wb") as file:
        pickle.dump(metadata, file)

    target, restored = build()
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(RuntimeError, match="dtype or shape differs"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    for name, value in before.items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    assert not restored.optimizers_dict["all"].state


def _canonical_ep_ple_multi_cold_worker(rank, root, phase, endpoint):
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.checkpoint.layout import weights_dir
    from veomni.optim.optimizer import MultiOptimizer

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        ep = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        ple = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ple_fsdp", "ple"))
        rows, other = _reference()
        ple_name, expert_name = "block.ngram_embedding.shard_0.weight", "experts.weight"
        expert = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        local_expert = expert.chunk(2, dim=0)[ep.get_coordinate()[1]].contiguous()
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep", "ple"), extra_parallel_fsdp_device_mesh={"ep": ep, "ple": ple}
        )
        model = _Model()
        model.experts = torch.nn.Module()
        model.block.ngram_embedding["shard_0"].weight = torch.nn.Parameter(
            distribute_tensor(rows, ple, [Shard(1), Shard(0)])
        )
        model.experts.weight = torch.nn.Parameter(distribute_tensor(local_expert, ep["ep_fsdp"], [Shard(0)]))
        with torch.no_grad():
            model.other.weight.copy_(other)
        model._fqn2spec_info = {
            ple_name: SpecInfo("ple", Shard(0), ple_name, ple, 1, 1, tuple(rows.shape), False),
            expert_name: SpecInfo("ep", Shard(0), expert_name, ep, 0, None, tuple(expert.shape), True),
            "other.weight": SpecInfo("ep", Replicate(), "other.weight", ep),
        }
        parameters = {
            "ple": model.block.ngram_embedding["shard_0"].weight,
            "expert": model.experts.weight,
            "dense": model.other.weight,
        }
        leaves = {name: torch.optim.AdamW([value], lr=0.0025, foreach=False) for name, value in parameters.items()}
        optimizer = MultiOptimizer(model, leaves, list(leaves))

        def snapshot():
            result = {}
            for name, param in parameters.items():
                local = param.to_local() if hasattr(param, "to_local") else param
                result[name] = {"param": local.detach().clone(), "state": {}}
                for key, value in leaves[name].state.get(param, {}).items():
                    local_value = value.to_local() if hasattr(value, "to_local") else value
                    result[name]["state"][key] = (
                        local_value.detach().clone() if torch.is_tensor(local_value) else local_value
                    )
            return result

        def step(value, sparse):
            ple_grad = torch.full_like(rows, value)
            ple_grad[11:] = 0
            distributed_ple_grad = distribute_tensor(ple_grad, ple, [Shard(1), Shard(0)])
            parameters["ple"].grad = distributed_ple_grad if not sparse or ple.get_local_rank("ple") == 0 else None
            parameters["expert"].grad = (
                torch.full_like(parameters["expert"], value) if not sparse or ep.get_local_rank("ep") == 0 else None
            )
            parameters["dense"].grad = torch.full_like(parameters["dense"], value)
            optimizer.step()
            optimizer.zero_grad()

        checkpoint = str(Path(root) / "checkpoint")
        state = {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None}
        reference = Path(root) / f"reference-{rank}.pt"
        if phase == "save":
            step(0.1, True)
            before = snapshot()
            DistributedCheckpointer.save(checkpoint, state, parallel_state=parallel)
            metadata = dcp.FileSystemReader(weights_dir(checkpoint)).read_metadata()
            keys = [key for key in metadata.state_dict_metadata if key.endswith(ple_name)]
            assert len(keys) == 1 and metadata.state_dict_metadata[keys[0]].size == (11, 8)
            step(0.2, False)
            torch.save({"before": before, "after": snapshot()}, reference)
        else:
            DistributedCheckpointer.load(checkpoint, state, parallel_state=parallel)
            expected = torch.load(reference, weights_only=True)
            torch.testing.assert_close(snapshot(), expected["before"], rtol=0, atol=0)
            step(0.2, False)
            torch.testing.assert_close(snapshot(), expected["after"], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_proven_multi_optimizer_ep_ple_cold_resume_and_update(tmp_path):
    for phase in ("save", "resume"):
        spawn_with_rendezvous(_canonical_ep_ple_multi_cold_worker, args=(str(tmp_path), phase), nprocs=4)
