# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FSDP2 + expert-parallel optimizer/DCP cold-recovery regression on Ascend."""

import hashlib
import json
import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.ci._npu_moe_fp32_cold_worker import run
from veomni.checkpoint.layout import optimizer_dir, weights_dir
from veomni.utils.device import IS_NPU_AVAILABLE


def _assert_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    else:
        assert left == right


def _assert_ep2_checkpoint_layout(checkpoint_dir):
    weights = FileSystemReader(weights_dir(str(checkpoint_dir))).read_metadata().state_dict_metadata
    optimizer = FileSystemReader(optimizer_dir(str(checkpoint_dir))).read_metadata().state_dict_metadata
    marker = "__veomni_multi_optimizer_layout_v1__"
    rank_local = f"optimizer.{marker}" in optimizer
    if rank_local:
        keys = [marker, *(f"__veomni_rank_{rank}____inventory" for rank in range(2))]
        assert all(isinstance(optimizer[f"optimizer.{key}"], BytesStorageMetadata) for key in keys)
        scratch = {"optimizer": dict.fromkeys(keys, "")}
        dcp.load(scratch, checkpoint_id=optimizer_dir(str(checkpoint_dir)), no_dist=True)
        layout = json.loads(scratch["optimizer"][marker])
        assert set(layout) == {"generation", "layout", "inventory_digests"}
        assert isinstance(layout["generation"], str) and layout["generation"]
        geometry = json.loads(layout["layout"])
        assert set(geometry) == {"version", "ranks", "layouts"}
        assert geometry["version"] == 1 and geometry["ranks"] == [0, 1]
        assert len(geometry["layouts"]) == 2
        assert all(
            isinstance(item, str) and len(item) == 64 and all(char in "0123456789abcdef" for char in item)
            for item in geometry["layouts"]
        )
        generation_key = "__veomni_multi_generation_v1__"
        assert isinstance(weights[generation_key], BytesStorageMetadata)
        model_state = {generation_key: ""}
        dcp.load(model_state, checkpoint_id=weights_dir(str(checkpoint_dir)), no_dist=True)
        assert model_state[generation_key] == layout["generation"]
        assert len(layout["inventory_digests"]) == 2
        for rank, key in enumerate(keys[1:]):
            inventory = json.loads(scratch["optimizer"][key])
            prefix = f"optimizer.__veomni_rank_{rank}__"
            assert inventory == sorted(
                name for name in optimizer if name.startswith(prefix) and name != f"optimizer.{key}"
            )
            digest = hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()
            assert digest == layout["inventory_digests"][rank]
    for name, shape in (
        ("layer.experts.gate_up_proj", (4, 64, 256)),
        ("layer.experts.down_proj", (4, 256, 32)),
    ):
        weight = weights[f"model.{name}"]
        assert tuple(weight.size) == shape
        assert weight.properties.dtype == torch.float32
        local_shape = (2, *shape[1:])
        canonical_chunks = {((0, 0, 0), local_shape), ((2, 0, 0), local_shape)}
        assert len(weight.chunks) == len(canonical_chunks)
        assert {(tuple(chunk.offsets), tuple(chunk.sizes)) for chunk in weight.chunks} == canonical_chunks
        for field in ("exp_avg", "exp_avg_sq"):
            owners = range(2) if rank_local else (None,)
            for rank in owners:
                prefix = f"__veomni_rank_{rank}__" if rank is not None else ""
                entry = optimizer[f"optimizer.{prefix}state.{name}.{field}"]
                expected_shape = local_shape if rank_local else shape
                expected_chunks = {((0, 0, 0), local_shape)} if rank_local else canonical_chunks
                assert tuple(entry.size) == expected_shape
                assert entry.properties.dtype == torch.float32
                assert len(entry.chunks) == len(expected_chunks)
                assert {(tuple(chunk.offsets), tuple(chunk.sizes)) for chunk in entry.chunks} == expected_chunks


def _save_cpu_expert_metadata(rank, checkpoint_dir, checkpoint_format, endpoint):
    from datetime import timedelta

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        mesh = init_device_mesh("cpu", (2,))
        model, optimizer = {}, {}
        for name, shape in (
            ("layer.experts.gate_up_proj", (4, 64, 256)),
            ("layer.experts.down_proj", (4, 256, 32)),
        ):
            tensor = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
            tensor = (
                tensor[2 * rank : 2 * rank + 2]
                if checkpoint_format == "bare"
                else distribute_tensor(tensor, mesh, [Shard(0)])
            )
            model[name] = tensor
            rank_local = checkpoint_format in ("rank_local", "corrupt_inventory", "wrong_generation", "wrong_layout")
            prefix = f"__veomni_rank_{rank}__" if rank_local else ""
            value = tensor.to_local() if rank_local else tensor
            optimizer[f"{prefix}state.{name}.exp_avg"] = value
            optimizer[f"{prefix}state.{name}.exp_avg_sq"] = value
        if checkpoint_format in ("rank_local", "corrupt_inventory", "wrong_generation", "wrong_layout"):
            prefix = f"__veomni_rank_{rank}__"
            optimizer[f"{prefix}__scalars"] = b"cpu-test"
            inventory = sorted(f"optimizer.{key}" for key in optimizer)
            local_digest = hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()
            digests = [None, None]
            dist.all_gather_object(digests, local_digest)
            if checkpoint_format == "corrupt_inventory":
                digests[1] = "0" * 64
            optimizer[f"{prefix}__inventory"] = json.dumps(inventory)
            geometry = {
                "version": 1,
                "ranks": [1, 0] if checkpoint_format == "wrong_layout" else [0, 1],
                "layouts": ["a" * 64, "b" * 64],
            }
            optimizer["__veomni_multi_optimizer_layout_v1__"] = json.dumps(
                {"generation": "cpu-test", "layout": json.dumps(geometry), "inventory_digests": digests}
            )
        model_state = {"model": model}
        if checkpoint_format in ("rank_local", "corrupt_inventory", "wrong_generation", "wrong_layout"):
            model_state["__veomni_multi_generation_v1__"] = (
                "other-generation" if checkpoint_format == "wrong_generation" else "cpu-test"
            )
        dcp.save(model_state, checkpoint_id=weights_dir(checkpoint_dir))
        dcp.save({"optimizer": optimizer}, checkpoint_id=optimizer_dir(checkpoint_dir))
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize(
    "checkpoint_format", ["bare", "canonical", "rank_local", "corrupt_inventory", "wrong_generation", "wrong_layout"]
)
def test_ep2_metadata_requires_proven_expert_shards(tmp_path, checkpoint_format):
    from tests.tools.launch_utils import spawn_with_rendezvous

    checkpoint_dir = str(tmp_path / "ep2-dcp")
    spawn_with_rendezvous(_save_cpu_expert_metadata, args=(checkpoint_dir, checkpoint_format), nprocs=2)
    if checkpoint_format in ("canonical", "rank_local"):
        _assert_ep2_checkpoint_layout(checkpoint_dir)
    else:
        with pytest.raises(AssertionError):
            _assert_ep2_checkpoint_layout(checkpoint_dir)


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2))
@pytest.mark.parametrize("ep", [1, 2])
def test_native_fsdp_expert_fp32_dcp_cold_resume(tmp_path, ep):
    if not IS_NPU_AVAILABLE or torch.npu.device_count() < 2:
        if os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1":
            pytest.fail("VEOMNI_REQUIRE_NPU_TESTS=1 requires two native NPU devices")
        pytest.skip("requires two native Ascend devices")
    from tests.tools.launch_utils import spawn_with_rendezvous

    for phase in ("fresh", "resume"):
        spawn_with_rendezvous(run, args=(phase, str(tmp_path), ep), nprocs=2)
    fresh_by_rank = []
    for rank in range(2):
        fresh = torch.load(tmp_path / f"ep{ep}-fresh-rank{rank}.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(tmp_path / f"ep{ep}-resume-rank{rank}.pt", map_location="cpu", weights_only=True)
        _assert_equal(fresh, resumed)
        fresh_by_rank.append(fresh)
    if ep == 2:
        for name in fresh_by_rank[0]["before"]:
            for row in (0, 1):
                left = fresh_by_rank[0]["before"][name]["param"][row]
                right = fresh_by_rank[1]["before"][name]["param"][row]
                assert not torch.equal(left, right), f"symmetric EP fixture for {name}: param row {row}"
            for field in ("m", "v"):
                left = fresh_by_rank[0]["before"][name][field][1]
                right = fresh_by_rank[1]["before"][name][field][1]
                assert not torch.equal(left, right), f"symmetric EP fixture for {name}: {field}"
        _assert_ep2_checkpoint_layout(tmp_path / "ep2-dcp")
