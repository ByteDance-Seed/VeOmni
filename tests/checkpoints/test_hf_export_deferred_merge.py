# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""HF export of an oversized row-sharded ``emb`` table: kept split by rank, merged offline.

Consolidating such a table during training is one rank streaming the whole
tensor into one file -- about an hour for DeepSeek-V4.1's 197 GB n-gram tables
on an HDFS mount -- while the job waits. The export leaves its per-rank shards
and a marker instead, loaders refuse the directory, and
``scripts/merge_hf_export.py`` writes the table out afterwards.
"""

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from safetensors.torch import load_file
from torch.distributed.tensor import Replicate, Shard

import veomni.utils.save_safetensor_utils as save_utils
from veomni.checkpoint.layout import HF_PENDING_MERGE_FILENAME, HF_SHARDED_DIRNAME
from veomni.models.module_utils import _resolve_safetensors_shards, _resolve_state_dict_iterators


WORLD_SIZE = 4
ROWS, COLS = 3, 8
# Below the table (WORLD_SIZE * ROWS * COLS bf16 = 192 B) and the dense weight
# (128 B), above the bias (16 B).
MAX_SHARD_BYTES = 100


def _model(table_spec, dense_spec=None) -> nn.Module:
    model = nn.Module()
    model._fqn2spec_info = {"table": table_spec}
    if dense_spec is not None:
        model._fqn2spec_info["dense"] = dense_spec
    return model


def _local_rows(rank: int) -> torch.Tensor:
    return (torch.arange(ROWS * COLS, dtype=torch.float32).view(ROWS, COLS) + 100 * rank).to(torch.bfloat16)


def _replicated() -> dict:
    generator = torch.Generator().manual_seed(0)
    return {
        "dense": torch.randn(8, 8, generator=generator).to(torch.bfloat16),
        "bias": torch.randn(8, generator=generator).to(torch.bfloat16),
    }


def _export(model, state, save_path) -> None:
    with (
        patch.object(save_utils, "_MAX_SHARD_BYTES", MAX_SHARD_BYTES),
        patch.object(save_utils, "get_model_save_state", lambda *args, **kwargs: dict(state)),
    ):
        save_utils._save_hf_safetensor_distributed(model, save_path, None, None)


def _worker(rank: int, init_file: str, save_path: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD_SIZE)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from veomni.checkpoint.dcp_checkpointer import restore_extra_parallel_dim

        mesh = init_device_mesh("cpu", (1, WORLD_SIZE), mesh_dim_names=("emb_fsdp", "emb"))
        table = restore_extra_parallel_dim(_local_rows(rank), mesh, mesh["emb_fsdp"], ep_shard_dim=0)
        model = _model(
            SimpleNamespace(para_name="emb", placement=Shard(0)),
            SimpleNamespace(para_name="emb", placement=Replicate()),
        )
        _export(model, {"table": table, **_replicated()}, save_path)
    finally:
        dist.destroy_process_group()


def _load_export(save_path: str) -> dict:
    weight_map = json.loads((Path(save_path) / "model.safetensors.index.json").read_text())["weight_map"]
    loaded = {}
    for name in sorted(set(weight_map.values())):
        loaded.update(load_file(os.path.join(save_path, name)))
    assert set(loaded) == set(weight_map)
    return loaded


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo required")
def test_an_emb_sharded_table_is_kept_split_and_merged_offline():
    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "hf_ckpt", "ngram")
        mp.spawn(_worker, args=(os.path.join(tmp, "init"), save_path), nprocs=WORLD_SIZE, join=True)

        weight_map = json.loads((Path(save_path) / "model.safetensors.index.json").read_text())["weight_map"]
        pending = json.loads((Path(save_path) / HF_PENDING_MERGE_FILENAME).read_text())["weight_map"]
        assert set(weight_map) == {"table", "dense", "bias"}
        assert pending == {"table": weight_map["table"]}
        # Only the table waits; the replicated dense weight is just as oversized but has one copy.
        assert not (Path(save_path) / weight_map["table"]).exists()
        assert (Path(save_path) / weight_map["dense"]).exists()
        shards = sorted(p.name for p in (Path(save_path) / HF_SHARDED_DIRNAME).iterdir())
        assert len(shards) == WORLD_SIZE
        assert all(weight_map["table"].replace("model-", "") in name for name in shards)

        for resolve in (_resolve_safetensors_shards, _resolve_state_dict_iterators):
            with pytest.raises(RuntimeError, match="merge_hf_export"):
                resolve(save_path)

        save_utils.merge_pending_hf_shards(save_path)

        assert not (Path(save_path) / HF_PENDING_MERGE_FILENAME).exists()
        assert not (Path(save_path) / HF_SHARDED_DIRNAME).exists()
        loaded = _load_export(save_path)
        expected_table = torch.cat([_local_rows(rank) for rank in range(WORLD_SIZE)])
        torch.testing.assert_close(loaded["table"], expected_table, atol=0, rtol=0)
        for name, value in _replicated().items():
            torch.testing.assert_close(loaded[name], value, atol=0, rtol=0)
        _resolve_safetensors_shards(save_path)


def test_a_small_emb_table_or_an_unplanned_tensor_is_consolidated_as_before():
    state = {"table": _local_rows(0), **_replicated()}
    for model in (
        _model(SimpleNamespace(para_name="emb", placement=Replicate())),
        _model(SimpleNamespace(para_name="ep", placement=Shard(0))),
        nn.Module(),
    ):
        with tempfile.TemporaryDirectory() as save_path:
            _export(model, {**state, "table": torch.cat([state["table"]] * WORLD_SIZE)}, save_path)
            assert not (Path(save_path) / HF_PENDING_MERGE_FILENAME).exists()
            assert not (Path(save_path) / HF_SHARDED_DIRNAME).exists()
            assert set(_load_export(save_path)) == {"table", "dense", "bias"}


def test_a_merge_interrupted_before_its_marker_is_removed_can_run_again():
    table = torch.cat([_local_rows(rank) for rank in range(WORLD_SIZE)])
    with tempfile.TemporaryDirectory() as save_path:
        _export(_model(SimpleNamespace(para_name="emb", placement=Shard(0))), {"table": table}, save_path)
        with patch.object(save_utils.os, "remove", side_effect=OSError("killed")):
            with pytest.raises(OSError, match="killed"):
                save_utils.merge_pending_hf_shards(save_path)
        save_utils.merge_pending_hf_shards(save_path)
        torch.testing.assert_close(_load_export(save_path)["table"], table, atol=0, rtol=0)


def test_the_merge_script_finds_every_pending_export_below_a_root():
    spec = importlib.util.spec_from_file_location(
        "merge_hf_export", Path(__file__).resolve().parents[2] / "scripts" / "merge_hf_export.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    with tempfile.TemporaryDirectory() as root:
        for module, pending in (("ngram", True), ("llm", False), ("text_encoder", True)):
            os.makedirs(os.path.join(root, module, HF_SHARDED_DIRNAME))
            if pending:
                Path(root, module, HF_PENDING_MERGE_FILENAME).write_text("{}")
        # A marker inside the shard directory is not an export.
        Path(root, "llm", HF_SHARDED_DIRNAME, HF_PENDING_MERGE_FILENAME).write_text("{}")
        assert script.pending_exports(root) == [os.path.join(root, "ngram"), os.path.join(root, "text_encoder")]
