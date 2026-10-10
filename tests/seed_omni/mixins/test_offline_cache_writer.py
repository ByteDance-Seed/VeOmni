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

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset

from veomni.data.seed_omni.seedomni_transform import process_seedomni_cached_example
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.models.seed_omni.utils.offline_cache import SeedOmniOfflineCacheWriter, _first_rank_of_its_dp_rank


def test_seedomni_cached_transform_unpickles_conversation_list() -> None:
    conversation = [
        ConversationItem(
            type="image",
            value=torch.ones(2, 3, 4, 4),
            role="assistant",
            meta={"cache": "test_cache"},
        )
    ]

    out = process_seedomni_cached_example({"conversation_list": pickle.dumps(conversation)})

    restored = out[0]["conversation_list"]
    assert restored[0].type == "image"
    assert torch.equal(restored[0].value, conversation[0].value)
    assert restored[0].meta == {"cache": "test_cache"}


def test_offline_cache_writer_preserves_dummy_and_encoded_cache(tmp_path) -> None:
    writer = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=1)
    real_text = ConversationItem(type="text", value="prompt", role="user")
    real_cache = ConversationItem(
        type="image",
        value=torch.arange(8, dtype=torch.float32).view(2, 1, 2, 2),
        role="assistant",
        meta={"cache": "test_cache"},
    )
    dummy = ConversationItem(
        type="image",
        value=torch.zeros(1),
        role="user",
        is_dummy=True,
    )

    writer.save_conversation_list([[real_text, dummy, real_cache]])
    writer.flush()

    files = sorted(tmp_path.glob("shard_000000.parquet"))
    assert len(files) == 1
    dataset = load_dataset("parquet", data_files=[str(files[0])], split="train")
    restored = process_seedomni_cached_example(dataset[0])[0]["conversation_list"]

    assert [item.role for item in restored] == ["user", "user", "assistant"]
    assert [item.is_dummy for item in restored] == [False, True, False]
    assert restored[1].type == "image"
    assert torch.equal(restored[1].value, dummy.value)
    assert restored[1].meta == {}
    assert restored[2].type == "image"
    assert torch.equal(restored[2].value, real_cache.value)
    assert restored[2].meta == {"cache": "test_cache"}


def test_offline_cache_writer_finalize_compacts_shard_numbers(tmp_path) -> None:
    writer = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=1)
    writer.rank = 7
    writer.world_size = 8

    writer.save_conversation_list([[ConversationItem(type="text", value="prompt", role="user")]])
    assert (tmp_path / "shard_000007.parquet").exists()

    writer.finalize()

    assert not (tmp_path / "shard_000007.parquet").exists()
    assert (tmp_path / "shard_000000.parquet").exists()


@pytest.mark.parametrize(
    ("dp_ranks", "writers"),
    [([0, 1, 2, 3], [0, 1, 2, 3]), ([0, 0, 1, 1], [0, 2]), ([0, 1, 0, 1], [0, 1])],
)
def test_one_rank_per_dp_rank_writes(dp_ranks, writers) -> None:
    """SP / CP / TP ranks share a ``dp_rank`` and its batch; a second writer would race on the same shard."""
    assert [rank for rank in range(len(dp_ranks)) if _first_rank_of_its_dp_rank(dp_ranks, rank)] == writers


def test_a_rank_that_does_not_write_leaves_no_shard(tmp_path) -> None:
    writer = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=1)
    writer.writes = False

    writer.save_conversation_list([[ConversationItem(type="text", value="prompt", role="user")]])
    writer.finalize()

    assert not list(tmp_path.glob("shard_*.parquet"))


def _sp_pair_rank_main(rank: int, rendezvous: str, cache_dir: str, out_dir: str) -> None:
    import veomni.models.seed_omni.utils.offline_cache as offline_cache

    offline_cache.get_parallel_state = lambda: SimpleNamespace(dp_rank=rank // 2, dp_size=2)
    dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous}", world_size=4, rank=rank)
    try:
        writer = SeedOmniOfflineCacheWriter(cache_dir, max_rows_per_shard=1)
        for step in range(2):
            value = f"dp{rank // 2}-step{step}"
            writer.save_conversation_list([[ConversationItem(type="text", value=value, role="user")]])
        writer.finalize()
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump({"writes": writer.writes}, f)
    finally:
        dist.destroy_process_group()


def test_sp_peers_write_one_copy_of_their_batch(tmp_path) -> None:
    """Two SP pairs: ranks 0/1 share dp_rank 0 and ranks 2/3 share dp_rank 1."""
    cache_dir = tmp_path / "cache"
    mp.spawn(_sp_pair_rank_main, args=(str(tmp_path / "rendezvous"), str(cache_dir), str(tmp_path)), nprocs=4)

    writes = [json.loads((tmp_path / f"rank{rank}.json").read_text())["writes"] for rank in range(4)]
    assert writes == [True, False, True, False]
    shards = sorted(path.name for path in cache_dir.glob("shard_*.parquet"))
    assert shards == [f"shard_{i:06d}.parquet" for i in range(4)]
    values = sorted(
        process_seedomni_cached_example(row)[0]["conversation_list"][0].value
        for shard in shards
        for row in load_dataset("parquet", data_files=[str(cache_dir / shard)], split="train")
    )
    assert values == ["dp0-step0", "dp0-step1", "dp1-step0", "dp1-step1"]
