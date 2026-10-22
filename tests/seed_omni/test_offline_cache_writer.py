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
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.seed_omni.seedomni_transform import process_seedomni_cached_example
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.models.seed_omni.utils.offline_cache import SeedOmniOfflineCacheWriter, _first_rank_of_its_dp_rank


_CI_CPU_4 = LaunchSpec(nproc=4, device="cpu")

_CI_CPU_2 = LaunchSpec(nproc=2, device="cpu")


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
    writer.open()
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


def test_finalize_keeps_the_shard_names_resume_reads_ownership_from(tmp_path) -> None:
    """Renaming in finalize could leave names that resume attributes to the wrong rank if it were interrupted."""
    writer = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=1)
    writer.open()
    writer.rank = 7
    writer.world_size = 8

    writer.save_conversation_list([[ConversationItem(type="text", value="prompt", role="user")]])
    writer.finalize()

    assert sorted(path.name for path in tmp_path.glob("shard_*.parquet")) == ["shard_000007.parquet"]
    assert (tmp_path / ".offline_cache_complete").exists()


@pytest.mark.parametrize(
    ("dp_ranks", "writers"),
    [([0, 1, 2, 3], [0, 1, 2, 3]), ([0, 0, 1, 1], [0, 2]), ([0, 1, 0, 1], [0, 1])],
)
def test_one_rank_per_dp_rank_writes(dp_ranks, writers) -> None:
    """SP / CP / TP ranks share a ``dp_rank`` and its batch; a second writer would race on the same shard."""
    assert [rank for rank in range(len(dp_ranks)) if _first_rank_of_its_dp_rank(dp_ranks, rank)] == writers


def test_a_rank_that_does_not_write_leaves_no_shard(tmp_path) -> None:
    writer = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=1)
    writer.open()
    writer.writes = False

    writer.save_conversation_list([[ConversationItem(type="text", value="prompt", role="user")]])
    writer.finalize()

    assert not list(tmp_path.glob("shard_*.parquet"))


def _text(value: str) -> list[list[ConversationItem]]:
    return [[ConversationItem(type="text", value=value, role="user")]]


def _cached_values(cache_dir) -> list[str]:
    return sorted(
        process_seedomni_cached_example(row)[0]["conversation_list"][0].value
        for shard in sorted(cache_dir.glob("shard_*.parquet"))
        for row in load_dataset("parquet", data_files=[str(shard)], split="train")
    )


def test_a_resumed_writer_keeps_every_row_the_cursor_covers_and_drops_the_replayed_ones(tmp_path) -> None:
    first = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=2)
    first.open()
    for value in ("s0", "s1", "s2"):
        first.save_conversation_list(_text(value))
    # "s2" is still buffered: the checkpoint has to put it on disk.
    state = first.state_dict()
    assert state == {"shard_index": 2, "rows_written": 3, "world_size": 1}
    # Written after the checkpoint, then the run dies; the resumed cursor replays these steps.
    for value in ("s3", "s4"):
        first.save_conversation_list(_text(value))
    assert (tmp_path / "shard_000002.parquet").exists()

    resumed = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=2)
    resumed.load_state_dict(state)
    resumed.open()
    assert sorted(path.name for path in tmp_path.glob("shard_*.parquet")) == [
        "shard_000000.parquet",
        "shard_000001.parquet",
    ]
    for value in ("s3", "s4", "s5"):
        resumed.save_conversation_list(_text(value))
    resumed.finalize()

    assert _cached_values(tmp_path) == ["s0", "s1", "s2", "s3", "s4", "s5"]
    assert resumed.rows_written == 6


def test_a_finalized_cache_cannot_be_resumed_into(tmp_path) -> None:
    writer = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=1)
    writer.open()
    writer.save_conversation_list(_text("s0"))
    state = writer.state_dict()
    writer.finalize()

    resumed = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=1)
    resumed.load_state_dict(state)
    with pytest.raises(FileExistsError, match="already finalized"):
        resumed.open()


def test_a_cache_cannot_be_resumed_under_another_dp_size(tmp_path) -> None:
    writer = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=1)
    writer.open()
    writer.save_conversation_list(_text("s0"))
    state = {**writer.state_dict(), "world_size": 2}

    resumed = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=1)
    resumed.load_state_dict(state)
    with pytest.raises(ValueError, match="data-parallel size 2"):
        resumed.open()


def test_a_fresh_writer_refuses_a_directory_with_shards(tmp_path) -> None:
    (tmp_path / "shard_000000.parquet").write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="already contains"):
        SeedOmniOfflineCacheWriter(str(tmp_path)).open()


def test_a_fresh_writer_refuses_a_leftover_completion_marker(tmp_path) -> None:
    """With only the shards deleted, this run would later refuse its own resume as already finalized."""
    (tmp_path / ".offline_cache_complete").touch()

    with pytest.raises(FileExistsError, match="already contains"):
        SeedOmniOfflineCacheWriter(str(tmp_path)).open()


def test_nothing_is_written_before_the_directory_check(tmp_path) -> None:
    writer = SeedOmniOfflineCacheWriter(str(tmp_path), max_rows_per_shard=1)

    with pytest.raises(RuntimeError, match="open"):
        writer.save_conversation_list(_text("s0"))


def _sp_pair_rank_main(rank: int, rendezvous: str, cache_dir: str, out_dir: str) -> None:
    import veomni.models.seed_omni.utils.offline_cache as offline_cache

    offline_cache.get_parallel_state = lambda: SimpleNamespace(dp_rank=rank // 2, dp_size=2)
    dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous}", world_size=4, rank=rank)
    try:
        writer = SeedOmniOfflineCacheWriter(cache_dir, max_rows_per_shard=1)
        writer.open()
        for step in range(2):
            value = f"dp{rank // 2}-step{step}"
            writer.save_conversation_list([[ConversationItem(type="text", value=value, role="user")]])
        writer.finalize()
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump({"writes": writer.writes}, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_CPU_4)
def test_sp_peers_write_one_copy_of_their_batch(tmp_path) -> None:
    """Two SP pairs: ranks 0/1 share dp_rank 0 and ranks 2/3 share dp_rank 1."""
    cache_dir = tmp_path / "cache"
    mp.spawn(
        _sp_pair_rank_main, args=(str(tmp_path / "rendezvous"), str(cache_dir), str(tmp_path)), nprocs=_CI_CPU_4.nproc
    )

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


def _existing_shard_check_rank_main(rank: int, rendezvous: str, cache_dir: str, out_dir: str) -> None:
    """Rank 0 plants a shard after a barrier that only the other ranks wait on first.

    The writer must all-gather the check and barrier after a clean result, so a
    late rank cannot see an early rank's first flush as an init-time conflict.
    """
    import veomni.models.seed_omni.utils.offline_cache as offline_cache

    offline_cache.get_parallel_state = lambda: SimpleNamespace(dp_rank=rank, dp_size=2)
    dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous}", world_size=2, rank=rank)
    try:
        # Both ranks open writers against an empty directory; the barrier
        # inside the check keeps them aligned before either can flush.
        writer = SeedOmniOfflineCacheWriter(cache_dir, max_rows_per_shard=1)
        writer.open()
        writer.save_conversation_list([[ConversationItem(type="text", value=f"r{rank}", role="user")]])
        writer.finalize()
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump({"ok": True}, f)
    except Exception as exc:  # noqa: BLE001 - surface to the parent process
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, f)
        raise
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_existing_shard_check_agrees_across_ranks_before_any_flush(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    mp.spawn(
        _existing_shard_check_rank_main,
        args=(str(tmp_path / "rendezvous"), str(cache_dir), str(tmp_path)),
        nprocs=_CI_CPU_2.nproc,
    )
    for rank in range(2):
        payload = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert payload["ok"], payload
    assert sorted(path.name for path in cache_dir.glob("shard_*.parquet")) == [
        "shard_000000.parquet",
        "shard_000001.parquet",
    ]


def _existing_shard_reject_rank_main(rank: int, rendezvous: str, cache_dirs: list[str], out_dir: str) -> None:
    import veomni.models.seed_omni.utils.offline_cache as offline_cache

    offline_cache.get_parallel_state = lambda: SimpleNamespace(dp_rank=rank, dp_size=2)
    # A rank that skipped the all-gathered verdict would sit in the post-check
    # barrier alone; the timeout turns that hang into a failure.
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=60),
    )
    try:
        try:
            SeedOmniOfflineCacheWriter(cache_dirs[rank], max_rows_per_shard=1).open()
        except FileExistsError as exc:
            with open(f"{out_dir}/rank{rank}.json", "w") as f:
                json.dump({"raised": True, "msg": str(exc)}, f)
        else:
            with open(f"{out_dir}/rank{rank}.json", "w") as f:
                json.dump({"raised": False}, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_existing_shard_check_fails_on_every_rank_together(tmp_path) -> None:
    """A shard only rank 0 can see must still fail rank 1, not leave it waiting."""
    cache_dirs = [tmp_path / "cache_rank0", tmp_path / "cache_rank1"]
    for cache_dir in cache_dirs:
        cache_dir.mkdir()
    (cache_dirs[0] / "shard_000000.parquet").write_bytes(b"stale")

    mp.spawn(
        _existing_shard_reject_rank_main,
        args=(str(tmp_path / "rendezvous"), [str(d) for d in cache_dirs], str(tmp_path)),
        nprocs=_CI_CPU_2.nproc,
    )
    for rank in range(2):
        payload = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert payload["raised"] is True
