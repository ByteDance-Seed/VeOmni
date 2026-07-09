from __future__ import annotations

import pickle

import torch
from datasets import load_dataset

from veomni.data.seed_omni.seedomni_transform import process_seedomni_cached_example
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.models.seed_omni.utils.offline_cache import SeedOmniOfflineCacheWriter


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
        role="dummy",
        source="bagel_vae_context",
    )

    writer.save_conversation_list([[real_text, dummy, real_cache]])
    writer.flush()

    files = sorted(tmp_path.glob("shard_000000.parquet"))
    assert len(files) == 1
    dataset = load_dataset("parquet", data_files=[str(files[0])], split="train")
    restored = process_seedomni_cached_example(dataset[0])[0]["conversation_list"]

    assert [item.role for item in restored] == ["user", "dummy", "assistant"]
    assert restored[1].type == "image"
    assert restored[1].source == "bagel_vae_context"
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
