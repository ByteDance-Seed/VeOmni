"""Tests for ``seedomni_transform``, ``SeedOmniCollator``, and ``ModulePreprocessorBase`` contracts."""

from __future__ import annotations

import io
import pickle

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.dataset import get_length_by_labels_fn, get_length_fn
from veomni.data.seed_omni.collator import SeedOmniCollator
from veomni.data.seed_omni.seedomni_transform import (
    process_seedomni_cached_example,
    process_seedomni_example,
)
from veomni.models.seed_omni.modules.module_processing_base import ModulePreprocessorBase
from veomni.models.seed_omni.processing_omni import OmniProcessor
from veomni.models.seed_omni.utils.conversation import (
    _IMG_TAG_KEY,
    _OMNI_TOKEN_LEN,
    ConversationItem,
    iter_desired_items,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _DummyImageNormalizePreprocessor(ModulePreprocessorBase):
    """Picklable test preprocessor that normalizes user images and appends a per-sample dummy anchor."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16, patch_size: int = 2) -> None:
        self.dtype = dtype
        self.patch_size = patch_size

    def preprocess_conversations(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs
    ) -> None:
        del kwargs
        for sample in conversation_list:
            user_images = [
                it for it in iter_desired_items([sample], types=["image"], roles=["user"]) if not it.is_dummy
            ]
            if user_images:
                for item in user_images:
                    item.value = item.value.to(dtype=self.dtype) / 255.0
                    h, w = item.value.shape[-2], item.value.shape[-1]
                    item.meta[_OMNI_TOKEN_LEN] = (h // self.patch_size) * (w // self.patch_size)
            elif not inference:
                sample.append(
                    ConversationItem(
                        type="image",
                        value=torch.zeros(3, 4, 4, dtype=self.dtype),
                        role="user",
                        is_dummy=True,
                        meta={_IMG_TAG_KEY: "und", _OMNI_TOKEN_LEN: 4},
                    )
                )


class _DummyTextTokenizePreprocessor(ModulePreprocessorBase):
    """Picklable test text preprocessor that converts text strings to token tensors with labels and _omni_token_len."""

    def preprocess_conversations(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs
    ) -> None:
        del inference, kwargs
        for sample in conversation_list:
            for item in sample:
                if item.type != "text" or item.is_dummy:
                    continue
                token_ids = torch.tensor([ord(ch) for ch in str(item.value)], dtype=torch.long)
                item.value = token_ids
                item.meta["attention_mask"] = torch.ones_like(token_ids)
                item.meta["labels"] = (
                    token_ids.clone() if item.role == "assistant" else torch.full_like(token_ids, -100)
                )
                item.meta[_OMNI_TOKEN_LEN] = int(token_ids.numel())


def _png_bytes(width: int = 8, height: int = 8) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_collator_strict_key_alignment_rejects_mismatched_sample_keys():
    collator = SeedOmniCollator()
    features = [
        {"conversation_list": [ConversationItem(type="text", value="a", role="user")], "ds_idx": 0},
        {"conversation_list": [ConversationItem(type="text", value="b", role="user")]},
    ]
    with pytest.raises(ValueError, match="sample 1 has keys"):
        collator(features)


def test_collator_preserves_mixed_resolution_images_without_stacking():
    """Mixed-resolution images survive preprocessing and collation unstacked and normalized per item."""
    pre = _DummyImageNormalizePreprocessor(dtype=torch.bfloat16)
    reloaded_pre = pickle.loads(pickle.dumps(pre))
    processor = OmniProcessor({"vision": reloaded_pre})
    collator = SeedOmniCollator()

    small_img = torch.randint(0, 255, (3, 14, 14), dtype=torch.uint8)
    wide_img = torch.randint(0, 255, (3, 28, 42), dtype=torch.uint8)
    tall_img = torch.randint(0, 255, (3, 56, 20), dtype=torch.uint8)

    features = [
        {
            **processor.preprocess(
                [
                    ConversationItem(type="image", value=small_img, role="user"),
                    ConversationItem(type="image", value=wide_img, role="user"),
                    ConversationItem(type="text", value="compare these", role="user"),
                ]
            ),
            "ds_idx": 0,
        },
        {
            **processor.preprocess(
                [
                    ConversationItem(type="image", value=tall_img, role="user"),
                    ConversationItem(type="text", value="describe this", role="user"),
                ]
            ),
            "ds_idx": 1,
        },
    ]

    batch = collator(features)

    assert set(batch.keys()) == {"conversation_list", "ds_idx"}
    assert batch["ds_idx"] == [0, 1]
    s0_imgs = [it.value for it in batch["conversation_list"][0] if it.type == "image"]
    s1_imgs = [it.value for it in batch["conversation_list"][1] if it.type == "image"]
    assert [tuple(t.shape) for t in s0_imgs] == [(3, 14, 14), (3, 28, 42)]
    assert [tuple(t.shape) for t in s1_imgs] == [(3, 56, 20)]
    assert all(t.dtype == torch.bfloat16 for t in [*s0_imgs, *s1_imgs])


def test_module_preprocessor_appends_per_sample_dummy_when_modality_absent():
    processor = OmniProcessor({"vision": _DummyImageNormalizePreprocessor(dtype=torch.bfloat16)})
    collator = SeedOmniCollator()
    features = [
        processor.preprocess([ConversationItem(type="text", value="hi", role="user")]),
        processor.preprocess([ConversationItem(type="text", value="yo", role="user")]),
    ]

    batch = collator(features)

    dummies = [
        it for it in iter_desired_items(batch["conversation_list"], types=["image"], roles=["user"]) if it.is_dummy
    ]
    assert len(dummies) == 2
    assert dummies[0] is batch["conversation_list"][0][-1]
    assert dummies[1] is batch["conversation_list"][1][-1]
    assert all(d.value.shape == (3, 4, 4) and d.meta[_OMNI_TOKEN_LEN] == 4 for d in dummies)


def test_seedomni_transform_runs_processor_and_supports_dyn_bsz_lengths():
    processor = OmniProcessor(
        {
            "vision": _DummyImageNormalizePreprocessor(dtype=torch.bfloat16, patch_size=2),
            "text": _DummyTextTokenizePreprocessor(),
        }
    )

    # 1) Image + text sample via process_seedomni_example (8x8 image -> 16 tokens; "hi" -> 2 user tokens; "ok!" -> 3 assistant tokens).
    mm_example = {
        "source_name": "sharegpt4v_sft",
        "images": [_png_bytes(8, 8)],
        "conversations": [
            {"from": "human", "value": "<image>hi"},
            {"from": "gpt", "value": "ok!"},
        ],
    }
    (mm_sample,) = process_seedomni_example(mm_example, processor=processor)
    assert get_length_fn(mm_sample) == 16 + 2 + 3
    assert get_length_by_labels_fn(mm_sample) == 3

    # 2) Text-only cached sample via process_seedomni_cached_example (vision appends 4-token dummy; "abcd" -> 4 user; "ef" -> 2 assistant).
    cached_row = {
        "conversation_list": pickle.dumps(
            [
                ConversationItem(type="text", value="abcd", role="user"),
                ConversationItem(type="text", value="ef", role="assistant"),
            ]
        )
    }
    (cached_sample,) = process_seedomni_cached_example(cached_row, processor=processor)
    # Total includes the 4-token dummy; effective excludes dummy and user prompt.
    assert get_length_fn(cached_sample) == 4 + 2 + 4
    assert get_length_by_labels_fn(cached_sample) == 2


def test_dyn_bsz_skips_seedomni_samples_exceeding_token_budget():
    from torch.utils.data import IterableDataset

    from veomni.data.dataset import DynamicBatchingSizeDataset
    from veomni.data.dynamic_batching import TextBatchingStrategy

    processor = OmniProcessor(
        {
            "vision": _DummyImageNormalizePreprocessor(dtype=torch.bfloat16, patch_size=2),
            "text": _DummyTextTokenizePreprocessor(),
        }
    )

    short_example = {
        "source_name": "sharegpt4v_sft",
        "images": [_png_bytes(4, 4)],  # 4 visual tokens
        "conversations": [
            {"from": "human", "value": "<image>hi"},  # 2 text tokens
            {"from": "gpt", "value": "ok"},  # 2 text tokens -> 8 tokens total
        ],
    }
    long_example = {
        "source_name": "sharegpt4v_sft",
        "images": [_png_bytes(8, 8)],  # 16 visual tokens
        "conversations": [
            {"from": "human", "value": "<image>hello"},  # 5 text tokens
            {"from": "gpt", "value": "world"},  # 5 text tokens -> 26 tokens total
        ],
    }

    (short_sample,) = process_seedomni_example(short_example, processor=processor)
    (long_sample,) = process_seedomni_example(long_example, processor=processor)
    assert get_length_fn(short_sample) == 8
    assert get_length_fn(long_sample) == 26

    # 1) Main-process dyn_bsz (TextBatchingStrategy) skips overlong samples (> token_micro_bsz).
    strategy = TextBatchingStrategy(token_micro_bsz=10, buffer_size=1)
    strategy.put_item(long_sample)
    assert len(strategy.buffer) == 0
    strategy.put_item(short_sample)
    assert len(strategy.buffer) == 1
    micro_batch = strategy.get_micro_batch(step=0)
    assert len(micro_batch) == 1 and get_length_fn(micro_batch[0]) == 8

    # 2) Worker-process dyn_bsz (DynamicBatchingSizeDataset) also skips overlong samples.
    class _Stream(IterableDataset):
        def __iter__(self):
            yield [long_sample]
            yield [short_sample]

    dyn_ds = DynamicBatchingSizeDataset(
        dataset=_Stream(),
        micro_batch_seq_length=10,
        ready_for_micro_batch_threshold=1,
        dynamic_batching_collate_fn=SeedOmniCollator(),
        save_by_idx=False,
    )
    batches = list(dyn_ds)
    assert len(batches) == 1
    assert len(batches[0]["conversation_list"]) == 1


def _two_rank_rejected_shard_worker(rank: int, rendezvous: str, out_dir: str) -> None:
    import json
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy
    from veomni.trainer.base import VeOmniIter

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=30),
    )
    try:
        overlong = {
            "conversation_list": [
                ConversationItem(
                    type="text",
                    value=torch.ones(26, dtype=torch.long),
                    role="user",
                    meta={_OMNI_TOKEN_LEN: 26},
                )
            ]
        }
        valid = {
            "conversation_list": [
                ConversationItem(
                    type="text",
                    value=torch.ones(8, dtype=torch.long),
                    role="user",
                    meta={_OMNI_TOKEN_LEN: 8},
                )
            ]
        }
        # Rank 0's shard contains only rejected (> 10 tokens) samples; Rank 1's shard has valid samples.
        shard = [[overlong], [overlong]] if rank == 0 else [[valid], [valid]]
        reports = {}
        for use_prefetcher in (False, True):
            dl = DynamicBatchSizeDataLoader(
                shard,
                batching_strategy=TextBatchingStrategy(token_micro_bsz=10, buffer_size=1),
                collate_fn=SeedOmniCollator(),
                num_micro_batch=1,
                length=2,
                drop_last=True,
            )
            data_iter = VeOmniIter(dl, use_background_prefetcher=use_prefetcher)
            try:
                batches = list(data_iter)
            finally:
                data_iter.stop()
            reports[f"prefetch_{int(use_prefetcher)}"] = len(batches)
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(reports), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_dyn_bsz_dataloader_stops_coordinately_when_rank_shard_has_only_rejected_samples(tmp_path):
    """When one rank's shard contains only over-budget samples, DynamicBatchSizeDataLoader
    must detect no progress (not loop forever) and coordinate with peer ranks to end together
    both on direct iteration and through VeOmniIter(use_background_prefetcher=True)."""
    import json

    import torch.multiprocessing as mp

    mp.spawn(
        _two_rank_rejected_shard_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        report = json.loads((tmp_path / f"rank{rank}.json").read_text(encoding="utf-8"))
        assert report == {"prefetch_0": 0, "prefetch_1": 0}
