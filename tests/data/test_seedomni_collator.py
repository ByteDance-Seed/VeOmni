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

"""Unit + lightweight integration tests for ``SeedOmniCollator`` (Feature D1).

Two layers:
1. ``SeedOmniCollator`` standalone unit tests — covering happy path, key
   alignment, missing ``conversation_list``, empty input, and pass-through
   of arbitrary extra keys.
2. End-to-end smoke test — wires up a tiny in-memory dataset →
   ``MappingDataset`` (with the seedomni transform) →
   ``MakeMicroBatchCollator(internal=SeedOmniCollator())`` →
   ``torch.utils.data.DataLoader`` and verifies the resulting micro-batch
   shape matches the V2 contract: heterogeneous image tensors stay as a
   list (no torch.stack), ``conversation_list`` is ``list[list[ConversationItem]]``,
   and all items are :class:`ConversationItem` with ``role`` at the data boundary.

This avoids spinning up the full ``OmniTrainer`` (which would try to
load real Janus weights from HDFS) — instead we exercise the data layer
in isolation.  ``OmniTrainer._build_collate_fn`` itself is also tested
directly with a stub ``BaseTrainer`` so we don't need a model.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from veomni.data.data_collator import MakeMicroBatchCollator, SeedOmniCollator
from veomni.data.dataset import MappingDataset
from veomni.data.multimodal.seedomni_transform import process_seedomni_example
from veomni.models.seed_omni.conversation import ConversationItem


# ─────────────────────────── unit: SeedOmniCollator ───────────────────────────


def _make_sample(text="hi", role="user"):
    return {
        "conversation_list": [
            ConversationItem(type="text", value=text, role=role),
        ],
    }


def test_collator_happy_path_two_samples():
    collator = SeedOmniCollator()
    out = collator([_make_sample("a"), _make_sample("b", role="assistant")])

    assert set(out.keys()) == {"conversation_list"}
    assert isinstance(out["conversation_list"], list)
    assert len(out["conversation_list"]) == 2  # batch size

    # Each entry is the per-sample list of ConversationItem
    assert isinstance(out["conversation_list"][0][0], ConversationItem)
    assert out["conversation_list"][0][0].value == "a"
    assert out["conversation_list"][0][0].role == "user"
    assert out["conversation_list"][1][0].value == "b"
    assert out["conversation_list"][1][0].role == "assistant"


def test_collator_passes_through_extra_keys():
    """Any sample-level key (e.g. precomputed embeddings during offline
    training) is gathered as a list of length batch_size — no schema
    knowledge baked into the collator."""
    collator = SeedOmniCollator()
    samples = [
        {"conversation_list": [], "extra_tensor": torch.tensor([1, 2, 3])},
        {"conversation_list": [], "extra_tensor": torch.tensor([4, 5, 6])},
    ]
    out = collator(samples)

    assert set(out.keys()) == {"conversation_list", "extra_tensor"}
    assert len(out["extra_tensor"]) == 2
    assert torch.equal(out["extra_tensor"][0], torch.tensor([1, 2, 3]))


def test_collator_empty_features_raises():
    with pytest.raises(ValueError, match="empty feature list"):
        SeedOmniCollator()([])


def test_collator_missing_conversation_list_raises():
    with pytest.raises(KeyError, match="conversation_list"):
        SeedOmniCollator()([{"input_ids": torch.tensor([1, 2])}])


def test_collator_inconsistent_keys_raises():
    """If samples disagree on which keys they carry we fail loudly —
    silently dropping or zeroing fields is an open door for nasty bugs."""
    samples = [
        {"conversation_list": [], "extra": 1},
        {"conversation_list": []},  # missing 'extra'
    ]
    with pytest.raises(ValueError, match="all samples in a batch must carry the same key set"):
        SeedOmniCollator()(samples)


def test_collator_does_not_stack_image_tensors():
    """Heterogeneous image shapes across samples MUST stay as a list —
    the data layer can't know how to pad / stack them; the vision
    encoder's pre_forward does that after its image_processor matches
    shapes (per V2 contract)."""
    img_small = torch.zeros(3, 8, 8, dtype=torch.uint8)
    img_big = torch.zeros(3, 16, 16, dtype=torch.uint8)
    samples = [
        {
            "conversation_list": [
                ConversationItem(type="image", value=img_small, role="user"),
            ]
        },
        {
            "conversation_list": [
                ConversationItem(type="image", value=img_big, role="user"),
            ]
        },
    ]
    out = SeedOmniCollator()(samples)

    # Items should keep their original shapes.
    assert out["conversation_list"][0][0].value.shape == (3, 8, 8)
    assert out["conversation_list"][1][0].value.shape == (3, 16, 16)


# ─────────────────────────── integration: dataset → transform → collator ───────────────────────────


def test_pipeline_dataset_to_micro_batch(tmp_path):
    """Full data-layer pipeline smoke test, no model and no distributed setup.

    Mirrors the ``dyn_bsz=False`` path of ``build_native_dataloader``:
    the dataset emits ``list[dict]`` (length-1 from the transform), the
    DataLoader assembles a batch of those, and ``MakeMicroBatchCollator``
    strips the outer wrapper and feeds samples into ``SeedOmniCollator``.
    """
    # 4 source samples (2 with image, 2 text-only) — two micro-batches of 2.
    Image.new("RGB", (32, 24), (10, 20, 30)).save(tmp_path / "a.jpg")
    Image.new("RGB", (40, 30), (40, 50, 60)).save(tmp_path / "b.jpg")
    raw_data = [
        {
            "source_name": "sharegpt4v_sft",
            "conversations": [
                {"from": "human", "value": "<image>describe"},
                {"from": "gpt", "value": "img a"},
            ],
            "images": [str(tmp_path / "a.jpg")],
        },
        {
            "source_name": "sharegpt4v_sft",
            "conversations": [
                {"from": "human", "value": "say hi"},
                {"from": "gpt", "value": "hi"},
            ],
        },
        {
            "source_name": "sharegpt4v_sft",
            "conversations": [
                {"from": "human", "value": "<image>label this"},
                {"from": "gpt", "value": "img b"},
            ],
            "images": [str(tmp_path / "b.jpg")],
        },
        {
            "source_name": "sharegpt4v_sft",
            "conversations": [
                {"from": "human", "value": "what is up"},
                {"from": "gpt", "value": "all good"},
            ],
        },
    ]

    dataset = MappingDataset(raw_data, transform=process_seedomni_example)
    assert len(dataset) == 4
    # Each __getitem__ returns the [dict] wrapper from the transform.
    assert isinstance(dataset[0], list) and len(dataset[0]) == 1
    assert "conversation_list" in dataset[0][0]

    collator = MakeMicroBatchCollator(num_micro_batch=2, internal_data_collator=SeedOmniCollator())

    # Use a synchronous, single-worker DataLoader so the test is hermetic.
    loader = DataLoader(
        dataset,
        batch_size=4,  # one fetch == one optimizer step (4 samples / 2 micro-batches of 2)
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        drop_last=True,
    )

    micro_batches = next(iter(loader))
    assert isinstance(micro_batches, list) and len(micro_batches) == 2

    # First micro-batch: sample 0 (image) + sample 1 (text-only).
    mb0 = micro_batches[0]
    assert set(mb0.keys()) == {"conversation_list"}
    assert len(mb0["conversation_list"]) == 2  # micro-batch size

    # Sample 0: should contain an image item (uint8 (C, H, W)) plus text turns.
    sample0_items = mb0["conversation_list"][0]
    types = [it.type for it in sample0_items]
    assert "image" in types

    img_item = next(it for it in sample0_items if it.type == "image")
    assert isinstance(img_item.value, torch.Tensor)
    assert img_item.value.dtype == torch.uint8
    assert img_item.value.shape[0] == 3  # channels

    # Schema check: every item across both micro-batches is a ConversationItem with role.
    for mb in micro_batches:
        for sample_items in mb["conversation_list"]:
            for it in sample_items:
                assert isinstance(it, ConversationItem)
                assert it.type and it.role in ("system", "user", "assistant")


# ─────────────────────────── OmniTrainer._build_collate_fn ───────────────────────────


from veomni.trainer.omni_trainer import OmniTrainer  # noqa: E402  (deliberate, isolates the import here)


def test_omni_trainer_picks_seedomni_collator_for_seedomni_data_type():
    """Direct test of the dispatch logic: when ``data_type='seedomni'``
    the trainer must pick ``SeedOmniCollator`` (not ``MainCollator``)."""
    from types import SimpleNamespace

    # Stub the OmniTrainer just enough for ``_build_collate_fn`` —
    # we don't want to load any model or open any files.
    trainer = OmniTrainer.__new__(OmniTrainer)
    base_stub = SimpleNamespace()
    base_stub.args = SimpleNamespace(data=SimpleNamespace(data_type="seedomni"))
    trainer.base = base_stub

    trainer._build_collate_fn()

    assert isinstance(trainer.base.collate_fn, SeedOmniCollator), (
        f"expected SeedOmniCollator for data_type='seedomni', got {type(trainer.base.collate_fn).__name__}"
    )


def test_omni_trainer_falls_back_to_main_collator_for_other_data_types():
    """Non-``seedomni`` data_types must keep using ``BaseTrainer._build_collate_fn``
    (which builds ``MainCollator``).  Mirrors the existing text-only path
    so V1 contract isn't accidentally broken."""
    from types import SimpleNamespace

    trainer = OmniTrainer.__new__(OmniTrainer)
    base_stub = SimpleNamespace()
    base_stub.args = SimpleNamespace(
        data=SimpleNamespace(data_type="conversation"),
        train=SimpleNamespace(pad_to_length=False),
    )
    # Spy the fallback: replace BaseTrainer._build_collate_fn with a
    # side-effect we can detect, so we don't need a real BaseTrainer.
    called = {}

    def fake_base_build():
        called["yes"] = True
        base_stub.collate_fn = "stub_main_collator"

    base_stub._build_collate_fn = fake_base_build
    trainer.base = base_stub

    trainer._build_collate_fn()

    assert called.get("yes"), "expected delegation to base._build_collate_fn for non-seedomni data_type"
    assert trainer.base.collate_fn == "stub_main_collator"
