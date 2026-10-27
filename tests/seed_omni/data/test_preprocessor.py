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
