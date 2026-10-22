"""Tests for ``SeedOmniCollator`` and ``ModulePreprocessorBase`` contracts (#1063)."""

from __future__ import annotations

import pickle

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.seed_omni.collator import SeedOmniCollator
from veomni.models.seed_omni.modules.module_processing_base import ModulePreprocessorBase
from veomni.models.seed_omni.processing_omni import OmniProcessor
from veomni.models.seed_omni.utils.conversation import (
    _IMG_TAG_KEY,
    ConversationItem,
    iter_desired_items,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _DummyImageNormalizePreprocessor(ModulePreprocessorBase):
    """Picklable test preprocessor that normalizes user images and appends a batch anchor."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        self.dtype = dtype

    def preprocess_conversations(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs
    ) -> None:
        del kwargs
        user_images = [
            it for it in iter_desired_items(conversation_list, types=["image"], roles=["user"]) if not it.is_dummy
        ]
        for item in user_images:
            item.value = item.value.to(dtype=self.dtype) / 255.0
        if not user_images and not inference:
            self.append_batch_anchor(
                conversation_list,
                ConversationItem(
                    type="image",
                    value=torch.zeros(3, 4, 4, dtype=self.dtype),
                    role="user",
                    is_dummy=True,
                    meta={_IMG_TAG_KEY: "und"},
                ),
            )


def test_collator_strict_key_alignment_rejects_mismatched_sample_keys():
    collator = SeedOmniCollator(processor=OmniProcessor({}))
    features = [
        {"conversation_list": [ConversationItem(type="text", value="a", role="user")], "ds_idx": 0},
        {"conversation_list": [ConversationItem(type="text", value="b", role="user")]},
    ]
    with pytest.raises(ValueError, match="sample 1 has keys"):
        collator(features)


def test_collator_preserves_mixed_resolution_images_without_stacking():
    """Mixed-resolution images survive collation unstacked and normalized per item."""
    pre = _DummyImageNormalizePreprocessor(dtype=torch.bfloat16)
    reloaded_pre = pickle.loads(pickle.dumps(pre))
    collator = SeedOmniCollator(processor=OmniProcessor({"vision": reloaded_pre}))

    small_img = torch.randint(0, 255, (3, 14, 14), dtype=torch.uint8)
    wide_img = torch.randint(0, 255, (3, 28, 42), dtype=torch.uint8)
    tall_img = torch.randint(0, 255, (3, 56, 20), dtype=torch.uint8)

    features = [
        {
            "conversation_list": [
                ConversationItem(type="image", value=small_img, role="user"),
                ConversationItem(type="image", value=wide_img, role="user"),
                ConversationItem(type="text", value="compare these", role="user"),
            ],
            "ds_idx": 0,
        },
        {
            "conversation_list": [
                ConversationItem(type="image", value=tall_img, role="user"),
                ConversationItem(type="text", value="describe this", role="user"),
            ],
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


def test_module_preprocessor_base_appends_single_batch_anchor_when_modality_absent():
    pre = _DummyImageNormalizePreprocessor(dtype=torch.bfloat16)
    collator = SeedOmniCollator(processor=OmniProcessor({"vision": pre}))
    features = [
        {"conversation_list": [ConversationItem(type="text", value="hi", role="user")]},
        {"conversation_list": [ConversationItem(type="text", value="yo", role="user")]},
    ]

    batch = collator(features)

    dummies = [
        it for it in iter_desired_items(batch["conversation_list"], types=["image"], roles=["user"]) if it.is_dummy
    ]
    assert len(dummies) == 1
    assert dummies[0] is batch["conversation_list"][0][-1]
    assert dummies[0].value.shape == (3, 4, 4)
