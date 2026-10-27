# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DPO loader buffers must not silently cross data-semantics versions."""

import copy
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.data_transform import process_dpo_example
from veomni.data.dataset import DynamicBatchingSizeDataset, get_length_fn
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback
from veomni.trainer.text_dpo_trainer import TextDPOTrainer
from veomni.utils.dpo_provenance import DPO_FEATURE_KEY, DPOFeatureSegments


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _trainer(enabled, allow_legacy=False):
    trainer = TextDPOTrainer.__new__(TextDPOTrainer)
    config = SimpleNamespace(explicit_segment_provenance=enabled, allow_legacy_data_resume=allow_legacy)
    checkpoint = SimpleNamespace(load_path="/unused")
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(dpo_config=config, train=SimpleNamespace(checkpoint=checkpoint, global_rank=0)),
        train_dataloader=MagicMock(),
    )
    return trainer


@pytest.mark.parametrize(
    "current,saved,allow_legacy,raises",
    [
        (False, None, False, True),
        (False, None, True, False),
        (True, None, True, True),
        (True, {"version": 2, "enabled": True, "plaintext_transform_version": 1}, False, False),
        (False, {"version": 2, "enabled": False, "plaintext_transform_version": 1}, False, False),
        (False, {"version": 2, "enabled": True, "plaintext_transform_version": 1}, True, True),
        (True, {"version": 1, "enabled": True}, True, True),
        (True, {"version": 2, "enabled": 1, "plaintext_transform_version": 1}, True, True),
        (True, {"version": 2, "enabled": True, "plaintext_transform_version": 2}, True, True),
    ],
)
def test_checkpoint_mode_contract(current, saved, allow_legacy, raises):
    trainer = _trainer(current, allow_legacy)
    if raises:
        with pytest.raises(ValueError):
            trainer.validate_dpo_provenance_checkpoint(saved)
    else:
        trainer.validate_dpo_provenance_checkpoint(saved)


def test_checkpoint_mode_is_checked_before_loader_restore(monkeypatch):
    trainer = _trainer(True)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {"global_step": 3, "train_dataloader": {"cursor": 7}}
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    with pytest.raises((ValueError, RuntimeError), match="trainer state|provenance"):
        callback.load_global_state()
    trainer.train_dataloader.load_state_dict.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_missing_checkpoint_trainer_state_does_not_assume_safe_mode(monkeypatch, enabled):
    trainer = _trainer(enabled)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: None
    monkeypatch.setattr("veomni.checkpoint.legacy_v0_1_12.apply_legacy_global_state", lambda *_: None)
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    with pytest.raises((ValueError, RuntimeError), match="trainer state|provenance"):
        callback.load_global_state()
    trainer.train_dataloader.load_state_dict.assert_not_called()


def test_explicit_legacy_resume_keeps_old_missing_state_fallback(monkeypatch):
    trainer = _trainer(False, allow_legacy=True)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: None
    monkeypatch.setattr("veomni.checkpoint.legacy_v0_1_12.apply_legacy_global_state", lambda *_: None)
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    assert callback.load_global_state() is None
    trainer.train_dataloader.load_state_dict.assert_not_called()


def test_full_sample_buffer_serialization_keeps_typed_provenance():
    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return list(range(len(text)))

    feature = process_dpo_example(
        {"chosen": "abc", "rejected": "d"}, tokenizer=Tokenizer(), record_segment_provenance=True
    )[0]
    buffer = io.BytesIO()
    torch.save(copy.deepcopy({"buffer": [feature]}), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)["buffer"][0]
    assert restored[DPO_FEATURE_KEY] == DPOFeatureSegments(3, 1)


@pytest.mark.parametrize("save_by_idx", [False, True])
def test_dynamic_batch_buffer_restore_preserves_provenance(save_by_idx):
    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return list(range(len(text)))

    class Dataset(torch.utils.data.IterableDataset):
        output_index_for_resume = False

        def get_item(self, index):
            return process_dpo_example(
                {"chosen": "abc", "rejected": "d"}, tokenizer=Tokenizer(), record_segment_provenance=True
            )

        def __iter__(self):
            return iter([])

    def build():
        return DynamicBatchingSizeDataset(
            Dataset(),
            micro_batch_seq_length=8,
            ready_for_micro_batch_threshold=2,
            dynamic_batching_collate_fn=lambda x: x,
            get_length_fn=get_length_fn,
            save_by_idx=save_by_idx,
        )

    original = build()
    feature = original.dataset.get_item(0)[0]
    original._buffer = [(feature, 4, 4)]
    original._buffer_of_output_index = [(0, 0)]
    original._buffer_token_count = original._buffer_physical_token_count = 4
    serialized = io.BytesIO()
    torch.save(original.state_dict(), serialized)
    serialized.seek(0)
    resumed = build()
    resumed.load_state_dict(torch.load(serialized, weights_only=False))
    assert resumed._buffer[0][0][DPO_FEATURE_KEY] == DPOFeatureSegments(3, 1)
