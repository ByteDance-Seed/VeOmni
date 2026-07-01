from __future__ import annotations

import pytest

from veomni.arguments import OmniTrainingArguments


def test_omni_training_args_defaults_to_train_workflow() -> None:
    args = OmniTrainingArguments()

    assert args.train_type == "train"


def test_omni_training_args_requires_offline_cache_dir() -> None:
    with pytest.raises(ValueError, match="offline_cache_dir"):
        OmniTrainingArguments(train_type="offline_cache")


def test_omni_training_args_accepts_offline_cache_dir() -> None:
    args = OmniTrainingArguments(train_type="offline_cache", offline_cache_dir="/tmp/cache")

    assert args.train_type == "offline_cache"
    assert args.offline_cache_dir == "/tmp/cache"


def test_omni_training_args_rejects_reserved_train_and_cache() -> None:
    with pytest.raises(NotImplementedError, match="train_and_cache"):
        OmniTrainingArguments(train_type="train_and_cache")


def test_omni_training_args_rejects_unknown_train_type() -> None:
    with pytest.raises(ValueError, match="Unknown train.train_type 'other'"):
        OmniTrainingArguments(train_type="other")
