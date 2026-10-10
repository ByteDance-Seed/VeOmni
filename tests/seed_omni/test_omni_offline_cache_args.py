from __future__ import annotations

import pytest

from veomni.arguments.omni_arguments_types import OmniDataArguments, OmniTrainingArguments


def test_omni_training_args_default_to_online_encoding() -> None:
    args = OmniTrainingArguments()

    assert args.cache_mode == "full"


def test_omni_training_args_requires_offline_cache_dir() -> None:
    with pytest.raises(ValueError, match="offline_cache_dir"):
        OmniTrainingArguments(cache_mode="encode_only")


def test_omni_training_args_accepts_offline_cache_dir() -> None:
    args = OmniTrainingArguments(cache_mode="encode_only", offline_cache_dir="/tmp/cache")

    assert args.cache_mode == "encode_only"
    assert args.offline_cache_dir == "/tmp/cache"


def test_omni_training_args_trains_from_a_cache_without_a_cache_dir() -> None:
    assert OmniTrainingArguments(cache_mode="process_only").cache_mode == "process_only"


@pytest.mark.parametrize("cache_mode", ["offline_cache", "train_with_cache", "other"])
def test_omni_training_args_rejects_unknown_cache_mode(cache_mode) -> None:
    with pytest.raises(ValueError, match=f"Unknown train.cache_mode '{cache_mode}'"):
        OmniTrainingArguments(cache_mode=cache_mode)


@pytest.mark.parametrize(
    ("cache_mode", "support_cache", "expected"),
    [
        ("encode_only", True, "encode_only"),
        ("process_only", True, "process_only"),
        ("full", True, "full"),
        ("encode_only", False, "full"),
        ("process_only", False, "full"),
    ],
)
def test_only_modules_that_support_a_cache_follow_cache_mode(cache_mode, support_cache, expected) -> None:
    args = OmniTrainingArguments(cache_mode=cache_mode, offline_cache_dir="/tmp/cache")

    assert args.module_cache_mode(support_cache) == expected


def test_data_args_accepts_cached_seedomni() -> None:
    args = OmniDataArguments(data_type="seedomni_cached", train_path="/tmp/cache")

    assert args.data_type == "seedomni_cached"
