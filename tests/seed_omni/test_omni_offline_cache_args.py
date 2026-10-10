from __future__ import annotations

import pytest

from veomni.arguments.omni_arguments_types import OmniDataArguments, OmniTrainingArguments


def test_omni_training_args_default_to_online_training() -> None:
    assert OmniTrainingArguments().training_task == "online_training"


def test_omni_training_args_requires_offline_cache_dir() -> None:
    with pytest.raises(ValueError, match="offline_cache_dir"):
        OmniTrainingArguments(training_task="offline_embedding")


def test_omni_training_args_accepts_offline_cache_dir() -> None:
    args = OmniTrainingArguments(training_task="offline_embedding", offline_cache_dir="/tmp/cache")

    assert args.training_task == "offline_embedding"
    assert args.offline_cache_dir == "/tmp/cache"


def test_omni_training_args_trains_from_a_cache_without_a_cache_dir() -> None:
    assert OmniTrainingArguments(training_task="offline_training").training_task == "offline_training"


@pytest.mark.parametrize("training_task", ["encode_only", "process_only", "other"])
def test_omni_training_args_rejects_unknown_training_task(training_task) -> None:
    with pytest.raises(ValueError, match=f"Unknown train.training_task '{training_task}'"):
        OmniTrainingArguments(training_task=training_task)


def test_data_args_accepts_cached_seedomni() -> None:
    args = OmniDataArguments(data_type="seedomni_cached", train_path="/tmp/cache")

    assert args.data_type == "seedomni_cached"
