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

from types import SimpleNamespace

import pytest

from veomni.arguments.omni_arguments_types import (
    OmniDataArguments,
    OmniTrainingArguments,
    _validate_training_task_data,
)


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


def test_an_offline_embedding_run_reads_each_sample_once() -> None:
    with pytest.raises(ValueError, match="num_train_epochs` must be 1"):
        OmniTrainingArguments(training_task="offline_embedding", offline_cache_dir="/tmp/cache", num_train_epochs=2)


@pytest.mark.parametrize(
    ("training_task", "data_type", "ok"),
    [
        ("online_training", "seedomni", True),
        ("offline_embedding", "seedomni", True),
        ("offline_training", "seedomni_cached", True),
        ("offline_training", "seedomni", False),
        ("online_training", "seedomni_cached", False),
        ("offline_embedding", "seedomni_cached", False),
    ],
)
def test_only_offline_training_reads_the_cache(training_task, data_type, ok) -> None:
    train = SimpleNamespace(training_task=training_task)
    data = SimpleNamespace(data_type=data_type)

    if ok:
        _validate_training_task_data(train, data)
    else:
        with pytest.raises(ValueError, match="needs `data.data_type="):
            _validate_training_task_data(train, data)
