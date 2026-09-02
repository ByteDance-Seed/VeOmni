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

import random

import pytest

import veomni.data.dataset as dataset_module
from veomni.data.dataset import InterleavedMappingDataset, MappingDataset, build_dataset


def _read_cycle(dataset, cycle):
    start = cycle * len(dataset)
    return [dataset[index] for index in range(start, start + len(dataset))]


def test_mapping_dataset_overlength_indices_are_deterministic():
    first = MappingDataset(list(range(8)), seed=17)
    second = MappingDataset(list(range(8)), seed=17)

    expected = _read_cycle(first, 1)
    assert expected != list(range(8))
    assert sorted(expected) == list(range(8))
    assert _read_cycle(second, 1) == expected
    assert [second[index] for index in (12, 8, 15, 9)] == [expected[index] for index in (4, 0, 7, 1)]


def test_mapping_dataset_preserves_python_negative_index_semantics():
    dataset = MappingDataset(list(range(8)), seed=17)
    assert dataset[-1] == 7
    with pytest.raises(IndexError, match="out of range"):
        dataset[-9]


def test_mapping_dataset_cycle_is_reproducible_after_restart():
    uninterrupted = MappingDataset(list(range(8)), seed=9)
    expected = _read_cycle(uninterrupted, 2)

    resumed = MappingDataset(list(range(8)), seed=9)
    assert [resumed[index] for index in range(19, 24)] == expected[3:]


def test_mapping_dataset_mapping_is_independent_of_cycle_access_order():
    dataset = MappingDataset(list(range(8)), seed=9)
    expected = {index: dataset[index] for index in range(8, 32)}

    for index in (25, 9, 17, 31, 8, 24, 16):
        assert dataset[index] == expected[index]


def test_empty_mapping_dataset_raises_index_error():
    dataset = MappingDataset([], seed=9)
    with pytest.raises(IndexError, match="empty dataset"):
        dataset[0]
    with pytest.raises(IndexError, match="empty dataset"):
        dataset[-1]


def test_mapping_dataset_seed_controls_overlength_order_without_global_rng_side_effect():
    first = MappingDataset(list(range(8)), seed=3)
    second = MappingDataset(list(range(8)), seed=4)
    assert _read_cycle(first, 1) != _read_cycle(second, 1)

    random.seed(123)
    expected = random.random()
    random.seed(123)
    first[len(first)]
    assert random.random() == expected


def test_build_mapping_dataset_forwards_seed(monkeypatch):
    monkeypatch.setattr(dataset_module, "get_data_files", lambda _: (["unused.jsonl"], "json"))
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: list(range(8)))

    first = build_dataset(dataset_name="mapping", train_path="unused.jsonl", seed=3)
    second = build_dataset(dataset_name="mapping", train_path="unused.jsonl", seed=4)

    assert _read_cycle(first, 1) != _read_cycle(second, 1)


def test_overlength_read_does_not_change_first_cycle_order():
    dataset = MappingDataset(list(range(8)), seed=3)
    dataset[len(dataset)]
    assert _read_cycle(dataset, 0) == list(range(8))


def test_interleaved_mapping_dataset_uses_same_deterministic_mapping():
    data = [{"value": index, "ds_idx": index % 2} for index in range(8)]

    first = InterleavedMappingDataset(data, seed=7)
    second = InterleavedMappingDataset(data, seed=7)

    assert _read_cycle(first, 1) == _read_cycle(second, 1)
