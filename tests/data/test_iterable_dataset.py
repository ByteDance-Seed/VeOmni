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

from itertools import islice

import pytest
from torch.utils.data import IterableDataset

from veomni.data.dataset import ShardedIterableDataset, get_data_files


class _ListStream(IterableDataset):
    def __init__(self, values):
        self.values = list(values)
        self.epoch = 0

    def __iter__(self):
        yield from self.values

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def test_get_data_files_skips_non_table_artifacts(tmp_path):
    nested = tmp_path / "rank0"
    nested.mkdir()
    parquet_path = nested / "shard.parquet"
    parquet_path.write_bytes(b"PAR1")
    (tmp_path / "veomni_cli.yaml").write_text("train: {}\n")
    (tmp_path / "first.png").write_bytes(b"\x89PNG")

    files, loader = get_data_files(str(tmp_path))
    assert loader == "parquet"
    assert files == [str(parquet_path)]


def test_get_data_files_rejects_mixed_table_types(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "b.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Mixed data file types"):
        get_data_files(str(tmp_path))


def test_get_data_files_empty_directory_errors(tmp_path):
    (tmp_path / "readme.md").write_text("not data")
    with pytest.raises(FileNotFoundError, match="No supported data files"):
        get_data_files(str(tmp_path))


def test_drop_last_equalizes_ranks():
    source = list(range(15))
    ranks = [
        list(ShardedIterableDataset(_ListStream(source), dp_rank=rank, dp_size=8, repeat=False)) for rank in range(8)
    ]
    assert [len(items) for items in ranks] == [1] * 8
    assert [items[0] for items in ranks] == list(range(8))


def test_repeat_replays_the_stream():
    stream = ShardedIterableDataset(_ListStream([0, 1]), repeat=True)
    assert list(islice(stream, 5)) == [0, 1, 0, 1, 0]


def test_repeat_drops_incomplete_round_before_replay():
    source = [0, 1, 2]
    ranks = [
        list(islice(ShardedIterableDataset(_ListStream(source), dp_rank=rank, dp_size=2, repeat=True), 4))
        for rank in range(2)
    ]
    assert ranks[0] == [0, 0, 0, 0]
    assert ranks[1] == [1, 1, 1, 1]


def test_repeat_exits_when_source_shorter_than_dp():
    ranks = [
        list(islice(ShardedIterableDataset(_ListStream([0]), dp_rank=rank, dp_size=2, repeat=True), 4))
        for rank in range(2)
    ]
    assert ranks == [[], []]


def test_repeat_advances_inner_epoch():
    inner = _ListStream(["a"])
    stream = ShardedIterableDataset(inner, repeat=True, seed=10)
    stream.set_epoch(3)
    assert list(islice(stream, 2)) == ["a", "a"]
    assert inner.epoch == 14
