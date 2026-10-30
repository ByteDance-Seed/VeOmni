# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.trainer.base import BackgroundPrefetcher


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class MutableDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.history = []

    def __iter__(self):
        for item in range(len(self.history), 3):
            self.history.append(item)
            yield item

    def state_dict(self):
        return {"history": self.history}

    def load_state_dict(self, state):
        self.history = state["history"]


class DeferredState:
    def __init__(self, always_fail=False):
        self.cursor = 0
        self.always_fail = always_fail

    def __iter__(self):
        return self

    def __next__(self):
        self.cursor += 1
        return self.cursor

    def state_dict(self):
        if self.always_fail or self.cursor == 0:
            raise ValueError("cursor unavailable")
        return {"cursor": self.cursor}


class PickleOnlyState:
    def __init__(self, value):
        self.value = value

    def __deepcopy__(self, _memo):
        raise TypeError("use checkpoint serialization")


def test_checkpoint_before_first_consumption_is_rejected():
    dataset = MutableDataset()
    prefetcher = BackgroundPrefetcher(StatefulDataLoader(dataset, batch_size=None, num_workers=0))
    try:
        item, *_ = prefetcher.queue.get(timeout=5)
        assert dataset.history
        with pytest.raises(RuntimeError, match="recoverable data cursor"):
            prefetcher.state_dict()
        assert item == 0
    finally:
        prefetcher.stop()


def test_snapshot_after_consumption_cannot_be_mutated_by_prefetch():
    dataset = MutableDataset()
    prefetcher = BackgroundPrefetcher(StatefulDataLoader(dataset, batch_size=None, num_workers=0))
    try:
        assert next(prefetcher) == 0
        prefetcher.queue.get(timeout=5)
        prefetcher.queue.get(timeout=5)
        state = prefetcher.state_dict()
        assert dataset.history == [0, 1, 2]
        assert state["dataset_state"]["history"] == [0]
    finally:
        prefetcher.stop()
    resumed = StatefulDataLoader(MutableDataset(), batch_size=None, num_workers=0)
    resumed.load_state_dict(state)
    assert next(iter(resumed)) == 1


@pytest.mark.parametrize("context", [None, "spawn"])
def test_consumed_snapshot_restores_with_persistent_data_workers(context):
    def loader():
        return StatefulDataLoader(
            MutableDataset(), batch_size=None, num_workers=2, persistent_workers=True, multiprocessing_context=context
        )

    original = loader()
    prefetcher = BackgroundPrefetcher(original)
    restored = None
    try:
        next(prefetcher)
        state = prefetcher.state_dict()
        item = next(prefetcher)
        restored = loader()
        restored.load_state_dict(state)
        assert next(iter(restored)) == item
    finally:
        prefetcher.stop()
        for dataloader in (original, restored):
            iterator = getattr(dataloader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()


@pytest.mark.parametrize("always_fail", [False, True])
def test_unavailable_snapshot_only_prevents_checkpoint(always_fail):
    prefetcher = BackgroundPrefetcher(DeferredState(always_fail))
    try:
        with pytest.raises(RuntimeError, match="recoverable data cursor"):
            prefetcher.state_dict()
        assert next(prefetcher) == 1
        if always_fail:
            with pytest.raises(RuntimeError, match="recoverable data cursor"):
                prefetcher.state_dict()
        else:
            assert prefetcher.state_dict() == {"cursor": 1}
    finally:
        prefetcher.stop()


def test_unstateful_dataloader_still_iterates():
    prefetcher = BackgroundPrefetcher([1, 2, 3])
    try:
        assert prefetcher.state_dict() == {}
        assert list(prefetcher) == [1, 2, 3]
    finally:
        prefetcher.stop()


def test_nonleaf_tensor_state_remains_checkpointable():
    class NonLeafState(DeferredState):
        def state_dict(self):
            return {"value": torch.tensor(float(self.cursor), requires_grad=True) * 2}

    prefetcher = BackgroundPrefetcher(NonLeafState())
    try:
        assert next(prefetcher) == 1
        assert prefetcher.state_dict()["value"].item() == 2
    finally:
        prefetcher.stop()


def test_pickleable_state_without_deepcopy_remains_checkpointable():
    class PickleOnlyLoader(DeferredState):
        def state_dict(self):
            return {"value": PickleOnlyState(self.cursor)}

    prefetcher = BackgroundPrefetcher(PickleOnlyLoader())
    try:
        assert next(prefetcher) == 1
        assert prefetcher.state_dict()["value"].value == 1
    finally:
        prefetcher.stop()


def test_snapshot_copy_failure_does_not_return_aliased_state(monkeypatch):
    import veomni.trainer.base as base

    def fail_copy(_state):
        raise ValueError("copy failed")

    monkeypatch.setattr(base, "_copy_prefetch_state", fail_copy)
    prefetcher = BackgroundPrefetcher(DeferredState())
    try:
        assert next(prefetcher) == 1
        with pytest.raises(RuntimeError, match="recoverable data cursor"):
            prefetcher.state_dict()
    finally:
        prefetcher.stop()


def test_loader_state_is_not_queried_before_first_item():
    import threading

    class BlockingState(DeferredState):
        def __init__(self):
            super().__init__()
            self.ready = threading.Event()

        def __next__(self):
            item = super().__next__()
            self.ready.set()
            return item

        def state_dict(self):
            assert self.ready.wait(timeout=5)
            return {"cursor": self.cursor}

    prefetcher = BackgroundPrefetcher(BlockingState())
    try:
        assert next(prefetcher) == 1
        assert prefetcher.state_dict() == {"cursor": 1}
    finally:
        prefetcher.stop()
