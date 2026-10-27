"""DCP file buckets must bound copy-ahead without changing stored tensors."""

import queue
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.planner import WriteItemType

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.bounded_writer import BoundedFileSystemWriter, _buckets


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _item(elements=None, dtype=torch.float32):
    if elements is None:
        return SimpleNamespace(type=WriteItemType.BYTE_IO, tensor_data=None)
    return SimpleNamespace(
        type=WriteItemType.TENSOR,
        tensor_data=SimpleNamespace(
            size=(elements * 16,), chunk=SimpleNamespace(sizes=(elements,)), properties=SimpleNamespace(dtype=dtype)
        ),
    )


def test_bucket_limits_and_oversized_or_opaque_items():
    items = [_item(4), _item(4), _item(20), _item(), _item(4), _item(4), _item(4)]
    buckets = list(_buckets(items, 32))
    assert buckets == [items[0:2], items[2:3], items[3:6], items[6:7]]
    with pytest.raises(ValueError, match="positive"):
        list(_buckets(items, 0))


def test_bucketed_dcp_roundtrip_is_bitwise_exact(tmp_path):
    source = {f"parameter_{i}": torch.arange(16, dtype=torch.float32) + i / 100 for i in range(8)}
    source["bf16"] = torch.arange(16, dtype=torch.bfloat16)
    source["byte_payload"] = "opaque payload"
    writer = BoundedFileSystemWriter(tmp_path, max_bytes=128)
    dcp.save(source, storage_writer=writer, no_dist=True)
    assert len(list(tmp_path.glob("*.distcp"))) >= 5
    metadata_before = (tmp_path / ".metadata").read_bytes()
    with pytest.raises(CheckpointException, match="Checkpoint already exists"):
        dcp.save(source, storage_writer=BoundedFileSystemWriter(tmp_path, max_bytes=128), no_dist=True)
    assert (tmp_path / ".metadata").read_bytes() == metadata_before
    restored = {name: torch.empty_like(value) if torch.is_tensor(value) else "" for name, value in source.items()}
    dcp.load(restored, checkpoint_id=tmp_path, no_dist=True)
    for name, expected in source.items():
        if torch.is_tensor(expected):
            assert torch.equal(restored[name], expected)
        else:
            assert restored[name] == expected


def test_bucketed_dcp_propagates_write_failure(tmp_path, monkeypatch):
    writer = BoundedFileSystemWriter(tmp_path, max_bytes=16)

    def fail(*args, **kwargs):
        raise OSError("injected storage failure")

    monkeypatch.setattr(writer.fs, "create_stream", fail)
    with pytest.raises(CheckpointException, match="CheckpointException|injected"):
        dcp.save({"x": torch.ones(16)}, storage_writer=writer, no_dist=True)
    assert not (tmp_path / ".metadata").exists()

    class EmptyOnSecondTensor(DefaultSavePlanner):
        def __init__(self):
            super().__init__()
            self.seen = 0

        def resolve_data(self, item):
            self.seen += 1
            if self.seen == 2:
                raise queue.Empty("injected planner failure")
            return super().resolve_data(item)

    # Exercise failures both after a complete file and midway through one file.
    for bucket_bytes in (64, 128):
        partial = tmp_path / f"partial-{bucket_bytes}"
        with pytest.raises(CheckpointException, match="Incomplete DCP write"):
            dcp.save(
                {"x": torch.ones(16), "y": torch.ones(16)},
                storage_writer=BoundedFileSystemWriter(partial, max_bytes=bucket_bytes),
                planner=EmptyOnSecondTensor(),
                no_dist=True,
            )
        assert list(partial.glob("*.distcp"))
        assert not (partial / ".metadata").exists()


def test_bucketed_writer_releases_previous_temporary_storages(tmp_path):
    class TrackingPlanner(DefaultSavePlanner):
        def __init__(self):
            super().__init__()
            self.live = 0
            self.peak = 0

        def resolve_data(self, item):
            value = super().resolve_data(item)
            if not torch.is_tensor(value):
                return value
            value = value.clone()
            self.live += 1
            self.peak = max(self.peak, self.live)
            storage = value.untyped_storage()
            weakref.finalize(storage, self._release)
            # The Ascend image retains temporary storages through cycles in
            # the serialization path. Reproduce this even on other runtimes.
            cycle = {"storage": storage}
            cycle["self"] = cycle
            return value

        def _release(self):
            self.live -= 1

    planner = TrackingPlanner()
    source = {f"parameter_{i}": torch.ones(16, dtype=torch.float32) for i in range(20)}
    dcp.save(source, storage_writer=BoundedFileSystemWriter(tmp_path, max_bytes=128), planner=planner, no_dist=True)
    # Two 64-byte tensors fit in one file. Unreachable cycles must be collected
    # at file boundaries, and peak must not grow with the total number of files.
    assert planner.peak <= 2
    assert planner.peak < len(source)
