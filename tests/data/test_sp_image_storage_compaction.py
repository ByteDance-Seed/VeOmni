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

"""SP image slices should not retain the storage for all image patches."""

import copy
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data import data_collator as collators


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _parallel_state(size, rank):
    """Build the minimal SP state needed by the collator fixtures."""
    return SimpleNamespace(sp_enabled=size > 1, sp_size=size, sp_rank=rank)


@pytest.mark.parametrize("key", ("pixel_values", "pixel_values_videos"))
@pytest.mark.parametrize("rank", (0, 3, 7))
def test_image_slice_compacts_cpu_storage_without_changing_values(monkeypatch, key, rank):
    """Ordinary image and video views retain their values in smaller storage."""
    monkeypatch.setattr(collators, "get_parallel_state", lambda: _parallel_state(8, rank))
    sp_collator = collators.SequenceParallelCollator()
    source = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16)
    actual = sp_collator.sp_slice(key, source, dim=0)
    expected = source.narrow(0, rank * 8, 8)

    assert torch.equal(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()
    assert actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


def test_gradient_carrying_image_slice_preserves_view_and_gradient(monkeypatch):
    """Gradient-carrying image tensors keep view and autograd behavior."""
    monkeypatch.setattr(collators, "get_parallel_state", lambda: _parallel_state(8, 3))
    sp_collator = collators.SequenceParallelCollator()
    source = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16).requires_grad_()
    actual = sp_collator.sp_slice("pixel_values", source, dim=0)
    assert actual.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    actual.sum().backward()
    expected_grad = torch.zeros_like(source)
    expected_grad.narrow(0, 3 * 8, 8).fill_(1)
    assert torch.equal(source.grad, expected_grad)


def test_only_oversized_image_views_are_copied(monkeypatch):
    """Avoid copies for non-image or already compact slices."""
    monkeypatch.setattr(collators, "get_parallel_state", lambda: _parallel_state(1, 0))
    sp_collator = collators.SequenceParallelCollator()
    source = torch.randn(8, 16)
    unchanged = sp_collator.sp_slice("pixel_values", source, dim=0)
    assert unchanged.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()

    monkeypatch.setattr(collators, "get_parallel_state", lambda: _parallel_state(4, 1))
    sp_collator = collators.SequenceParallelCollator()
    text = sp_collator.sp_slice("input_ids", source, dim=0)
    assert text.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    # A zero-length shard should not retain a nonempty full-image backing store.
    empty = sp_collator.sp_slice("pixel_values", torch.randn(3, 16), dim=0)
    assert empty.numel() == 0 and empty.untyped_storage().nbytes() == 0


def test_non_grad_image_base_is_released_after_slicing(monkeypatch):
    """Compaction permits the original image backing allocation to be released."""
    monkeypatch.setattr(collators, "get_parallel_state", lambda: _parallel_state(8, 2))
    sp_collator = collators.SequenceParallelCollator()
    base = torch.arange(64 * 16, dtype=torch.float32)
    base_ref = weakref.ref(base)
    source = base.view(64, 16)
    shard = sp_collator.sp_slice("pixel_values", source, dim=0)
    del source, base
    gc.collect()
    assert base_ref() is None
    assert torch.equal(shard, torch.arange(16 * 16, 24 * 16, dtype=torch.float32).view(8, 16))


def test_pinned_image_slices_preserve_original_view(monkeypatch):
    """Exercise the pinned-memory guard without an accelerator allocator."""
    monkeypatch.setattr(collators, "get_parallel_state", lambda: _parallel_state(2, 1))
    sp_collator = collators.SequenceParallelCollator()
    source = torch.randn(8, 4)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "is_pinned", lambda tensor: True)
        pinned_shard = sp_collator.sp_slice("pixel_values", source, dim=0)
    assert pinned_shard.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()


@pytest.mark.parametrize("key", ("pixel_values", "pixel_values_videos"))
@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
def test_real_pinned_image_slices_preserve_original_view(monkeypatch, key):
    """Retain the original view for real accelerator-pinned image storage."""
    if not torch.accelerator.is_available():
        pytest.skip("requires an accelerator-backed pinned-memory allocator")
    monkeypatch.setattr(collators, "get_parallel_state", lambda: _parallel_state(2, 1))
    source = torch.arange(32, dtype=torch.float32).reshape(8, 4).pin_memory()
    assert source.is_pinned()
    shard = collators.SequenceParallelCollator().sp_slice(key, source, dim=0)
    assert shard.is_pinned()
    assert torch.equal(shard, source.narrow(0, 4, 4))
    assert shard.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()


def test_mixed_packing_metadata_and_weights_match_view_baseline(monkeypatch):
    """Compare packed values, boundaries, weights and metadata to view slicing."""
    monkeypatch.setattr(collators, "get_parallel_state", lambda: _parallel_state(4, 2))

    class OriginalSequenceParallelCollator(collators.SequenceParallelCollator):
        def sp_slice(self, key, feature, dim=-1):
            """Reproduce the previous un-compacted slice for comparison."""
            if isinstance(feature, list):
                chunk = len(feature) // self.sp_size
                return feature[self.sp_rank * chunk : (self.sp_rank + 1) * chunk]
            chunk = feature.size(dim) // self.sp_size
            return feature.narrow(dim, self.sp_rank * chunk, chunk)

    def metadata(batch, padding):
        """Record the image padding reported to a multimodal metadata hook."""
        batch["multimodal_metadata"] = {"image_padding": padding["pixel_values"]}

    features = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4]),
            "loss_weights": torch.tensor([0.0, 0.25, 0.5, 0.75]),
            "pixel_values": torch.arange(5 * 8, dtype=torch.float32).reshape(5, 8),
            "image_grid_thw": torch.tensor([[1, 1, 5]]),
        },
        {
            "input_ids": torch.tensor([5, 6, 7]),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "labels": torch.tensor([5, 6, 7]),
            "loss_weights": torch.tensor([0.0, 0.5, 1.0]),
        },
    ]
    info = {"loss_weights": collators.DataCollateInfo(-1, True, 0, 1)}
    current = collators.MainCollator(data_collate_info=info, metadata_collate_func=metadata)
    reference = collators.MainCollator(data_collate_info=info, metadata_collate_func=metadata)
    reference.preforward_pipeline[-1] = OriginalSequenceParallelCollator(
        collate_infos=reference.collate_infos, metadata_collate_func=metadata
    )
    actual = current(copy.deepcopy(features))
    expected = reference(copy.deepcopy(features))

    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, expected[name]), name
        else:
            assert value == expected[name], name
    assert actual["pixel_values"].untyped_storage().nbytes() < expected["pixel_values"].untyped_storage().nbytes()
