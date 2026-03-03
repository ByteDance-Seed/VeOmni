"""Tests for LiberoActionCollator and LiberoActionPackingCollator."""

import pytest
import torch

from veomni.data.multimodal.data_collator import LiberoActionCollator, LiberoActionPackingCollator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(seq_len, state_dim, pred_len, action_dim, img_tokens=4, seed=0):
    """Create a fake sample mimicking process_libero_sample_qwen3_vl output."""
    g = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(0, 100, (seq_len,), generator=g),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "position_ids": torch.stack(
            [
                torch.arange(seq_len),
                torch.arange(seq_len),
                torch.arange(seq_len),
            ]
        ),  # (3, seq_len)
        "image_mask": torch.cat(
            [
                torch.ones(img_tokens, dtype=torch.bool),
                torch.zeros(seq_len - img_tokens, dtype=torch.bool),
            ]
        ),
        "video_mask": torch.zeros(seq_len, dtype=torch.bool),
        "pixel_values": torch.randn(img_tokens, 3 * 14 * 14, generator=g),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "observation_state": torch.randn(state_dim, generator=g),
        "labels": torch.randn(pred_len, action_dim, generator=g),
    }


@pytest.fixture
def two_samples():
    """Two samples with different sequence lengths but same state/action dims."""
    s1 = _make_sample(seq_len=10, state_dim=8, pred_len=4, action_dim=7, img_tokens=4, seed=0)
    s2 = _make_sample(seq_len=14, state_dim=8, pred_len=4, action_dim=7, img_tokens=6, seed=1)
    return [s1, s2]


@pytest.fixture
def three_samples():
    """Three samples with varying sequence lengths."""
    return [_make_sample(seq_len=8, state_dim=8, pred_len=4, action_dim=7, img_tokens=2, seed=i) for i in range(3)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLiberoActionCollator:
    def test_output_keys(self, two_samples):
        collator = LiberoActionCollator()
        batch = collator(two_samples)
        expected_keys = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "image_mask",
            "video_mask",
            "pixel_values",
            "image_grid_thw",
            "observation_state",
            "labels",
        }
        assert expected_keys == set(batch.keys())

    def test_padding_features_shape(self, two_samples):
        """Sequence-dim fields should be padded to the longest sample."""
        collator = LiberoActionCollator()
        batch = collator(two_samples)
        max_len = 14  # max of 10, 14

        assert batch["input_ids"].shape == (2, max_len)
        assert batch["attention_mask"].shape == (2, max_len)
        assert batch["image_mask"].shape == (2, max_len)
        assert batch["video_mask"].shape == (2, max_len)
        # position_ids: (B, 3, max_len) after transpose handling
        assert batch["position_ids"].shape == (2, 3, max_len)

    def test_padding_values(self, two_samples):
        """Shorter sample should be zero-padded for input_ids and attention_mask."""
        collator = LiberoActionCollator()
        batch = collator(two_samples)
        # Sample 0 has seq_len=10, padded to 14 -> last 4 should be 0
        assert (batch["input_ids"][0, 10:] == 0).all()
        assert (batch["attention_mask"][0, 10:] == 0).all()
        assert (batch["image_mask"][0, 10:] == False).all()  # noqa: E712

    def test_concat_features(self, two_samples):
        """pixel_values and image_grid_thw should be concatenated along dim 0."""
        collator = LiberoActionCollator()
        batch = collator(two_samples)
        # s1: 4 img tokens, s2: 6 img tokens -> concat = 10
        assert batch["pixel_values"].shape[0] == 4 + 6
        # s1: (1,3), s2: (1,3) -> concat = (2,3)
        assert batch["image_grid_thw"].shape == (2, 3)

    def test_stack_features_shape(self, two_samples):
        """observation_state and labels should be stacked (uniform shape)."""
        collator = LiberoActionCollator()
        batch = collator(two_samples)
        assert batch["observation_state"].shape == (2, 8)
        assert batch["labels"].shape == (2, 4, 7)

    def test_stack_features_values(self, two_samples):
        """Stacked values should preserve original tensor content."""
        collator = LiberoActionCollator()
        batch = collator(two_samples)
        assert torch.equal(batch["observation_state"][0], two_samples[0]["observation_state"])
        assert torch.equal(batch["observation_state"][1], two_samples[1]["observation_state"])
        assert torch.equal(batch["labels"][0], two_samples[0]["labels"])
        assert torch.equal(batch["labels"][1], two_samples[1]["labels"])

    def test_three_samples(self, three_samples):
        """Batch of 3 should work correctly."""
        collator = LiberoActionCollator()
        batch = collator(three_samples)
        assert batch["input_ids"].shape[0] == 3
        assert batch["observation_state"].shape[0] == 3
        assert batch["labels"].shape[0] == 3

    def test_single_sample(self):
        """Batch of 1 should not crash."""
        s = _make_sample(seq_len=12, state_dim=8, pred_len=4, action_dim=7, seed=42)
        collator = LiberoActionCollator()
        batch = collator([s])
        assert batch["input_ids"].shape == (1, 12)
        assert batch["observation_state"].shape == (1, 8)
        assert batch["labels"].shape == (1, 4, 7)

    def test_uniform_seq_len_no_padding(self):
        """When all samples have the same seq_len, no padding should be added."""
        samples = [_make_sample(seq_len=10, state_dim=8, pred_len=4, action_dim=7, seed=i) for i in range(2)]
        collator = LiberoActionCollator()
        batch = collator(samples)
        assert batch["input_ids"].shape == (2, 10)
        # No extra zeros from padding
        assert (batch["attention_mask"] == 1).all()

    def test_position_ids_3d_transpose(self):
        """position_ids should be correctly transposed: input (3, L) -> output (B, 3, max_L)."""
        s1 = _make_sample(seq_len=6, state_dim=8, pred_len=4, action_dim=7, seed=0)
        s2 = _make_sample(seq_len=8, state_dim=8, pred_len=4, action_dim=7, seed=1)
        collator = LiberoActionCollator()
        batch = collator([s1, s2])
        # s1 position_ids: (3, 6), s2: (3, 8) -> padded to (B, 3, 8)
        assert batch["position_ids"].shape == (2, 3, 8)
        # First sample's values should be preserved in the non-padded region
        assert torch.equal(batch["position_ids"][0, :, :6], s1["position_ids"])

    def test_dtypes_preserved(self, two_samples):
        """Output dtypes should match expected types."""
        collator = LiberoActionCollator()
        batch = collator(two_samples)
        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["image_mask"].dtype == torch.bool
        assert batch["observation_state"].dtype == torch.float32
        assert batch["labels"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Packing collator tests
# ---------------------------------------------------------------------------


class TestLiberoActionPackingCollator:
    def test_output_keys(self, two_samples):
        collator = LiberoActionPackingCollator()
        batch = collator(two_samples)
        expected_keys = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "image_mask",
            "video_mask",
            "pixel_values",
            "image_grid_thw",
            "observation_state",
            "labels",
        }
        assert expected_keys == set(batch.keys())

    def test_packing_features_shape(self, two_samples):
        """Packed sequence features should have shape (1, packed_len)."""
        collator = LiberoActionPackingCollator()
        batch = collator(two_samples)
        packed_len = 10 + 14  # sum of seq_lens

        assert batch["input_ids"].shape == (1, packed_len)
        assert batch["attention_mask"].shape == (1, packed_len)
        assert batch["image_mask"].shape == (1, packed_len)
        assert batch["video_mask"].shape == (1, packed_len)
        # position_ids: (1, 3, packed_len) for 3D rope
        assert batch["position_ids"].shape == (1, 3, packed_len)

    def test_packed_length_equals_sum(self, three_samples):
        """Total packed length should equal sum of individual sample lengths."""
        collator = LiberoActionPackingCollator()
        batch = collator(three_samples)
        total_len = sum(s["input_ids"].shape[0] for s in three_samples)
        assert batch["input_ids"].shape == (1, total_len)

    def test_position_ids_concatenated(self, two_samples):
        """Each subsequence in position_ids should start at 0."""
        collator = LiberoActionPackingCollator()
        batch = collator(two_samples)
        pos = batch["position_ids"][0, 0, :]  # first rope dim
        # First subsequence starts at 0
        assert pos[0].item() == 0
        # Second subsequence starts at 0 (at offset 10)
        assert pos[10].item() == 0

    def test_concat_features(self, two_samples):
        """pixel_values and image_grid_thw should be concatenated along dim 0."""
        collator = LiberoActionPackingCollator()
        batch = collator(two_samples)
        assert batch["pixel_values"].shape[0] == 4 + 6
        assert batch["image_grid_thw"].shape == (2, 3)

    def test_stack_features_shape(self, two_samples):
        """observation_state and labels should be stacked (N, ...)."""
        collator = LiberoActionPackingCollator()
        batch = collator(two_samples)
        assert batch["observation_state"].shape == (2, 8)
        assert batch["labels"].shape == (2, 4, 7)

    def test_stack_features_values(self, two_samples):
        """Stacked values should preserve original tensor content."""
        collator = LiberoActionPackingCollator()
        batch = collator(two_samples)
        assert torch.equal(batch["observation_state"][0], two_samples[0]["observation_state"])
        assert torch.equal(batch["observation_state"][1], two_samples[1]["observation_state"])
        assert torch.equal(batch["labels"][0], two_samples[0]["labels"])
        assert torch.equal(batch["labels"][1], two_samples[1]["labels"])

    def test_single_sample(self):
        """Batch of 1 should produce (1, seq_len) packed output."""
        s = _make_sample(seq_len=12, state_dim=8, pred_len=4, action_dim=7, seed=42)
        collator = LiberoActionPackingCollator()
        batch = collator([s])
        assert batch["input_ids"].shape == (1, 12)
        assert batch["observation_state"].shape == (1, 8)
        assert batch["labels"].shape == (1, 4, 7)

    def test_packing_content_preserved(self, two_samples):
        """Packed input_ids should be the concatenation of individual samples."""
        collator = LiberoActionPackingCollator()
        batch = collator(two_samples)
        expected = torch.cat([two_samples[0]["input_ids"], two_samples[1]["input_ids"]], dim=0)
        assert torch.equal(batch["input_ids"][0], expected)

    def test_dtypes_preserved(self, two_samples):
        """Output dtypes should match expected types."""
        collator = LiberoActionPackingCollator()
        batch = collator(two_samples)
        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["image_mask"].dtype == torch.bool
        assert batch["observation_state"].dtype == torch.float32
        assert batch["labels"].dtype == torch.float32
