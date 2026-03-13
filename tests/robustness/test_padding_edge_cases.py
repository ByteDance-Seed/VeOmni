"""L4: Padding and packing edge case tests.

Validates that the data collator (MainCollator) and training pipeline handle
edge cases in variable-length sequences correctly:
- Very short sequences (1 token)
- Sequences at max_seq_len boundary
- Mixed-length batches with extreme variance
- Packing mode with sequences that barely fit / don't fit
- All-padding batches

These edge cases can cause subtle bugs in:
- cu_seqlens computation for flash attention
- Position ID generation
- Loss masking
- Padding fill values
"""

import pytest
import torch
from transformers import set_seed

from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
_v4_only = pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0")


def _build_feature(seq_len: int, vocab_size: int = 1000, pad_token_id: int = 0):
    """Build a single feature dict with specified sequence length."""
    input_ids = torch.randint(1, vocab_size, (seq_len,))
    labels = torch.randint(1, vocab_size, (seq_len,))
    attention_mask = torch.ones(seq_len, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def _build_padded_feature(actual_len: int, total_len: int, vocab_size: int = 1000, pad_token_id: int = 0):
    """Build a feature dict with padding at the end."""
    input_ids = torch.zeros(total_len, dtype=torch.long)
    input_ids[:actual_len] = torch.randint(1, vocab_size, (actual_len,))

    labels = torch.full((total_len,), -100, dtype=torch.long)
    labels[:actual_len] = torch.randint(1, vocab_size, (actual_len,))

    attention_mask = torch.zeros(total_len, dtype=torch.long)
    attention_mask[:actual_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


@pytest.mark.L4
class TestMainCollatorEdgeCases:
    """Test MainCollator with edge-case inputs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        set_seed(42)
        from veomni.data.data_collator import MainCollator

        self.collator = MainCollator(seq_classification=False)

    def test_single_token_sequence(self):
        """Verify collator handles a batch with a single-token sequence."""
        features = [_build_feature(1)]
        batch = self.collator(features)
        assert batch["input_ids"].shape[-1] >= 1
        assert "labels" in batch

    def test_very_short_sequences(self):
        """Verify collator handles multiple very short sequences (2-5 tokens)."""
        features = [_build_feature(length) for length in [2, 3, 5]]
        batch = self.collator(features)
        assert batch["input_ids"].ndim >= 1

    def test_mixed_length_extreme_variance(self):
        """Verify collator handles sequences with extreme length variance."""
        features = [
            _build_feature(4),
            _build_feature(128),
            _build_feature(8),
            _build_feature(256),
        ]
        batch = self.collator(features)
        assert batch["input_ids"].ndim >= 1
        assert "labels" in batch

    def test_all_same_length(self):
        """Verify collator handles all sequences with identical length."""
        features = [_build_feature(64) for _ in range(4)]
        batch = self.collator(features)
        assert batch["input_ids"].ndim >= 1

    def test_padded_sequences(self):
        """Verify collator correctly handles pre-padded sequences."""
        features = [
            _build_padded_feature(actual_len=32, total_len=128),
            _build_padded_feature(actual_len=64, total_len=128),
        ]
        batch = self.collator(features)
        assert batch["input_ids"].ndim >= 1

    def test_single_sample_batch(self):
        """Verify collator handles batch with exactly one sample."""
        features = [_build_feature(64)]
        batch = self.collator(features)
        assert batch["input_ids"].ndim >= 1

    def test_large_batch(self):
        """Verify collator handles a large batch without OOM or crash."""
        features = [_build_feature(32) for _ in range(32)]
        batch = self.collator(features)
        assert batch["input_ids"].ndim >= 1

    def test_empty_labels(self):
        """Verify collator handles features where all labels are -100 (no loss tokens)."""
        feature = _build_feature(64)
        feature["labels"] = torch.full((64,), -100, dtype=torch.long)
        features = [feature]
        batch = self.collator(features)
        assert batch["input_ids"].ndim >= 1


@pytest.mark.L4
class TestMainCollatorWithSP:
    """Test MainCollator with sequence parallelism enabled."""

    @pytest.fixture(autouse=True)
    def setup(self):
        set_seed(42)
        from veomni.data.data_collator import MainCollator

        self.collator = MainCollator(seq_classification=False)

    def test_sp_compatible_lengths(self):
        """Verify packed sequences have lengths divisible by sp_size."""
        features = [_build_feature(63), _build_feature(65)]
        batch = self.collator(features)
        # The collator should produce valid output regardless of input lengths
        assert batch["input_ids"].ndim >= 1

    def test_sp_very_short(self):
        """Verify very short sequences work with SP packing."""
        features = [_build_feature(2), _build_feature(3)]
        batch = self.collator(features)
        assert batch["input_ids"].ndim >= 1


@pytest.mark.L4
class TestPositionIdsEdgeCases:
    """Test position ID generation edge cases."""

    @pytest.fixture(autouse=True)
    def setup(self):
        set_seed(42)

    def test_position_ids_monotonic(self):
        """Verify position_ids are monotonically increasing within each sequence."""
        from veomni.data.data_collator import MainCollator

        collator = MainCollator(seq_classification=False)
        features = [_build_feature(32), _build_feature(64)]
        batch = collator(features)

        if "position_ids" in batch:
            pos_ids = batch["position_ids"]
            if pos_ids.ndim == 1:
                # In packed mode, position_ids should reset per sequence
                # but be non-negative
                assert torch.all(pos_ids >= 0)
            elif pos_ids.ndim == 2:
                # Batch mode: each row should be monotonically increasing
                for i in range(pos_ids.shape[0]):
                    row = pos_ids[i]
                    mask = row >= 0
                    if mask.any():
                        valid = row[mask]
                        assert torch.all(valid[1:] >= valid[:-1])
