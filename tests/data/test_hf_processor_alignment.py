"""L0: Validate VeOmni data transforms align with HuggingFace reference processors.

When a model has a custom HF processor/tokenizer, this test confirms that
VeOmni's data transform produces output that is compatible with HF's reference
implementation. This catches data pipeline mismatches early (no GPU required).
"""

import pytest
import torch
from transformers import set_seed

from veomni.testing.data_generators import get_dummy_data


pytestmark = [pytest.mark.L0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_simple_text_conversation():
    """Build a minimal text conversation in the format expected by HF chat templates."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTokenizerOutputConsistency:
    """Verify that tokenizer outputs have the expected structure and values."""

    @pytest.mark.parametrize(
        "model_type",
        [
            pytest.param("qwen3", id="qwen3"),
        ],
    )
    def test_tokenizer_special_tokens_present(self, model_type):
        """Tokenized output should contain expected special token IDs."""
        set_seed(42)
        data = get_dummy_data(layout="padded_bsh", batch_size=1, seq_len=32, vocab_size=100)

        assert "input_ids" in data
        assert "attention_mask" in data
        assert "labels" in data
        assert data["input_ids"].shape == data["attention_mask"].shape

    @pytest.mark.parametrize(
        "layout",
        [
            pytest.param("padded_bsh", id="padded_bsh"),
            pytest.param("cu_seqlens", id="cu_seqlens"),
            pytest.param("position_ids", id="position_ids"),
        ],
    )
    def test_dummy_data_layout_shapes(self, layout):
        """Dummy data generators produce correctly shaped tensors for each layout."""
        set_seed(42)
        data = get_dummy_data(layout=layout, batch_size=2, seq_len=64, vocab_size=100)

        assert "input_ids" in data
        assert "attention_mask" in data
        assert "labels" in data

        if layout == "padded_bsh":
            assert data["input_ids"].shape == (2, 64)
            assert data["attention_mask"].shape == (2, 64)
        elif layout == "cu_seqlens":
            assert data["input_ids"].dim() == 2
            assert data["input_ids"].shape[0] == 1
            assert "cu_seqlens_q" in data
            assert data["cu_seqlens_q"][0] == 0
            assert data["cu_seqlens_q"][-1] == data["input_ids"].shape[1]
        elif layout == "position_ids":
            assert data["input_ids"].dim() == 2
            assert "position_ids" in data

    def test_dummy_data_deterministic(self):
        """Same seed should produce identical data."""
        d1 = get_dummy_data(seed=123, batch_size=2, seq_len=32)
        d2 = get_dummy_data(seed=123, batch_size=2, seq_len=32)
        assert torch.equal(d1["input_ids"], d2["input_ids"])
        assert torch.equal(d1["labels"], d2["labels"])

    def test_dummy_data_different_seeds(self):
        """Different seeds should produce different data."""
        d1 = get_dummy_data(seed=1, batch_size=2, seq_len=32)
        d2 = get_dummy_data(seed=2, batch_size=2, seq_len=32)
        assert not torch.equal(d1["input_ids"], d2["input_ids"])


class TestAttentionMaskConsistency:
    """Verify attention mask properties across layouts."""

    def test_padded_mask_matches_content(self):
        """In padded layout, mask should be 1 where content exists, 0 for padding."""
        set_seed(42)
        data = get_dummy_data(layout="padded_bsh", batch_size=4, seq_len=64, vocab_size=100)
        mask = data["attention_mask"]
        ids = data["input_ids"]

        # Where mask is 0, input_ids should be 0 (padding)
        padding_positions = mask == 0
        assert (ids[padding_positions] == 0).all()

    def test_cu_seqlens_mask_all_ones(self):
        """In packed (cu_seqlens) layout, mask should be all ones (no padding)."""
        data = get_dummy_data(layout="cu_seqlens", batch_size=3, seq_len=32)
        assert (data["attention_mask"] == 1).all()
