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

"""Unit tests for veomni.data.evaluator."""

import pytest
import torch

from veomni.data.evaluator import (
    EVALUATOR_REGISTRY,
    AccuracyEvaluator,
    Evaluator,
    LossEvaluator,
    PerplexityEvaluator,
    TokenAccuracyEvaluator,
    build_evaluator,
    compute_metrics,
)


class TestPerplexityEvaluator:
    """Tests for PerplexityEvaluator."""

    def test_perfect_prediction(self):
        """When logits perfectly predict labels, perplexity should be ~1.0."""
        # Create logits where the correct next token gets overwhelming probability.
        # The evaluator applies a causal LM shift (logits[:-1] vs labels[1:]),
        # so position s in logits should predict labels[s+1].
        batch, seq, vocab = 2, 4, 10
        labels = torch.randint(0, vocab, (batch, seq))
        logits = torch.full((batch, seq, vocab), -10.0)
        for b in range(batch):
            for s in range(seq - 1):
                logits[b, s, labels[b, s + 1]] = 10.0

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        assert "perplexity" in result
        assert result["perplexity"] < 2.0  # Should be very close to 1.0

    def test_random_prediction(self):
        """Random logits should give perplexity close to vocab_size."""
        vocab = 100
        batch, seq = 4, 16
        labels = torch.randint(0, vocab, (batch, seq))
        logits = torch.randn(batch, seq, vocab)

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        # Random prediction perplexity ≈ vocab_size
        assert 20 < result["perplexity"] < 200

    def test_ignore_index(self):
        """Positions with label -100 should be excluded."""
        batch, seq, vocab = 1, 6, 10
        labels = torch.tensor([[1, 2, -100, -100, 5, 6]])
        logits = torch.randn(batch, seq, vocab)

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)

        # After causal LM shift: shift_labels = [2, -100, -100, 5, 6] (5 positions)
        # 2 positions are -100, so 3 valid tokens remain.
        # Count is summed across all batches; with batch=1 it equals 3.
        assert partial["perplexity_count"].item() > 0
        assert partial["perplexity_count"].item() <= batch * (seq - 1)

    def test_2d_logits(self):
        """Should handle 2D logits (batch, vocab) for classification."""
        batch, vocab = 4, 10
        labels = torch.randint(0, vocab, (batch,))
        logits = torch.randn(batch, vocab)

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        assert "perplexity" in result
        assert result["perplexity"] > 0


class TestAccuracyEvaluator:
    """Tests for AccuracyEvaluator."""

    def test_perfect_accuracy(self):
        """When argmax matches labels, accuracy should be 1.0."""
        batch, seq, vocab = 2, 8, 10
        labels = torch.randint(0, vocab, (batch, seq))
        logits = torch.full((batch, seq, vocab), -10.0)
        # Position s predicts labels[s+1] due to causal LM shift
        for b in range(batch):
            for s in range(seq - 1):
                logits[b, s, labels[b, s + 1]] = 10.0

        evaluator = AccuracyEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        assert result["accuracy"] == 1.0

    def test_zero_accuracy(self):
        """When argmax never matches, accuracy should be 0.0."""
        batch, seq, vocab = 2, 8, 10
        labels = torch.zeros(batch, seq, dtype=torch.long)
        # Set logits so argmax is always 1 (never 0)
        logits = torch.full((batch, seq, vocab), -10.0)
        logits[:, :, 1] = 10.0

        evaluator = AccuracyEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        assert result["accuracy"] == 0.0

    def test_ignore_index(self):
        """Ignored positions should not affect accuracy."""
        labels = torch.tensor([[1, 2, -100, -100, 5, 6]])
        logits = torch.randn(1, 6, 10)

        evaluator = AccuracyEvaluator()
        partial = evaluator.compute(logits, labels)

        # Count should exclude -100 positions
        assert partial["accuracy_count"].item() > 0


class TestTokenAccuracyEvaluator:
    """Tests for TokenAccuracyEvaluator."""

    def test_basic(self):
        batch, seq, vocab = 2, 4, 10
        labels = torch.randint(0, vocab, (batch, seq))
        logits = torch.randn(batch, seq, vocab)

        evaluator = TokenAccuracyEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        assert 0.0 <= result["token_accuracy"] <= 1.0


class TestLossEvaluator:
    """Tests for LossEvaluator."""

    def test_with_model_loss(self):
        """Should use model's native loss when provided."""
        batch, seq, vocab = 2, 4, 10
        labels = torch.randint(0, vocab, (batch, seq))
        logits = torch.randn(batch, seq, vocab)
        loss = torch.tensor(2.5)

        evaluator = LossEvaluator()
        partial = evaluator.compute(logits, labels, loss=loss)
        result = evaluator.aggregate(partial)

        assert "val_loss" in result
        assert abs(result["val_loss"] - 2.5) < 0.1

    def test_without_model_loss(self):
        """Should compute CE loss from logits when loss is None."""
        batch, seq, vocab = 2, 4, 10
        labels = torch.randint(0, vocab, (batch, seq))
        logits = torch.randn(batch, seq, vocab)

        evaluator = LossEvaluator()
        partial = evaluator.compute(logits, labels, loss=None)
        result = evaluator.aggregate(partial)

        assert "val_loss" in result
        assert result["val_loss"] > 0


class TestRegistry:
    """Tests for the evaluator registry."""

    def test_build_perplexity(self):
        evaluator = build_evaluator("perplexity")
        assert isinstance(evaluator, PerplexityEvaluator)

    def test_build_accuracy(self):
        evaluator = build_evaluator("accuracy")
        assert isinstance(evaluator, AccuracyEvaluator)

    def test_build_token_accuracy(self):
        evaluator = build_evaluator("token_accuracy")
        assert isinstance(evaluator, TokenAccuracyEvaluator)

    def test_build_loss(self):
        evaluator = build_evaluator("loss")
        assert isinstance(evaluator, LossEvaluator)

    def test_build_unknown(self):
        with pytest.raises(ValueError, match="Unknown evaluator name"):
            build_evaluator("nonexistent_metric")

    def test_registry_keys(self):
        keys = EVALUATOR_REGISTRY.valid_keys()
        assert "perplexity" in keys
        assert "accuracy" in keys
        assert "token_accuracy" in keys
        assert "loss" in keys


class TestComputeMetrics:
    """Tests for the compute_metrics utility."""

    def test_multiple_evaluators(self):
        batch, seq, vocab = 2, 8, 10
        labels = torch.randint(0, vocab, (batch, seq))
        logits = torch.randn(batch, seq, vocab)

        evaluators = [
            PerplexityEvaluator(),
            AccuracyEvaluator(),
            LossEvaluator(),
        ]
        results = compute_metrics(evaluators, logits, labels, loss=None)

        assert "perplexity" in results
        assert "accuracy" in results
        assert "val_loss" in results

    def test_empty_list(self):
        """Empty evaluator list returns empty dict."""
        results = compute_metrics([], torch.randn(2, 4, 10), torch.randint(0, 10, (2, 4)))
        assert results == {}


class TestPackedBatch:
    """Tests for packed-sample boundaries with IGNORE_INDEX=-100."""

    def test_packed_boundaries_excluded_from_perplexity(self):
        """Packed samples separated by -100 padding should not contribute to perplexity."""
        # Simulate a packed batch: two short sequences concatenated with -100 padding
        # Sequence 1: [1, 2, 3] followed by -100 padding
        # Sequence 2: [4, 5] followed by -100 padding
        # After causal LM shift, -100 positions must be excluded from NLL.
        labels = torch.tensor([[1, 2, 3, -100, -100, 4, 5, -100, -100, -100]])
        vocab = 10
        batch, seq = labels.shape
        logits = torch.randn(batch, seq, vocab)

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)

        # After shift: shift_labels = [2, 3, -100, -100, 4, 5, -100, -100, -100]
        # Valid positions: 2, 3, 4, 5 = 4 tokens (ignoring the -100s)
        assert partial["perplexity_count"].item() == 4

    def test_packed_boundaries_excluded_from_accuracy(self):
        """Packed sample boundaries with -100 should not affect accuracy."""
        labels = torch.tensor([[1, 2, -100, -100, 5, 6, -100, -100]])
        vocab = 10
        batch, seq = labels.shape
        logits = torch.randn(batch, seq, vocab)

        evaluator = AccuracyEvaluator()
        partial = evaluator.compute(logits, labels)

        # After shift: shift_labels = [2, -100, -100, 5, 6, -100, -100]
        # Valid positions: 2, 5, 6 = 3 tokens
        assert partial["accuracy_count"].item() == 3

    def test_all_ignored_returns_nan(self):
        """When all positions are -100, metrics should be NaN."""
        labels = torch.tensor([[-100, -100, -100, -100]])
        logits = torch.randn(1, 4, 10)

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        assert partial["perplexity_count"].item() == 0
        assert result["perplexity"] != result["perplexity"]  # NaN check
