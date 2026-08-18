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
        # Fixed labels for deterministic testing.
        # The evaluator applies a causal LM shift (logits[:-1] vs labels[1:]),
        # so position s in logits should predict labels[s+1].
        batch, seq, vocab = 2, 4, 10
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        logits = torch.full((batch, seq, vocab), -10.0)
        for b in range(batch):
            for s in range(seq - 1):
                logits[b, s, labels[b, s + 1]] = 10.0

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        assert "perplexity" in result
        assert result["perplexity"] == pytest.approx(1.0, rel=1e-2)

    def test_random_prediction(self):
        """Zero logits give uniform distribution, perplexity = vocab_size."""
        vocab = 100
        batch, seq = 4, 16
        labels = torch.arange(batch * seq).reshape(batch, seq) % vocab
        logits = torch.zeros(batch, seq, vocab)

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        assert result["perplexity"] == pytest.approx(float(vocab))

    def test_ignore_index(self):
        """Positions with label -100 should be excluded."""
        batch, seq, vocab = 1, 6, 10
        labels = torch.tensor([[1, 2, -100, -100, 5, 6]])
        logits = torch.zeros(batch, seq, vocab)

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        # After causal LM shift: shift_labels = [2, -100, -100, 5, 6] (5 positions)
        # 2 positions are -100, so 3 valid tokens remain.
        assert partial["perplexity_count"].item() == 3
        # Zero logits → uniform distribution → NLL = log(vocab) per token
        # Perplexity = exp(log(vocab)) = vocab
        assert result["perplexity"] == pytest.approx(float(vocab))

    def test_2d_logits(self):
        """Should handle 2D logits (batch, vocab) for classification."""
        batch, vocab = 4, 10
        labels = torch.tensor([1, 2, 3, 4])
        logits = torch.zeros(batch, vocab)

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        # Zero logits → uniform → perplexity = vocab
        assert partial["perplexity_count"].item() == batch
        assert result["perplexity"] == pytest.approx(float(vocab))


class TestAccuracyEvaluator:
    """Tests for AccuracyEvaluator."""

    def test_perfect_accuracy(self):
        """When argmax matches labels, accuracy should be 1.0."""
        batch, seq, vocab = 2, 8, 10
        labels = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])
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
        logits = torch.zeros(1, 6, 10)

        evaluator = AccuracyEvaluator()
        partial = evaluator.compute(logits, labels)

        # After shift: shift_labels = [2, -100, -100, 5, 6]
        # 3 valid positions; zero logits → argmax=0, none match labels
        assert partial["accuracy_count"].item() == 3

        result = evaluator.aggregate(partial)
        assert result["accuracy"] == 0.0


class TestTokenAccuracyEvaluator:
    """Tests for TokenAccuracyEvaluator."""

    def test_basic(self):
        batch, seq, vocab = 2, 4, 10
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        logits = torch.zeros(batch, seq, vocab)

        evaluator = TokenAccuracyEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        # Zero logits → argmax=0, none match labels (all > 0)
        # After shift: 3 valid positions per sample, 6 total
        assert partial["token_accuracy_count"].item() == 6
        assert result["token_accuracy"] == 0.0


class TestLossEvaluator:
    """Tests for LossEvaluator."""

    def test_with_model_loss(self):
        """Should use model's native loss when provided."""
        batch, seq, vocab = 2, 4, 10
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        logits = torch.zeros(batch, seq, vocab)
        loss = torch.tensor(2.5)

        evaluator = LossEvaluator()
        partial = evaluator.compute(logits, labels, loss=loss)
        result = evaluator.aggregate(partial)

        # After shift: 3 valid positions per sample, 6 total
        # LossEvaluator multiplies loss by token_count, aggregate divides by count
        assert partial["val_loss_count"].item() == 6
        assert result["val_loss"] == pytest.approx(2.5, rel=1e-4)

    def test_without_model_loss(self):
        """Should compute CE loss from logits when loss is None."""
        batch, seq, vocab = 2, 4, 10
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        logits = torch.zeros(batch, seq, vocab)

        evaluator = LossEvaluator()
        partial = evaluator.compute(logits, labels, loss=None)
        result = evaluator.aggregate(partial)

        # Zero logits → uniform distribution → CE loss = log(vocab) per token
        # After shift: 3 valid positions per sample, 6 total
        import math
        expected_loss = math.log(vocab)
        assert partial["val_loss_count"].item() == 6
        assert result["val_loss"] == pytest.approx(expected_loss, rel=1e-4)


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
        labels = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])
        logits = torch.zeros(batch, seq, vocab)

        evaluators = [
            PerplexityEvaluator(),
            AccuracyEvaluator(),
            LossEvaluator(),
        ]
        results = compute_metrics(evaluators, logits, labels, loss=None)

        assert "perplexity" in results
        assert "accuracy" in results
        assert "val_loss" in results
        # Zero logits → perplexity = vocab, accuracy = 0
        assert results["perplexity"] == pytest.approx(float(vocab))
        assert results["accuracy"] == 0.0

    def test_empty_list(self):
        """Empty evaluator list returns empty dict."""
        results = compute_metrics([], torch.zeros(2, 4, 10), torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
        assert results == {}


class TestPackedBatch:
    """Tests for packed-sample boundaries with IGNORE_INDEX=-100."""

    def test_packed_boundaries_excluded_from_perplexity(self):
        """Packed samples separated by -100 padding should not contribute to perplexity.

        In a packed batch, the first target after each boundary is also masked
        because the model's context at that position includes padding tokens.
        """
        # Two packed samples: [1, 2, 3] and [4, 5]
        # The first token of each new sample (4) is masked with -100 because
        # the model cannot predict it from the preceding padding context.
        labels = torch.tensor([[1, 2, 3, -100, -100, -100, 5, -100, -100, -100]])
        vocab = 10
        batch, seq = labels.shape
        logits = torch.zeros(batch, seq, vocab)

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)

        # After shift: shift_labels = [2, 3, -100, -100, -100, 5, -100, -100, -100]
        # Valid positions: 2, 3, 5 = 3 tokens
        assert partial["perplexity_count"].item() == 3

    def test_packed_boundaries_excluded_from_accuracy(self):
        """Packed sample boundaries with -100 should not affect accuracy.

        The first token after each boundary is masked because the model's
        context includes padding, making the prediction meaningless.
        """
        labels = torch.tensor([[1, 2, -100, -100, -100, 6, -100, -100]])
        vocab = 10
        batch, seq = labels.shape
        logits = torch.zeros(batch, seq, vocab)

        evaluator = AccuracyEvaluator()
        partial = evaluator.compute(logits, labels)

        # After shift: shift_labels = [2, -100, -100, -100, 6, -100, -100]
        # Valid positions: 2, 6 = 2 tokens
        assert partial["accuracy_count"].item() == 2

    def test_all_ignored_returns_nan(self):
        """When all positions are -100, metrics should be NaN."""
        labels = torch.tensor([[-100, -100, -100, -100]])
        logits = torch.zeros(1, 4, 10)

        evaluator = PerplexityEvaluator()
        partial = evaluator.compute(logits, labels)
        result = evaluator.aggregate(partial)

        assert partial["perplexity_count"].item() == 0
        assert result["perplexity"] != result["perplexity"]  # NaN check


class TestDistributedAggregation:
    """Integration tests for multi-rank metric aggregation.

    These tests initialize a real process group with the gloo backend and
    invoke the actual ``dist.all_reduce`` — no mocks.
    """

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @pytest.mark.skipif(
        not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
        reason="torch.distributed gloo backend not available",
    )
    def test_two_rank_perplexity_aggregation(self, tmp_path):
        """Verify token-weighted perplexity across two ranks via real all_reduce.

        Uses subprocess spawning with the gloo CPU backend so it runs anywhere
        without GPU dependencies. Rank 0 has perfectly predicted tokens (NLL≈0)
        while rank 1 has uniform logits (NLL=log(vocab)), so the aggregated
        perplexity = exp((0*2 + log(10)*1) / 3) = 10^(1/3).
        """
        import os
        import socket
        import subprocess
        import sys
        import textwrap

        # Find a free port for the rendezvous
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        master_port = str(sock.getsockname()[1])
        sock.close()

        script = tmp_path / "_two_rank_test.py"
        script.write_text(textwrap.dedent("""\
            import math
            import os
            import torch
            import torch.distributed as dist
            from veomni.data.evaluator import PerplexityEvaluator

            def main():
                rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                dist.init_process_group("gloo", rank=rank, world_size=world_size)

                vocab = 10
                if rank == 0:
                    # Perfect prediction: NLL ≈ 0 for 2 shifted tokens
                    labels = torch.tensor([[1, 2, 3]])
                    logits = torch.full((1, 3, vocab), -10.0)
                    logits[0, 0, 2] = 10.0  # predicts labels[1]=2
                    logits[0, 1, 3] = 10.0  # predicts labels[2]=3
                else:
                    # Uniform prediction: NLL = log(vocab) for 1 shifted token
                    labels = torch.tensor([[4, 5]])
                    logits = torch.zeros(1, 2, vocab)

                evaluator = PerplexityEvaluator()
                partial = evaluator.compute(logits, labels)
                result = evaluator.aggregate(partial)

                # Rank 0: 2 tokens, NLL ≈ 0
                # Rank 1: 1 token, NLL = log(10)
                # Total: 3 tokens, NLL = log(10)
                # Perplexity = exp(log(10) / 3) = 10^(1/3) ≈ 2.154
                expected = 10.0 ** (1.0 / 3.0)
                actual = result["perplexity"]
                assert abs(actual - expected) < 0.01, \\
                    f"Rank {rank}: expected {expected}, got {actual}"
                if rank == 0:
                    print("DISTRIBUTED_TEST_PASSED")

                dist.destroy_process_group()

            if __name__ == "__main__":
                main()
        """))

        env = os.environ.copy()
        env["WORLD_SIZE"] = "2"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = master_port
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        # Run two processes
        procs = []
        for r in range(2):
            env_rank = env.copy()
            env_rank["RANK"] = str(r)
            p = subprocess.Popen(
                [sys.executable, str(script)],
                env=env_rank,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            procs.append(p)

        outputs = []
        for p in procs:
            out, err = p.communicate(timeout=30)
            outputs.append(out.decode())
            if p.returncode != 0:
                pytest.fail(f"Rank process failed: {err.decode()}")

        assert "DISTRIBUTED_TEST_PASSED" in outputs[0], f"Rank 0 did not pass: {outputs[0]}"
