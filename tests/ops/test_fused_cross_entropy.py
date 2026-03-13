"""L1: Validate fused cross-entropy kernel against reference PyTorch implementation.

The ``ForCausalLMLoss`` patch is applied universally to all models. This test
ensures the fused implementation produces results numerically close to
PyTorch's native ``F.cross_entropy``.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import set_seed
from transformers.loss.loss_utils import fixed_cross_entropy

from veomni.ops.fused_cross_entropy.eager import eager_cross_entropy


pytestmark = [pytest.mark.L1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[128, 512, 2048], ids=["vocab128", "vocab512", "vocab2048"])
def vocab_size(request):
    return request.param


@pytest.fixture(params=[32, 256], ids=["seq32", "seq256"])
def seq_len(request):
    return request.param


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _reference_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: int | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Pure PyTorch cross-entropy as the ground-truth reference."""
    logits_flat = logits.view(-1, vocab_size).float()
    labels_flat = labels.view(-1)
    return fixed_cross_entropy(logits_flat, labels_flat, num_items_in_batch, ignore_index)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEagerCrossEntropyVsReference:
    """Compare eager_cross_entropy against PyTorch reference."""

    def test_loss_matches_reference(self, vocab_size, seq_len):
        """Eager fused CE loss should match PyTorch F.cross_entropy."""
        set_seed(42)
        batch_size = 2
        total_len = batch_size * seq_len

        logits = torch.randn(total_len, vocab_size, dtype=torch.float32)
        labels = torch.randint(0, vocab_size, (total_len,), dtype=torch.long)

        # Reference
        ref_loss = _reference_cross_entropy(logits, labels, vocab_size)

        # Eager implementation
        eager_loss, eager_logits = eager_cross_entropy(
            logits=logits.clone(),
            labels=labels.clone(),
            vocab_size=vocab_size,
        )

        torch.testing.assert_close(eager_loss, ref_loss, rtol=1e-4, atol=1e-4)

    def test_loss_with_ignore_index(self, vocab_size):
        """Loss should correctly ignore positions with ignore_index=-100."""
        set_seed(42)
        seq_len = 64
        logits = torch.randn(seq_len, vocab_size, dtype=torch.float32)
        labels = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)
        # Mask out half the positions
        labels[: seq_len // 2] = -100

        ref_loss = _reference_cross_entropy(logits, labels, vocab_size)
        eager_loss, _ = eager_cross_entropy(
            logits=logits.clone(),
            labels=labels.clone(),
            vocab_size=vocab_size,
        )

        torch.testing.assert_close(eager_loss, ref_loss, rtol=1e-4, atol=1e-4)

    def test_loss_all_ignored(self):
        """If all labels are ignore_index, loss should be 0."""
        set_seed(42)
        vocab_size = 128
        seq_len = 32
        logits = torch.randn(seq_len, vocab_size, dtype=torch.float32)
        labels = torch.full((seq_len,), -100, dtype=torch.long)

        ref_loss = _reference_cross_entropy(logits, labels, vocab_size)
        eager_loss, _ = eager_cross_entropy(
            logits=logits.clone(),
            labels=labels.clone(),
            vocab_size=vocab_size,
        )

        torch.testing.assert_close(eager_loss, ref_loss, rtol=1e-5, atol=1e-5)

    def test_gradient_matches_reference(self):
        """Gradients through eager CE should match reference."""
        set_seed(42)
        vocab_size = 128
        seq_len = 64

        logits_ref = torch.randn(seq_len, vocab_size, dtype=torch.float32, requires_grad=True)
        logits_eager = logits_ref.detach().clone().requires_grad_(True)
        labels = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)

        # Reference backward
        ref_loss = _reference_cross_entropy(logits_ref, labels, vocab_size)
        ref_loss.backward()

        # Eager backward
        eager_loss, _ = eager_cross_entropy(
            logits=logits_eager,
            labels=labels.clone(),
            vocab_size=vocab_size,
        )
        eager_loss.backward()

        torch.testing.assert_close(
            logits_eager.grad,
            logits_ref.grad,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_hidden_states_path(self):
        """When hidden_states + weights are provided, eager CE uses linear projection."""
        set_seed(42)
        hidden_dim = 64
        vocab_size = 128
        seq_len = 32

        hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.float32)
        weights = torch.randn(vocab_size, hidden_dim, dtype=torch.float32)
        labels = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)

        # Manually compute logits for reference
        logits_manual = F.linear(hidden_states, weights).float()
        ref_loss = _reference_cross_entropy(logits_manual, labels, vocab_size)

        # Eager with hidden_states path
        eager_loss, _ = eager_cross_entropy(
            logits=None,
            labels=labels,
            vocab_size=vocab_size,
            hidden_states=hidden_states,
            weights=weights,
        )

        torch.testing.assert_close(eager_loss, ref_loss, rtol=1e-4, atol=1e-4)


class TestCrossEntropyEdgeCases:
    """Edge cases and numerical stability."""

    def test_single_token(self):
        """Single-token input should produce valid loss."""
        vocab_size = 100
        logits = torch.randn(1, vocab_size, dtype=torch.float32)
        labels = torch.tensor([42], dtype=torch.long)

        loss, _ = eager_cross_entropy(logits=logits, labels=labels, vocab_size=vocab_size)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_large_logits_numerical_stability(self):
        """Very large logit values should not produce NaN/Inf loss."""
        vocab_size = 100
        logits = torch.randn(32, vocab_size, dtype=torch.float32) * 100
        labels = torch.randint(0, vocab_size, (32,), dtype=torch.long)

        loss, _ = eager_cross_entropy(logits=logits, labels=labels, vocab_size=vocab_size)
        assert torch.isfinite(loss)
