from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import veomni.ops.kernels.cross_entropy.chunk_loss as chunk_loss_module
from veomni.utils.device import IS_NPU_AVAILABLE


def test_chunk_loss_reuses_valid_token_denominator(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    original_sum = torch.Tensor.sum
    denominator_sum_calls = 0

    def counting_sum(self, *args, **kwargs):
        nonlocal denominator_sum_calls
        if self.dtype == torch.bool and self.shape == (1, 5):
            denominator_sum_calls += 1
        return original_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", counting_sum)

    hidden_states = torch.randn(1, 6, 4, requires_grad=True)
    weights = torch.randn(8, 4, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, 3, 4, 5]])

    loss, _ = chunk_loss_module.chunk_loss_function(
        hidden_states,
        weights,
        labels,
        chunk_size=2,
    )
    loss.backward()

    assert denominator_sum_calls == 1
    assert hidden_states.grad is not None
    assert weights.grad is not None


def test_chunk_loss_honors_explicit_shift_labels(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    hidden_states = torch.randn(1, 3, 4)
    weights = torch.randn(8, 4)
    labels = torch.tensor([[0, 1, 2]])
    shift_labels = torch.tensor([[7, -100, 3]])

    actual, _ = chunk_loss_module.chunk_loss_function(
        hidden_states,
        weights,
        labels,
        shift_labels=shift_labels,
        chunk_size=2,
    )
    expected = F.cross_entropy(
        F.linear(hidden_states, weights).float().reshape(-1, weights.size(0)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )

    torch.testing.assert_close(actual, expected)


def test_dispatch_forwards_chunk_size(monkeypatch):
    import veomni.ops.kernels.cross_entropy as dispatch

    seen = []

    def record(*args, **kwargs):
        seen.append(kwargs["chunk_size"])
        return torch.tensor(0.0), None

    monkeypatch.setattr(dispatch, "chunk_loss_function", record)
    for size in (1024, 512, 256):
        dispatch._chunk_loss_dispatch(hidden_states=None, weights=None, labels=None, chunk_size=size)
    assert seen == [1024, 512, 256]


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU"))]
)
def test_dispatch_loss_and_gradients_match_dense(monkeypatch, device):
    import veomni.ops.kernels.cross_entropy as dispatch

    # Non-SP shifts once; SP receives labels already shifted by the collator.
    for sp_enabled in (False, True):
        monkeypatch.setattr(
            chunk_loss_module, "get_parallel_state", lambda sp=sp_enabled: SimpleNamespace(sp_enabled=sp)
        )
        monkeypatch.setattr(chunk_loss_module, "reduce_sequence_parallel_loss", lambda loss, count: loss)
        torch.manual_seed(73)
        h = torch.randn(2, 13, 7, device=device)
        w = torch.randn(19, 7, device=device)
        labels = torch.randint(0, 19, (2, 13), device=device)
        labels[:, 2:6] = -100  # includes a fully ignored chunk
        refs = None
        for size in (32, 4, 3, 1):
            x, weight = h.clone().requires_grad_(), w.clone().requires_grad_()
            loss, _, _ = dispatch._chunk_loss_dispatch(
                hidden_states=x, weights=weight, labels=labels, vocab_size=19, chunk_size=size
            )
            loss.backward()
            if refs is None:
                xr, wr = h.clone().requires_grad_(), w.clone().requires_grad_()
                logits = torch.nn.functional.linear(xr if sp_enabled else xr[:, :-1], wr)
                target = labels if sp_enabled else labels[:, 1:]
                dense = torch.nn.functional.cross_entropy(logits.reshape(-1, 19), target.reshape(-1))
                dense.backward()
                refs = dense.detach(), xr.grad, wr.grad
            for result, ref in zip((loss, x.grad, weight.grad), refs):
                torch.testing.assert_close(result, ref, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode], ids=["no_grad", "inference_mode"])
@pytest.mark.parametrize("sp_enabled", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3, 16])
def test_evaluation_chunk_loss_matches_dense_without_backward(monkeypatch, mode, sp_enabled, chunk_size):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=sp_enabled))
    monkeypatch.setattr(chunk_loss_module, "reduce_sequence_parallel_loss", lambda loss, count: loss)
    monkeypatch.setattr(
        chunk_loss_module.ChunkLoss,
        "apply",
        lambda *a, **kw: pytest.fail("evaluation must not precompute backward gradients"),
    )
    torch.manual_seed(219)
    # Two samples exercise non-contiguous chunk views; one chunk is fully masked.
    hidden_states = torch.randn(2, 7, 4, requires_grad=True)
    weights = torch.randn(11, 4, requires_grad=True)
    labels = torch.tensor([[1, -100, -100, 4, 5, 6, 7], [2, -100, -100, -100, 6, 7, 8]])
    with mode():
        actual, _ = chunk_loss_module.chunk_loss_function(hidden_states, weights, labels, chunk_size=chunk_size)
        logits = F.linear(hidden_states if sp_enabled else hidden_states[:, :-1], weights).float()
        targets = labels if sp_enabled else labels[:, 1:]
        expected = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert not actual.requires_grad
    assert hidden_states.grad is None
    assert weights.grad is None
