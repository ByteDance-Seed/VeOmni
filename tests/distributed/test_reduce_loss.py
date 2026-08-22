import torch
import torch.distributed as dist

from veomni.distributed.sequence_parallel.loss import ReduceLoss


def test_reduce_loss_backward_is_token_weighted_mean(monkeypatch):
    # Simulate a 2-rank sequence-parallel group without a real process group:
    #   rank 0: local mean loss 2.0 over n0=3 valid tokens
    #   rank 1: local mean loss 4.0 over n1=5 valid tokens
    # ReduceLoss.forward computes the token-weighted global mean
    #   (2.0*3 + 4.0*5) / (3 + 5),
    # so d(loss_out)/d(loss_0) must be n0 / (n0 + n1) = 3 / 8, with no extra
    # world-size factor.
    n0, n1 = 3.0, 5.0
    loss1 = 4.0

    calls = []

    def fake_all_reduce(t, group=None):
        calls.append(t)
        if len(calls) == 1:
            # First call reduces `loss` (already multiplied by local n0); add
            # rank 1's contribution loss1 * n1.
            t.add_(loss1 * n1)
        else:
            # Second call reduces `num_valid_tokens`; add rank 1's token count.
            t.add_(n1)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)

    loss = torch.tensor([2.0], requires_grad=True)
    num_valid_tokens = torch.tensor([n0])
    out = ReduceLoss.apply(loss, num_valid_tokens, object())

    assert torch.allclose(out, torch.tensor([26.0 / 8.0]), atol=1e-6)

    out.backward()
    assert torch.allclose(loss.grad, torch.tensor([3.0 / 8.0]), atol=1e-6)
