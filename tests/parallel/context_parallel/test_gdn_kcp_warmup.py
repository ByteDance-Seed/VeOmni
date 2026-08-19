"""Host tests for checkpoint-safe KCP TTX preparation."""

import torch

from veomni.distributed.context_parallel import gdn_kcp


def _kwargs(reference):
    return dict(
        device=reference.device,
        num_heads=4,
        key_dim=8,
        value_dim=16,
        key_dtype=torch.bfloat16,
        value_dtype=torch.bfloat16,
        g_dtype=torch.float32,
        beta_dtype=torch.bfloat16,
        cp_group=None,
        reference=reference,
    )


def test_prepare_kcp_ttx_warmup_runs_before_checkpoint(monkeypatch):
    calls = []

    def fake_warmup(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "veomni.ops.kernels.gdn_kcp_affine_ttx.warmup_ttx_bc8_m1_forward_backward_for_shapes",
        fake_warmup,
    )
    monkeypatch.setattr(gdn_kcp.dist, "get_world_size", lambda group: 1)
    reference = torch.zeros(1)

    gdn_kcp.prepare_kcp_ttx_warmup(**_kwargs(reference))

    assert len(calls) == 1
    assert calls[0]["num_heads"] == 4
    assert calls[0]["key_dim"] == 8
    assert calls[0]["value_dim"] == 16


def test_prepare_kcp_ttx_warmup_fails_closed(monkeypatch):
    def failed_warmup(**kwargs):
        raise RuntimeError("synthetic compile failure")

    monkeypatch.setattr(
        "veomni.ops.kernels.gdn_kcp_affine_ttx.warmup_ttx_bc8_m1_forward_backward_for_shapes",
        failed_warmup,
    )
    monkeypatch.setattr(gdn_kcp.dist, "get_world_size", lambda group: 1)

    try:
        gdn_kcp.prepare_kcp_ttx_warmup(**_kwargs(torch.zeros(1)))
    except RuntimeError as exc:
        assert "coordinated KCP TTX warmup failed" in str(exc)
        assert "synthetic compile failure" in str(exc)
    else:  # pragma: no cover - the helper must never silently continue
        raise AssertionError("warmup failure was not propagated")
