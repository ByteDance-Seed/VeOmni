import pytest
import torch

from veomni.distributed.context_parallel.gdn_kcp import local_affine_summary, resolve_local_affine_impl
from veomni.ops.kernels.gated_delta_rule.affine_provider import (
    external_kcp_affine_summary,
    get_external_kcp_affine_summary_identity,
    prepare_external_kcp_affine_summary,
    register_external_kcp_affine_summary,
)
from veomni.ops.kernels.gated_delta_rule.backend_adapter import requires_external_kcp_affine


def test_external_kcp_affine_provider_registers_and_preserves_vjp(monkeypatch):
    monkeypatch.setattr(
        "veomni.ops.kernels.gated_delta_rule.affine_provider._EXTERNAL_KCP_AFFINE_SUMMARIES",
        {},
    )

    def provider(key, value, g, beta, *, cu_seqlens, use_qk_l2norm, eps):
        del cu_seqlens, use_qk_l2norm, eps
        total = key.sum() + value.sum() + g.sum() + beta.sum()
        return total.expand(1, 2, 3, 7).float()

    register_external_kcp_affine_summary("mojo", provider, identity="test.mojo.affine.v1")

    inputs = [torch.randn(1, 4, 2, width, requires_grad=True) for width in (3, 4)]
    g = torch.randn(1, 4, 2, requires_grad=True)
    beta = torch.randn(1, 4, 2, requires_grad=True)
    hm = external_kcp_affine_summary(
        inputs[0],
        inputs[1],
        g,
        beta,
        implementation="mojo",
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        use_qk_l2norm=False,
        eps=1e-6,
    )
    hm.sum().backward()

    assert get_external_kcp_affine_summary_identity("mojo") == "test.mojo.affine.v1"
    assert hm.shape == (1, 2, 3, 7)
    assert hm.dtype == torch.float32
    assert all(tensor.grad is not None for tensor in (*inputs, g, beta))


def test_only_mojo_declares_an_external_kcp_affine_requirement():
    assert requires_external_kcp_affine("mojo")
    assert not requires_external_kcp_affine("npu")
    assert not requires_external_kcp_affine("npu_ascendc")


def test_kcp_dispatches_explicit_external_provider_without_fallback(monkeypatch):
    monkeypatch.setattr(
        "veomni.ops.kernels.gated_delta_rule.affine_provider._EXTERNAL_KCP_AFFINE_SUMMARIES",
        {},
    )
    calls = []

    def provider(key, value, g, beta, *, cu_seqlens, use_qk_l2norm, eps):
        calls.append((key, value, g, beta, cu_seqlens, use_qk_l2norm, eps))
        return torch.zeros(1, 2, 3, 7, dtype=torch.float32)

    register_external_kcp_affine_summary("mojo", provider, identity="test.mojo.affine.v1")
    key = torch.zeros(1, 4, 2, 3)
    value = torch.zeros(1, 4, 2, 4)
    g = torch.zeros(1, 4, 2)
    beta = torch.zeros(1, 4, 2)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)

    assert resolve_local_affine_impl("external:mojo") == "external:mojo"
    hm = local_affine_summary(
        key,
        value,
        g,
        beta,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm=False,
        impl="external:mojo",
    )

    assert hm.shape == (1, 2, 3, 7)
    assert calls == [(key, value, g, beta, cu_seqlens, False, 1e-6)]


def test_external_kcp_affine_registration_and_output_contract_fail_closed(monkeypatch):
    monkeypatch.setattr(
        "veomni.ops.kernels.gated_delta_rule.affine_provider._EXTERNAL_KCP_AFFINE_SUMMARIES",
        {},
    )
    with pytest.raises(RuntimeError, match="not registered"):
        get_external_kcp_affine_summary_identity("mojo")

    def provider(key, value, g, beta, *, cu_seqlens, use_qk_l2norm, eps):
        del key, value, g, beta, cu_seqlens, use_qk_l2norm, eps
        return torch.zeros(1, 2, 3, 6)

    register_external_kcp_affine_summary("mojo", provider, identity="test.mojo.affine.v1")
    register_external_kcp_affine_summary("mojo", provider, identity="test.mojo.affine.v1")
    with pytest.raises(RuntimeError, match="different identity"):
        register_external_kcp_affine_summary("mojo", provider, identity="test.mojo.affine.v2")
    with pytest.raises(RuntimeError, match="hm contract"):
        external_kcp_affine_summary(
            torch.zeros(1, 4, 2, 3),
            torch.zeros(1, 4, 2, 4),
            torch.zeros(1, 4, 2),
            torch.zeros(1, 4, 2),
            implementation="mojo",
            cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            use_qk_l2norm=False,
            eps=1e-6,
        )


def test_external_kcp_affine_prepare_uses_shape_only_contract(monkeypatch):
    monkeypatch.setattr(
        "veomni.ops.kernels.gated_delta_rule.affine_provider._EXTERNAL_KCP_AFFINE_SUMMARIES",
        {},
    )
    seen = {}

    def provider(*args, **kwargs):
        raise AssertionError("provider should not execute during prepare")

    def prepare(**kwargs):
        seen.update(kwargs)

    register_external_kcp_affine_summary(
        "mojo",
        provider,
        identity="test.mojo.affine.v1",
        prepare=prepare,
    )
    prepare_external_kcp_affine_summary(
        "mojo",
        device=torch.device("cpu"),
        num_heads=2,
        key_dim=3,
        value_dim=4,
        key_dtype=torch.bfloat16,
        value_dtype=torch.bfloat16,
        g_dtype=torch.float32,
        beta_dtype=torch.bfloat16,
    )
    assert seen == {
        "device": torch.device("cpu"),
        "num_heads": 2,
        "key_dim": 3,
        "value_dim": 4,
        "key_dtype": torch.bfloat16,
        "value_dtype": torch.bfloat16,
        "g_dtype": torch.float32,
        "beta_dtype": torch.bfloat16,
    }


def test_external_kcp_affine_forwards_canonical_host_cu_only_when_provided(monkeypatch):
    monkeypatch.setattr(
        "veomni.ops.kernels.gated_delta_rule.affine_provider._EXTERNAL_KCP_AFFINE_SUMMARIES",
        {},
    )
    seen = {}

    def provider(key, value, g, beta, *, cu_seqlens, cu_seqlens_list, use_qk_l2norm, eps):
        del g, beta, use_qk_l2norm, eps
        seen["cu"] = cu_seqlens
        seen["host"] = cu_seqlens_list
        return torch.zeros(2, 2, key.shape[-1], value.shape[-1] + key.shape[-1], dtype=torch.float32)

    register_external_kcp_affine_summary("mojo", provider, identity="test.mojo.host-cu.v1")
    cu = torch.tensor([0, 0, 4], dtype=torch.int32)
    inputs = [torch.zeros(1, 4, 2, width) for width in (3, 4)]
    external_kcp_affine_summary(
        inputs[0],
        inputs[1],
        torch.zeros(1, 4, 2),
        torch.zeros(1, 4, 2),
        implementation="mojo",
        cu_seqlens=cu,
        cu_seqlens_list=(0, 0, 4),
        use_qk_l2norm=False,
        eps=1e-6,
    )
    assert seen == {"cu": cu, "host": (0, 0, 4)}
