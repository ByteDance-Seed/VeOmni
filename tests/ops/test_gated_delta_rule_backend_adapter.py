"""Tests for the explicit Qwen3.5 GDN metadata ABI adapter."""

import pytest
import torch

from veomni.ops.kernels.gated_delta_rule.backend_adapter import (
    build_gated_delta_rule_metadata_kwargs,
    call_chunk_gated_delta_rule,
    prepare_gated_delta_rule_qk,
    requires_chunked_varlen_metadata,
)
from veomni.ops.kernels.gated_delta_rule.normalization import (
    get_external_gated_delta_rule_l2norm_identity,
    register_external_gated_delta_rule_l2norm,
)


def _inputs():
    return dict(
        query=torch.zeros(1, 4, 2, 8),
        key=torch.zeros(1, 4, 2, 8),
        value=torch.zeros(1, 4, 2, 16),
        g=torch.zeros(1, 4, 2),
        beta=torch.zeros(1, 4, 2),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        cu_seqlens_list=[0, 4],
        chunk_indices={"start": torch.tensor([0])},
        chunk_indices_list={"start": [0]},
    )


def test_mojo_metadata_contract_is_cu_only():
    values = _inputs()
    values["chunk_indices"] = None
    values["chunk_indices_list"] = None
    with pytest.raises(ValueError, match="canonical proof"):
        build_gated_delta_rule_metadata_kwargs(
            "mojo",
            cu_seqlens=values["cu_seqlens"],
            cu_seqlens_list=values["cu_seqlens_list"],
            chunk_indices=None,
            chunk_indices_list=None,
        )
    kwargs = build_gated_delta_rule_metadata_kwargs(
        "mojo",
        cu_seqlens=values["cu_seqlens"],
        cu_seqlens_list=values["cu_seqlens_list"],
        chunk_indices=values["chunk_indices"],
        chunk_indices_list=values["chunk_indices_list"],
        metadata_is_canonical=True,
    )
    assert set(kwargs) == {"cu_seqlens"}
    assert kwargs["cu_seqlens"] is values["cu_seqlens"]
    assert not requires_chunked_varlen_metadata("mojo")


def test_ascendc_receives_full_metadata():
    values = _inputs()
    kwargs = build_gated_delta_rule_metadata_kwargs(
        "npu_ascendc",
        cu_seqlens=values["cu_seqlens"],
        cu_seqlens_list=values["cu_seqlens_list"],
        chunk_indices=values["chunk_indices"],
        chunk_indices_list=values["chunk_indices_list"],
    )
    assert set(kwargs) == {"cu_seqlens", "cu_seqlens_list", "chunk_indices", "chunk_indices_list"}
    assert requires_chunked_varlen_metadata("npu_ascendc")


def test_vendored_npu_receives_host_cu_but_not_chunk_maps():
    values = _inputs()
    values["chunk_indices"] = None
    values["chunk_indices_list"] = None
    kwargs = build_gated_delta_rule_metadata_kwargs(
        "npu",
        cu_seqlens=values["cu_seqlens"],
        cu_seqlens_list=values["cu_seqlens_list"],
        chunk_indices=None,
        chunk_indices_list=None,
    )
    assert set(kwargs) == {"cu_seqlens", "cu_seqlens_list"}
    assert not requires_chunked_varlen_metadata("npu")


def test_call_mojo_does_not_forward_unsupported_keywords():
    values = _inputs()
    values["chunk_indices"] = None
    values["chunk_indices_list"] = None
    seen = {}

    def strict_mojo(
        query, key, value, *, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens
    ):
        seen.update(
            {
                "query": query,
                "key": key,
                "value": value,
                "g": g,
                "beta": beta,
                "initial_state": initial_state,
                "output_final_state": output_final_state,
                "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
                "cu_seqlens": cu_seqlens,
            }
        )
        return "ok"

    result = call_chunk_gated_delta_rule(
        strict_mojo,
        implementation="mojo",
        metadata_is_canonical=True,
        **values,
    )
    assert result == "ok"
    assert set(seen) == {
        "query",
        "key",
        "value",
        "g",
        "beta",
        "initial_state",
        "output_final_state",
        "use_qk_l2norm_in_kernel",
        "cu_seqlens",
    }


def test_call_ascendc_forwards_full_contract():
    values = _inputs()
    seen = {}

    def strict_ascendc(query, key, value, **kwargs):
        seen.update(kwargs)
        return "ok"

    assert call_chunk_gated_delta_rule(strict_ascendc, implementation="npu_ascendc", **values) == "ok"
    assert {"cu_seqlens", "cu_seqlens_list", "chunk_indices", "chunk_indices_list"} <= set(seen)


def test_unknown_backend_fails_closed():
    values = _inputs()
    with pytest.raises(RuntimeError, match="no declared metadata ABI"):
        build_gated_delta_rule_metadata_kwargs(
            "unknown",
            cu_seqlens=values["cu_seqlens"],
            cu_seqlens_list=values["cu_seqlens_list"],
            chunk_indices=values["chunk_indices"],
            chunk_indices_list=values["chunk_indices_list"],
            metadata_is_canonical=True,
        )


def test_metadata_length_mismatch_fails_closed():
    values = _inputs()
    with pytest.raises(ValueError, match="length mismatch"):
        build_gated_delta_rule_metadata_kwargs(
            "mojo",
            cu_seqlens=values["cu_seqlens"],
            cu_seqlens_list=[0, 2, 4],
            chunk_indices=None,
            chunk_indices_list=None,
        )


def test_mojo_chunk_metadata_fails_closed_instead_of_being_dropped():
    values = _inputs()
    with pytest.raises(ValueError, match="does not accept chunk metadata"):
        build_gated_delta_rule_metadata_kwargs(
            "mojo",
            cu_seqlens=values["cu_seqlens"],
            cu_seqlens_list=values["cu_seqlens_list"],
            chunk_indices=values["chunk_indices"],
            chunk_indices_list=values["chunk_indices_list"],
        )


def test_host_device_cu_value_mismatch_fails_closed():
    values = _inputs()
    values["chunk_indices"] = None
    values["chunk_indices_list"] = None
    with pytest.raises(ValueError, match="values mismatch"):
        build_gated_delta_rule_metadata_kwargs(
            "mojo",
            cu_seqlens=values["cu_seqlens"],
            cu_seqlens_list=[0, 3],
            chunk_indices=None,
            chunk_indices_list=None,
            metadata_is_canonical=True,
        )


def test_empty_segment_metadata_is_preserved_for_ascendc():
    values = _inputs()
    values["cu_seqlens"] = torch.tensor([0, 0, 4], dtype=torch.int32)
    values["cu_seqlens_list"] = [0, 0, 4]
    values["chunk_indices"] = {"start": torch.tensor([], dtype=torch.int32)}
    values["chunk_indices_list"] = {"start": []}
    kwargs = build_gated_delta_rule_metadata_kwargs(
        "npu_ascendc",
        cu_seqlens=values["cu_seqlens"],
        cu_seqlens_list=values["cu_seqlens_list"],
        chunk_indices=values["chunk_indices"],
        chunk_indices_list=values["chunk_indices_list"],
    )
    assert kwargs["cu_seqlens_list"] == [0, 0, 4]


def test_chunk_metadata_key_sets_must_match():
    values = _inputs()
    values["chunk_indices_list"] = {"other": [0]}
    with pytest.raises(ValueError, match="chunk metadata key sets differ"):
        build_gated_delta_rule_metadata_kwargs(
            "npu_ascendc",
            cu_seqlens=values["cu_seqlens"],
            cu_seqlens_list=values["cu_seqlens_list"],
            chunk_indices=values["chunk_indices"],
            chunk_indices_list=values["chunk_indices_list"],
        )


def test_mojo_qk_norm_fails_closed_without_external_provider(monkeypatch):
    monkeypatch.setattr(
        "veomni.ops.kernels.gated_delta_rule.normalization._EXTERNAL_L2NORM_PROVIDERS",
        {},
    )
    values = _inputs()
    with pytest.raises(RuntimeError, match="external L2Norm provider is not registered"):
        get_external_gated_delta_rule_l2norm_identity("mojo")
    with pytest.raises(RuntimeError, match="external L2Norm provider is not registered"):
        prepare_gated_delta_rule_qk(values["query"], values["key"], implementation="mojo")


def test_mojo_qk_norm_uses_registered_provider_once_per_tensor_and_disables_kernel_norm(monkeypatch):
    monkeypatch.setattr(
        "veomni.ops.kernels.gated_delta_rule.normalization._EXTERNAL_L2NORM_PROVIDERS",
        {},
    )
    calls = []

    def exact_provider(tensor, *, eps):
        calls.append((tensor, eps))
        return tensor + 3

    register_external_gated_delta_rule_l2norm("mojo", exact_provider, identity="test.mojo.l2norm.v1")
    values = _inputs()
    query, key, use_kernel_norm = prepare_gated_delta_rule_qk(values["query"], values["key"], implementation="mojo")

    assert calls == [(values["query"], 1e-6), (values["key"], 1e-6)]
    assert torch.equal(query, values["query"] + 3)
    assert torch.equal(key, values["key"] + 3)
    assert not use_kernel_norm


def test_mojo_external_norm_preserves_exact_custom_vjp(monkeypatch):
    monkeypatch.setattr(
        "veomni.ops.kernels.gated_delta_rule.normalization._EXTERNAL_L2NORM_PROVIDERS",
        {},
    )

    class ExactNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            return tensor * 2

        @staticmethod
        def backward(ctx, grad_output):
            # Deliberately differs from the derivative of ``tensor * 2``.  The
            # adapter must preserve the provider's custom backward rather than
            # rebuild normalization with Open-VeOmni tensor expressions.
            return grad_output * 7

    def exact_provider(tensor, *, eps):
        assert eps == 1e-6
        return ExactNorm.apply(tensor)

    register_external_gated_delta_rule_l2norm("mojo", exact_provider, identity="test.mojo.custom-vjp")
    query = torch.randn(2, 4, requires_grad=True)
    key = torch.randn(2, 4, requires_grad=True)
    norm_query, norm_key, use_kernel_norm = prepare_gated_delta_rule_qk(
        query,
        key,
        implementation="mojo",
    )
    (norm_query.sum() + norm_key.sum()).backward()

    assert not use_kernel_norm
    assert torch.equal(query.grad, torch.full_like(query.grad, 7))
    assert torch.equal(key.grad, torch.full_like(key.grad, 7))
    assert torch.isfinite(norm_query).all()
    assert torch.isfinite(norm_key).all()


def test_external_provider_registration_is_idempotent_but_rejects_identity_drift(monkeypatch):
    monkeypatch.setattr(
        "veomni.ops.kernels.gated_delta_rule.normalization._EXTERNAL_L2NORM_PROVIDERS",
        {},
    )

    def provider(tensor, *, eps):
        return tensor

    register_external_gated_delta_rule_l2norm("mojo", provider, identity="test.mojo.l2norm.v1")
    register_external_gated_delta_rule_l2norm("mojo", provider, identity="test.mojo.l2norm.v1")
    assert get_external_gated_delta_rule_l2norm_identity("mojo") == "test.mojo.l2norm.v1"
    with pytest.raises(RuntimeError, match="already registered with a different identity"):
        register_external_gated_delta_rule_l2norm("mojo", provider, identity="test.mojo.l2norm.v2")

    def other_provider(tensor, *, eps):
        return tensor

    with pytest.raises(RuntimeError, match="identity is already bound to a different callable"):
        register_external_gated_delta_rule_l2norm("mojo", other_provider, identity="test.mojo.l2norm.v1")


def test_external_provider_output_contract_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "veomni.ops.kernels.gated_delta_rule.normalization._EXTERNAL_L2NORM_PROVIDERS",
        {},
    )

    def wrong_dtype(tensor, *, eps):
        return tensor.double()

    register_external_gated_delta_rule_l2norm("mojo", wrong_dtype, identity="test.mojo.bad-dtype")
    values = _inputs()
    with pytest.raises(RuntimeError, match="changed the tensor contract"):
        prepare_gated_delta_rule_qk(values["query"], values["key"], implementation="mojo")


def test_non_mojo_default_keeps_backend_internal_norm():
    values = _inputs()
    query, key, use_kernel_norm = prepare_gated_delta_rule_qk(values["query"], values["key"], implementation="npu")
    assert query is values["query"]
    assert key is values["key"]
    assert use_kernel_norm
