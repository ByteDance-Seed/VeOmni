"""Tests for the explicit Qwen3.5 GDN metadata ABI adapter."""

import pytest
import torch

from veomni.ops.kernels.gated_delta_rule.backend_adapter import (
    build_gated_delta_rule_metadata_kwargs,
    call_chunk_gated_delta_rule,
    requires_chunked_varlen_metadata,
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
