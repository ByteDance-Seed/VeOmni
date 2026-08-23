import os

import pytest
import torch

import veomni.distributed.context_parallel.gdn_kcp as gdn_kcp_module
import veomni.ops.kernels.gdn_kcp_affine_ttx as ttx_module
import veomni.ops.kernels.gdn_kcp_affine_ttx_bwd as ttx_bwd_module
from veomni.distributed.context_parallel.gdn_kcp import (
    assert_kcp_comm_bytes_independent_of_seq,
    local_affine_summary,
    local_affine_summary_fused_torch,
    local_affine_summary_recurrent,
    pack_affine_hm,
    prefix_merge_initial_state,
    resolve_local_affine_impl,
)
from veomni.distributed.context_parallel.gdn_lossless import GdnLosslessRuntimePlan
from veomni.distributed.context_parallel.gdn_ownership import build_gdn_lossless_plan
from veomni.distributed.context_parallel.gdn_runtime import make_gdn_cp_runtime_observer
from veomni.ops.kernels.gated_delta_rule.normalization import producer_dtype_l2norm
from veomni.ops.kernels.gdn_kcp_affine_ttx import (
    TTX_BC8_M1_CONFIG,
    _prepare_ttx_forward_operands,
    _TtxLocalAffineSummaryFn,
    ttx_bc8_m1_torch_reference,
    ttx_local_affine_summary,
    validate_ttx_bc8_m1_contract,
    validate_ttx_bc8_m1_cu_seqlens,
    validate_ttx_bc8_m1_shape,
)
from veomni.utils.device import IS_NPU_AVAILABLE


@pytest.mark.parametrize(("raw", "expected"), [(None, 128), ("128", 128), ("256", 256)])
def test_ttx_backward_chunk_contract(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("VEOMNI_GDN_AFFINE_BWD_CHUNK", raising=False)
    else:
        monkeypatch.setenv("VEOMNI_GDN_AFFINE_BWD_CHUNK", raw)
    assert ttx_bwd_module._bwd_chunk() == expected


@pytest.mark.parametrize("raw", ["", "64", "512", "abc", "128.0"])
def test_ttx_backward_chunk_contract_rejects_unsupported_values(monkeypatch, raw):
    monkeypatch.setenv("VEOMNI_GDN_AFFINE_BWD_CHUNK", raw)
    with pytest.raises(ValueError, match="must be 128 or 256"):
        ttx_bwd_module._bwd_chunk()


@pytest.mark.parametrize(("raw", "expected"), [(None, 8), ("8", 8), ("32", 32)])
def test_ttx_backward_replay_column_tile_contract(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("VEOMNI_GDN_AFFINE_REPLAY_COLUMN_TILE", raising=False)
    else:
        monkeypatch.setenv("VEOMNI_GDN_AFFINE_REPLAY_COLUMN_TILE", raw)
    assert ttx_bwd_module._fwd_coltile_bc() == expected


@pytest.mark.parametrize("raw", ["", "4", "16", "64", "abc", "8.0"])
def test_ttx_backward_replay_column_tile_contract_rejects_unsupported_values(monkeypatch, raw):
    monkeypatch.setenv("VEOMNI_GDN_AFFINE_REPLAY_COLUMN_TILE", raw)
    with pytest.raises(ValueError, match="must be 8 or 32"):
        ttx_bwd_module._fwd_coltile_bc()


def test_all_gather_affine_hm_propagates_zero_grad_through_participation_token_cp1():
    source = torch.tensor(3.0, requires_grad=True)
    participation = source * 0
    local_hm = torch.ones(1, 1, 2, 3, dtype=torch.float32)

    gathered = gdn_kcp_module.all_gather_affine_hm(
        local_hm,
        cp_group=None,
        cp_size=1,
        cp_rank=0,
        participate=participation,
    )
    gathered.sum().backward()

    assert source.grad is not None
    torch.testing.assert_close(source.grad, torch.zeros_like(source), rtol=0, atol=0)


def _inputs(tokens: int = 7):
    torch.manual_seed(17)
    key = torch.randn(1, tokens, 2, 4, dtype=torch.float32) * 0.1
    value = torch.randn(1, tokens, 2, 3, dtype=torch.float32) * 0.1
    g = -torch.rand(1, tokens, 2, dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn(1, tokens, 2, dtype=torch.float32))
    return key, value, g, beta


def test_portable_affine_reference_matches_batched_reference():
    tensors = _inputs()
    eager = local_affine_summary_recurrent(*tensors)
    batched = local_affine_summary_fused_torch(*tensors)
    torch.testing.assert_close(batched, eager, rtol=1e-6, atol=1e-6)
    assert batched.dtype is torch.float32


def test_portable_affine_cu_contract_preserves_empty_segments_as_identity():
    tensors = _inputs(tokens=4)
    cu_seqlens = torch.tensor([0, 0, 4], dtype=torch.int32)

    summary = local_affine_summary_fused_torch(*tensors, cu_seqlens=cu_seqlens)
    he, matrix = gdn_kcp_module.unpack_affine_hm(summary, v_dim=3)

    assert summary.shape[0] == 2
    torch.testing.assert_close(he[0], torch.zeros_like(he[0]), rtol=0, atol=0)
    expected_identity = torch.eye(4, dtype=torch.float32).expand(2, 4, 4)
    torch.testing.assert_close(matrix[0], expected_identity, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("cu_seqlens", "message"),
    [
        (torch.tensor([0], dtype=torch.int32), "at least two boundaries"),
        (torch.tensor([0, 3, 2, 4], dtype=torch.int32), "nondecreasing"),
        (torch.tensor([0, 3], dtype=torch.int32), "end at T=4"),
        (torch.tensor([0.0, 4.0]), "integer tensor"),
    ],
)
def test_portable_affine_cu_contract_rejects_malformed_metadata(cu_seqlens, message):
    with pytest.raises(ValueError, match=message):
        local_affine_summary_fused_torch(*_inputs(tokens=4), cu_seqlens=cu_seqlens)


def test_affine_dispatch_is_explicit_and_fail_closed():
    assert resolve_local_affine_impl() == "ttx"
    assert resolve_local_affine_impl("ttx_bc8_m1") == "ttx"
    assert resolve_local_affine_impl("torch_reference") == "fused_torch"
    with pytest.raises(ValueError, match="Unknown KCP affine implementation"):
        resolve_local_affine_impl("fallback")
    reference = local_affine_summary(*_inputs(), impl="torch_reference")
    assert reference.dtype is torch.float32


def test_ttx_contract_is_typed_and_does_not_mutate_environment():
    def _affine_env() -> dict[str, str]:
        return {name: value for name, value in os.environ.items() if name.startswith("VEOMNI_GDN_AFFINE")}

    before = _affine_env()
    contract = validate_ttx_bc8_m1_contract()
    after = _affine_env()
    assert contract is TTX_BC8_M1_CONFIG
    assert contract.forward_column_tile == 32
    assert contract.backward_time_tile == 128
    assert contract.backward_replay_column_tile == 8
    assert before == after


@pytest.mark.parametrize(("k_dim", "v_dim"), [(96, 128), (128, 160), (0, 128), (128, 0)])
def test_ttx_contract_rejects_shapes_outside_forward_backward_common_domain(k_dim, v_dim):
    with pytest.raises(RuntimeError, match="forward/backward common domain"):
        validate_ttx_bc8_m1_shape(k_dim=k_dim, v_dim=v_dim)


@pytest.mark.parametrize(("k_dim", "v_dim"), [(32, 32), (64, 128), (128, 128)])
def test_ttx_contract_accepts_shapes_in_forward_backward_common_domain(k_dim, v_dim):
    validate_ttx_bc8_m1_shape(k_dim=k_dim, v_dim=v_dim)


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([1, 4], "start at 0"),
        ([0, 2, 1, 4], "nondecreasing"),
        ([0, 3], "end at T=4"),
    ],
)
def test_ttx_forward_cu_contract_rejects_invalid_boundaries(points, message):
    with pytest.raises(ValueError, match=message):
        validate_ttx_bc8_m1_cu_seqlens(
            torch.tensor(points, dtype=torch.int32),
            batch_size=1,
            token_count=4,
        )


def test_ttx_forward_cu_contract_accepts_empty_segments_and_rejects_nonpacked_batch():
    validate_ttx_bc8_m1_cu_seqlens(
        torch.tensor([0, 0, 4], dtype=torch.int32),
        batch_size=1,
        token_count=4,
    )
    with pytest.raises(ValueError, match="batch=1"):
        validate_ttx_bc8_m1_cu_seqlens(
            torch.tensor([0, 4], dtype=torch.int32),
            batch_size=2,
            token_count=4,
        )


def test_ttx_custom_function_rejects_invalid_cu_before_forward_kernel(monkeypatch):
    launched = False

    def fail_if_launched(*args, **kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("TTX forward kernel must not launch for invalid CU metadata")

    monkeypatch.setattr(ttx_module, "_ttx_local_affine_summary_fwd", fail_if_launched)
    key, value, g, beta = _inputs(tokens=4)
    with pytest.raises(ValueError, match="end at T=4"):
        _TtxLocalAffineSummaryFn.apply(
            key,
            value,
            g,
            beta,
            torch.tensor([0, 3], dtype=torch.int32),
            torch.tensor(0.0),
            torch.tensor(1e-6),
            torch.tensor(1.0),
        )
    assert not launched


def test_ttx_operand_contract_preserves_fp32_decay_and_producer_dtypes():
    key, value, g, beta = _inputs(tokens=5)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)
    beta = beta.to(torch.bfloat16)
    # Values deliberately not exactly representable in bf16. Quantizing this
    # operand changes exp(g), so the production contract must retain fp32.
    g = g + torch.linspace(0.00013, 0.00371, g.numel()).reshape_as(g)

    operands = _prepare_ttx_forward_operands(
        key,
        value,
        g,
        beta,
        use_qk_l2norm=False,
        eps=1e-6,
    )
    assert [operand.dtype for operand in operands[:4]] == [
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
    ]
    torch.testing.assert_close(operands[2], g, rtol=0, atol=0)

    expected = local_affine_summary_fused_torch(key, value, g, beta, use_qk_l2norm=False)
    actual = ttx_bc8_m1_torch_reference(key, value, g, beta, use_qk_l2norm=False)
    quantized_g = local_affine_summary_fused_torch(
        key,
        value,
        g.to(torch.bfloat16),
        beta,
        use_qk_l2norm=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, quantized_g)


def test_ttx_operand_contract_rejects_internal_normalization():
    with pytest.raises(ValueError, match="must be pre-normalized"):
        _prepare_ttx_forward_operands(*_inputs(tokens=3), use_qk_l2norm=True, eps=1e-6)


def test_producer_dtype_l2norm_matches_npu_gdr_literal_and_not_fp32_rewrite():
    torch.manual_seed(731)
    key = (torch.randn(2, 7, 3, 128) * 0.3).to(torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        literal = (key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)).to(key.dtype)
        actual = producer_dtype_l2norm(key)
    fp32_rewrite = (key.float() * torch.rsqrt(key.float().pow(2).sum(dim=-1, keepdim=True) + 1e-6)).to(key.dtype)

    torch.testing.assert_close(actual, literal, rtol=0, atol=0)
    assert not torch.equal(actual, fp32_rewrite)


def test_shared_producer_dtype_l2norm_output_and_vjp_match_independent_oracle():
    torch.manual_seed(947)
    candidate = (torch.randn(2, 5, 2, 128) * 0.2).to(torch.bfloat16).requires_grad_(True)
    oracle = candidate.detach().clone().requires_grad_(True)
    actual = producer_dtype_l2norm(candidate)
    expected = (oracle * torch.rsqrt((oracle * oracle).sum(dim=-1, keepdim=True) + 1e-6)).to(oracle.dtype)
    upstream = torch.randn_like(actual)

    actual_grad = torch.autograd.grad(actual, candidate, upstream)[0]
    expected_grad = torch.autograd.grad(expected, oracle, upstream)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_ttx_torch_reference_vjp_matches_local_affine_math():
    base = _inputs(tokens=5)
    candidate_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in base]
    oracle_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in base]
    candidate = ttx_bc8_m1_torch_reference(*candidate_inputs)
    oracle_key_fp32 = oracle_inputs[0].float()
    oracle_rstd = torch.rsqrt(oracle_key_fp32.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
    oracle_key = (oracle_key_fp32 * oracle_rstd).to(dtype=oracle_inputs[0].dtype)
    oracle = local_affine_summary_fused_torch(
        oracle_key,
        *oracle_inputs[1:],
        use_qk_l2norm=False,
    )
    grad_hm = torch.linspace(-0.3, 0.4, candidate.numel(), dtype=torch.float32).reshape_as(candidate)
    candidate_grads = torch.autograd.grad(candidate, candidate_inputs, grad_hm)
    oracle_grads = torch.autograd.grad(oracle, oracle_inputs, grad_hm)

    torch.testing.assert_close(candidate, oracle, rtol=0, atol=0)
    for candidate_grad, oracle_grad in zip(candidate_grads, oracle_grads):
        torch.testing.assert_close(candidate_grad, oracle_grad, rtol=1e-6, atol=1e-6)


def test_ttx_bf16_production_custom_function_consumes_shared_normalized_key(monkeypatch):
    """Exercise the real custom Function wiring without requiring an NPU.

    Stub only the device kernels.  The production Function must still prepare
    and save its exact forward operands, unpack them in backward, and apply the
    normalization VJP to the analytical kernel gradient.
    """

    captured = {}

    def fake_ttx_forward(key, value, g, beta, **kwargs):
        assert kwargs["use_qk_l2norm"] is False
        captured["forward_key"] = key.detach().clone()
        return key.float()

    def fake_ttx_backward(key, value, g, beta, grad_hm, **kwargs):
        assert kwargs["use_qk_l2norm"] is False
        captured["backward_key"] = key.detach().clone()
        return (
            grad_hm.to(dtype=key.dtype),
            torch.zeros_like(value),
            torch.zeros_like(g),
            torch.zeros_like(beta),
        )

    monkeypatch.setattr(ttx_module, "_ttx_local_affine_summary_fwd", fake_ttx_forward)
    monkeypatch.setattr(ttx_bwd_module, "ttx_local_affine_analytical_bwd", fake_ttx_backward)

    torch.manual_seed(919)
    candidate_key = (torch.randn(2, 3, 2, 8) * 0.3).to(torch.bfloat16).requires_grad_(True)
    oracle_key = candidate_key.detach().clone().requires_grad_(True)
    value = torch.randn(2, 3, 2, 5, dtype=torch.bfloat16)
    g = torch.randn(2, 3, 2, dtype=torch.float32)
    beta = torch.randn(2, 3, 2, dtype=torch.bfloat16)
    candidate_normalized = producer_dtype_l2norm(candidate_key)
    candidate = _TtxLocalAffineSummaryFn.apply(
        candidate_normalized,
        value,
        g,
        beta,
        torch.empty(0, dtype=torch.int32),
        torch.tensor(0.0),
        torch.tensor(1e-6),
        torch.tensor(0.0),
    )
    oracle = producer_dtype_l2norm(oracle_key).float()
    upstream = torch.randn_like(candidate)

    candidate_grad = torch.autograd.grad(candidate, candidate_key, upstream)[0]
    oracle_grad = torch.autograd.grad(oracle, oracle_key, upstream)[0]
    torch.testing.assert_close(candidate, oracle, rtol=0, atol=0)
    torch.testing.assert_close(candidate_grad, oracle_grad, rtol=0, atol=0)
    torch.testing.assert_close(captured["forward_key"], oracle.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(captured["backward_key"], captured["forward_key"], rtol=0, atol=0)


def test_ttx_public_autograd_entry_rejects_cpu_before_custom_function():
    with pytest.raises(RuntimeError, match="on one Ascend NPU device"):
        ttx_local_affine_summary(*_inputs(tokens=4))


def test_ttx_forward_backward_warmup_enables_grad_and_caches_only_success(monkeypatch):
    ttx_module._TTX_FORWARD_BACKWARD_WARMUP_CACHE.clear()
    forward_grad_modes = []
    warmup_dtypes = []
    synchronize_calls = []

    monkeypatch.setattr(ttx_module, "validate_ttx_bc8_m1_inputs", lambda *args, **kwargs: None)

    class FakeNpu:
        @staticmethod
        def synchronize():
            synchronize_calls.append(True)

    monkeypatch.setattr(ttx_module.torch, "npu", FakeNpu(), raising=False)

    def fake_summary(key, value, g, beta, **kwargs):
        forward_grad_modes.append(torch.is_grad_enabled())
        warmup_dtypes.append((key.dtype, value.dtype, g.dtype, beta.dtype))
        return key.float().sum() + value.float().sum() + g.float().sum() + beta.float().sum()

    monkeypatch.setattr(ttx_module, "ttx_local_affine_summary", fake_summary)
    key, value, g, beta = _inputs(tokens=0)
    with torch.no_grad():
        ttx_module.warmup_ttx_bc8_m1_forward_backward(key, value, g, beta)
        ttx_module.warmup_ttx_bc8_m1_forward_backward(key, value, g, beta)

    assert forward_grad_modes == [True]
    assert warmup_dtypes == [(key.dtype, value.dtype, g.dtype, beta.dtype)]
    assert synchronize_calls == [True]
    assert len(ttx_module._TTX_FORWARD_BACKWARD_WARMUP_CACHE) == 1

    ttx_module._TTX_FORWARD_BACKWARD_WARMUP_CACHE.clear()
    with torch.inference_mode():
        ttx_module.warmup_ttx_bc8_m1_forward_backward(key, value, g, beta)
    assert forward_grad_modes == [True, True]
    assert synchronize_calls == [True, True]
    assert len(ttx_module._TTX_FORWARD_BACKWARD_WARMUP_CACHE) == 1

    ttx_module._TTX_FORWARD_BACKWARD_WARMUP_CACHE.clear()
    failures = []

    def failing_summary(*args, **kwargs):
        failures.append(True)
        raise RuntimeError("synthetic warmup failure")

    monkeypatch.setattr(ttx_module, "ttx_local_affine_summary", failing_summary)
    with pytest.raises(RuntimeError, match="synthetic warmup failure"):
        ttx_module.warmup_ttx_bc8_m1_forward_backward(key, value, g, beta)
    assert failures == [True]
    assert not ttx_module._TTX_FORWARD_BACKWARD_WARMUP_CACHE


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend NPU")
def test_ttx_production_forward_backward_matches_torch_oracle_on_npu():
    torch.manual_seed(20260814)
    device = torch.device("npu:0")
    cu = torch.tensor([0, 0, 64, 128], dtype=torch.int32, device=device)
    key = (torch.randn(1, 128, 2, 128, device=device) * 0.05).to(torch.bfloat16)
    value = (torch.randn(1, 128, 2, 128, device=device) * 0.05).to(torch.bfloat16)
    g = (-torch.rand(1, 128, 2, device=device) * 0.05).float()
    beta = torch.sigmoid(torch.randn(1, 128, 2, device=device)).to(torch.bfloat16)
    candidate_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (key, value, g, beta)]
    oracle_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (key, value, g, beta)]

    candidate = ttx_local_affine_summary(*candidate_inputs, cu_seqlens=cu, use_qk_l2norm=False)
    oracle = ttx_bc8_m1_torch_reference(*oracle_inputs, cu_seqlens=cu, use_qk_l2norm=False)
    grad_hm = torch.randn_like(candidate) * 0.01
    candidate_grads = torch.autograd.grad(candidate, candidate_inputs, grad_hm)
    oracle_grads = torch.autograd.grad(oracle, oracle_inputs, grad_hm)

    torch.testing.assert_close(candidate, oracle, rtol=2e-3, atol=5e-4)
    for actual, expected in zip(candidate_grads, oracle_grads):
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-3)


def test_prefix_merge_composes_rank_affine_transforms():
    # K=V=1: rank transforms are S <- M*S + he.
    ag_hm = torch.tensor(
        [
            [[[[2.0, 3.0]]]],
            [[[[5.0, 7.0]]]],
            [[[[11.0, 13.0]]]],
        ],
        dtype=torch.float32,
    )
    rank0 = prefix_merge_initial_state(ag_hm, cp_rank=0, v_dim=1)
    rank1 = prefix_merge_initial_state(ag_hm, cp_rank=1, v_dim=1)
    rank2 = prefix_merge_initial_state(ag_hm, cp_rank=2, v_dim=1)
    torch.testing.assert_close(rank0, torch.tensor([[[[0.0]]]]))
    torch.testing.assert_close(rank1, torch.tensor([[[[2.0]]]]))
    torch.testing.assert_close(rank2, torch.tensor([[[[19.0]]]]))


def test_prefix_merge_keeps_fp32_output_and_vjp_under_active_autocast():
    torch.manual_seed(20260823)
    cp_size, num_seqs, num_heads, key_dim, value_dim = 4, 1, 2, 16, 16
    he = torch.randn(cp_size, num_seqs, num_heads, key_dim, value_dim) * 0.01
    eye = torch.eye(key_dim).view(1, 1, 1, key_dim, key_dim)
    matrix = eye + torch.randn(cp_size, num_seqs, num_heads, key_dim, key_dim) * 0.001
    base = pack_affine_hm(he.float(), matrix.float())
    upstream = torch.randn(num_seqs, num_heads, key_dim, value_dim)

    oracle_hm = base.detach().clone().requires_grad_(True)
    oracle = prefix_merge_initial_state(oracle_hm, cp_rank=3, v_dim=value_dim)
    oracle_grad = torch.autograd.grad(oracle, oracle_hm, upstream)[0]

    candidate_hm = base.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        candidate = prefix_merge_initial_state(candidate_hm, cp_rank=3, v_dim=value_dim)
    candidate_grad = torch.autograd.grad(candidate, candidate_hm, upstream)[0]

    assert candidate.dtype == torch.float32
    torch.testing.assert_close(candidate, oracle, rtol=0, atol=0)
    torch.testing.assert_close(candidate_grad, oracle_grad, rtol=0, atol=0)


def test_runtime_identity_seals_lossless_layout_and_ttx_backend():
    global_plan = build_gdn_lossless_plan([256], cp_size=4, ulysses_size=1)
    runtime = GdnLosslessRuntimePlan(global_plan=global_plan, local=global_plan.rank_plan(2))
    observer = make_gdn_cp_runtime_observer("kcp", plan=runtime)
    snapshot = observer.snapshot()
    assert snapshot.identity.implementation == "kcp"
    assert snapshot.identity.layout == "lossless_sparse_packed"
    assert snapshot.identity.affine_backend == "ttx_bc8_m1"
    assert snapshot.identity.ownership_plan_hash == global_plan.plan_hash
    assert snapshot.observed_cp_ranks == (2,)
    assert snapshot.as_dict()["balanced"] is True


def test_affine_collective_bytes_do_not_depend_on_sequence_length():
    expected = 8 * 2 * 4 * (3 + 4) * 4
    assert assert_kcp_comm_bytes_independent_of_seq(cp_size=8, num_heads=2, k_dim=4, v_dim=3) == expected
