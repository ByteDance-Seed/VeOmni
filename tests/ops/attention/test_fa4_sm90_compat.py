# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# See the License for the specific language governing limitations
# under the License.


from functools import partial
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops.kernels.attention.standard.flex import _fa4_sm90_compat as fa4_sm90_compat
from veomni.utils.device import IS_CUDA_AVAILABLE


@pytest.mark.parametrize(
    ("mask_seqlen", "mask_causal", "mask_local", "has_mask", "has_aux", "expected"),
    [
        (False, False, False, True, True, False),
        (True, False, False, True, True, True),
        (False, True, False, True, True, False),
        (False, False, True, True, True, False),
        (False, False, False, False, True, False),
        (False, False, False, True, False, False),
    ],
)
def test_fa4_sm90_compat_keeps_auxiliary_bounds_on_later_partial_tiles(
    mask_seqlen, mask_causal, mask_local, has_mask, has_aux, expected
):
    calls = []
    cute = SimpleNamespace(jit=lambda fn: fn, Tensor=object, TiledMma=object)
    cutlass = SimpleNamespace(Int32=int, Constexpr=dict, const_expr=lambda condition: condition)
    forward = fa4_sm90_compat._make_sm90_mask_forwarder(lambda *args: calls.append(args), cute, cutlass)
    mask = SimpleNamespace(tile_m=128, tile_n=128, seqlen_q=129, seqlen_k=259, qhead_per_kvhead_packgqa=1)
    prefix = (mask, object(), 0, 0, 1, 0, object())
    mask_mod = object() if has_mask else None
    auxiliary = [object()] if has_aux else None
    divisors = (object(), object())
    forward(*prefix, mask_seqlen, mask_causal, mask_local, mask_mod, auxiliary, divisors)
    assert len(calls) == 1
    forwarded_mask = calls[0][10]
    if not mask_seqlen and not mask_causal and not mask_local and has_mask and has_aux:
        assert isinstance(forwarded_mask, partial)
        assert forwarded_mask.keywords == {"mask_mod": mask_mod}
    else:
        assert forwarded_mask is mask_mod
    assert calls == [(*prefix, expected, mask_causal, mask_local, forwarded_mask, auxiliary, divisors)]


@pytest.mark.parametrize(
    ("length", "pack_gqa", "m_block", "expected"),
    [
        (128, 1, 0, False),
        (129, 1, 0, False),
        (129, 1, 1, True),
        (256, 1, 1, False),
        (64, 2, 0, False),
        (65, 2, 1, True),
        (33, 4, 0, False),
        (33, 4, 1, True),
        (128, 7, 6, False),
        (129, 7, 6, False),
        (129, 7, 7, True),
        (127, 7, 6, True),
    ],
)
def test_fa4_additional_bounds_only_cover_physical_query_padding(length, pack_gqa, m_block, expected):
    calls = []
    cute = SimpleNamespace(jit=lambda fn: fn, Tensor=object, TiledMma=object)
    cutlass = SimpleNamespace(Int32=int, Constexpr=dict, const_expr=lambda condition: condition)
    forward = fa4_sm90_compat._make_sm90_mask_forwarder(lambda *args: calls.append(args), cute, cutlass)
    mask = SimpleNamespace(tile_m=128, tile_n=128, seqlen_q=length, seqlen_k=512, qhead_per_kvhead_packgqa=pack_gqa)
    prefix = (mask, object(), 0, 0, m_block, 0, object())
    mask_mod, auxiliary, divisors = object(), [object()], (object(), object())
    forward(*prefix, False, False, False, mask_mod, auxiliary, divisors)
    assert len(calls) == 1
    forwarded_mask = calls[0][10]
    assert isinstance(forwarded_mask, partial) == expected
    if expected:
        assert forwarded_mask.keywords == {"mask_mod": mask_mod}
    else:
        assert forwarded_mask is mask_mod
    assert calls == [(*prefix, False, False, False, forwarded_mask, auxiliary, divisors)]


@pytest.mark.parametrize(
    ("q_length", "k_length", "tile_n", "n_block", "q_only"),
    [
        (128, 129, 128, 1, False),
        (128, 129, 128, -1, False),
        (129, 259, 128, 0, True),
        (129, 259, 128, 1, True),
        (129, 259, 128, 2, False),
        (129, 256, 128, 1, True),
        (129, 256, 128, 2, False),
        (129, 128, 128, 0, True),
        (129, 127, 128, 0, False),
        (129, 259, 128, -1, False),
        (0, 259, 128, 0, False),
        (129, 0, 128, 0, False),
        (129, 259, 256, 0, True),
        (129, 259, 256, 1, False),
    ],
)
def test_fa4_query_only_guard_requires_a_complete_k_tile(q_length, k_length, tile_n, n_block, q_only):
    calls = []
    cute = SimpleNamespace(jit=lambda fn: fn, Tensor=object, TiledMma=object)
    cutlass = SimpleNamespace(Int32=int, Constexpr=dict, const_expr=lambda condition: condition)
    forward = fa4_sm90_compat._make_sm90_mask_forwarder(lambda *args: calls.append(args), cute, cutlass)
    mask = SimpleNamespace(tile_m=128, tile_n=tile_n, seqlen_q=q_length, seqlen_k=k_length, qhead_per_kvhead_packgqa=1)
    prefix = (mask, object(), 0, 0, max(q_length - 1, 0) // 128, n_block, object())
    callback, aux = object(), [object()]
    forward(*prefix, False, False, False, callback, aux)
    assert len(calls) == 1
    assert calls[0][7] == (not q_only)
    assert isinstance(calls[0][10], partial) == q_only
    if not q_only:
        assert calls[0][10] is callback


@pytest.mark.parametrize(
    ("query_idx", "expected_idx", "expected_mask"), [(0, 0, True), (64, 64, True), (65, 0, False), (129, 0, False)]
)
def test_fa4_query_only_callback_uses_logical_indices_and_preserves_arguments(query_idx, expected_idx, expected_mask):
    class Buffer(list):
        def load(self):
            return list(self)

    cute = SimpleNamespace(
        jit=lambda fn: fn, Tensor=object, TiledMma=object, make_rmem_tensor=lambda n, dtype: Buffer([None] * n)
    )
    cutlass = SimpleNamespace(Int32=int, Boolean=bool, Constexpr=dict, const_expr=lambda condition: condition)
    calls = []
    forward = fa4_sm90_compat._make_sm90_mask_forwarder(lambda *args: calls.append(args), cute, cutlass)
    info = SimpleNamespace(seqlen_q=65)
    mask = SimpleNamespace(tile_m=128, tile_n=128, seqlen_q=65, seqlen_k=256, qhead_per_kvhead_packgqa=2)
    batch, head, key, aux = [2], [7], [11], [object()]
    seen = []

    def callback(b, h, q, k, seqlen, tensors):
        seen.append((b, h, q, k, seqlen, tensors))
        return [True]

    forward(mask, object(), 0, 0, 1, 0, object(), False, False, False, callback, aux)
    guarded = calls[0][10]
    assert guarded(batch, head, [query_idx], key, info, aux) == [expected_mask]
    assert seen == [(batch, head, [expected_idx], key, info, aux)]
    assert seen[0][0] is batch and seen[0][1] is head and seen[0][3] is key
    assert seen[0][4] is info and seen[0][5] is aux


def test_fa4_existing_bounds_do_not_enter_tail_dispatch():
    calls = []
    cute = SimpleNamespace(jit=lambda fn: fn, Tensor=object, TiledMma=object)
    cutlass = SimpleNamespace(Int32=int, Constexpr=dict, const_expr=lambda condition: condition)
    forward = fa4_sm90_compat._make_sm90_mask_forwarder(lambda *args: calls.append(args), cute, cutlass)
    # Missing tile attributes make any unintended runtime tail check fail.
    prefix = tuple(object() for _ in range(7))
    mask_mod, auxiliary, divisors = object(), [object()], (object(), object())
    forward(*prefix, True, False, False, mask_mod, auxiliary, divisors)
    assert calls == [(*prefix, True, False, False, mask_mod, auxiliary, divisors)]


@pytest.mark.parametrize("installed_version", [None, "4.0.0b35"])
def test_fa4_sm90_compat_skips_absent_or_different_dependency(monkeypatch, installed_version):
    def package_version(name):
        if installed_version is None:
            raise fa4_sm90_compat.PackageNotFoundError(name)
        return installed_version

    monkeypatch.setattr(fa4_sm90_compat, "version", package_version)
    assert fa4_sm90_compat.install_fa4_sm90_aux_bounds_fix() is False


def test_fa4_sm90_compat_skips_missing_optional_import(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def optional_import(name, *args, **kwargs):
        if name == "cutlass":
            raise ImportError("optional cutlass dependency absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(fa4_sm90_compat, "version", lambda name: "4.0.0b16")
    monkeypatch.setattr(builtins, "__import__", optional_import)
    assert fa4_sm90_compat.install_fa4_sm90_aux_bounds_fix() is False


def test_fa4_sm90_compat_install_is_idempotent():
    if not fa4_sm90_compat.install_fa4_sm90_aux_bounds_fix():
        pytest.skip("requires the pinned FA4 optional dependency")
    from flash_attn.cute import interface
    from flash_attn.cute.mask import AttentionMask

    original = AttentionMask.apply_mask
    sm100 = AttentionMask.apply_mask_sm100
    caches = (interface._flash_attn_fwd.compile_cache, interface._flash_attn_bwd.compile_cache)
    retained = fa4_sm90_compat._retired_compile_caches
    assert fa4_sm90_compat.install_fa4_sm90_aux_bounds_fix()
    assert AttentionMask.apply_mask is original
    assert AttentionMask.apply_mask_sm100 is sm100
    assert (interface._flash_attn_fwd.compile_cache, interface._flash_attn_bwd.compile_cache) == caches
    assert fa4_sm90_compat._retired_compile_caches is retained
    assert len(retained) == 2


@pytest.mark.parametrize("failed_cache", ["fwd", "bwd"])
def test_fa4_sm90_cache_initialization_failure_keeps_installation_retryable(monkeypatch, failed_cache):
    if not fa4_sm90_compat.install_fa4_sm90_aux_bounds_fix():
        pytest.skip("requires the pinned FA4 optional dependency")
    from flash_attn.cute import cache_utils, interface
    from flash_attn.cute.mask import AttentionMask

    def original(
        self,
        acc_S,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        thr_mma,
        mask_seqlen,
        mask_causal,
        mask_local=False,
        mask_mod=None,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        pass

    old_caches = (object(), object())
    retained = fa4_sm90_compat._retired_compile_caches
    monkeypatch.setattr(AttentionMask, "apply_mask", original)
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", old_caches[0])
    monkeypatch.setattr(interface._flash_attn_bwd, "compile_cache", old_caches[1])
    monkeypatch.setattr(fa4_sm90_compat, "_retired_compile_caches", retained)

    def unavailable_cache(name):
        if name.endswith(failed_cache):
            raise OSError("cache directory unavailable")
        return object()

    monkeypatch.setattr(cache_utils, "get_jit_cache", unavailable_cache)
    with pytest.raises(OSError, match="cache directory unavailable"):
        fa4_sm90_compat.install_fa4_sm90_aux_bounds_fix()
    assert AttentionMask.apply_mask is original
    assert interface._flash_attn_fwd.compile_cache is old_caches[0]
    assert interface._flash_attn_bwd.compile_cache is old_caches[1]
    assert fa4_sm90_compat._retired_compile_caches is retained

    replacement = {
        "veomni_sm90_aux_bounds_q_only_k_bounds_materialized_fwd": object(),
        "veomni_sm90_aux_bounds_q_only_k_bounds_materialized_bwd": object(),
    }
    monkeypatch.setattr(cache_utils, "get_jit_cache", replacement.__getitem__)
    assert fa4_sm90_compat.install_fa4_sm90_aux_bounds_fix()
    assert AttentionMask.apply_mask is not original
    assert (
        interface._flash_attn_fwd.compile_cache
        is replacement["veomni_sm90_aux_bounds_q_only_k_bounds_materialized_fwd"]
    )
    assert (
        interface._flash_attn_bwd.compile_cache
        is replacement["veomni_sm90_aux_bounds_q_only_k_bounds_materialized_bwd"]
    )
    assert fa4_sm90_compat._retired_compile_caches == old_caches


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FA4 requires CUDA")
@pytest.mark.parametrize("token_major", [False, True], ids=["bhsd", "thd"])
@pytest.mark.parametrize(("query_heads", "kv_heads"), [(4, 4), (7, 1), (8, 4), (28, 4)])
@pytest.mark.parametrize(("q_length", "k_length"), [(128, 259), (129, 259), (256, 129), (259, 129)])
def test_fa4_tail_bounds_preserve_asymmetric_forward_and_gradients(
    monkeypatch,
    query_heads,
    kv_heads,
    q_length,
    k_length,
    token_major,
    record_property,
    permute_kv=False,
    outer_compile=False,
):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.functional import scaled_dot_product_attention

    from veomni.ops.kernels.attention.standard import flex

    if flex.get_gpu_compute_capability(torch.device("cuda")) != 90 or not flex._fa4._flash_attn_cute_available():
        pytest.skip("requires NVIDIA SM90 and FA4 for the SM90 compatibility tests")
    q_groups = torch.arange(q_length, device="cuda", dtype=torch.int32) % 5
    k_groups = torch.arange(k_length, device="cuda", dtype=torch.int32) % 5

    def mask_mod(batch, head, q, k):
        return q_groups[q] == k_groups[k]

    mask = create_block_mask(mask_mod, None, None, q_length, k_length, device="cuda", BLOCK_SIZE=128)
    if permute_kv:
        # Force the ragged K block into a later False-mask call. Native b16
        # does not validate sparse ordering; the compatibility fallback must
        # retain full bounds protection when the K tile is incomplete.
        assert mask.kv_indices.shape[-1] == 2
        assert torch.all(mask.kv_num_blocks == 2)
        mask.kv_indices = mask.kv_indices.clone()
        mask.kv_indices[..., -1, :] = mask.kv_indices[..., -1, :].flip(-1)
    dense = q_groups[:, None] == k_groups[None, :]
    inputs = [
        torch.randn(1, heads, length, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for heads, length in [(query_heads, q_length), (kv_heads, k_length), (kv_heads, k_length)]
    ]
    references = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    if token_major:
        inputs = [tensor.squeeze(0).transpose(0, 1).contiguous().detach().requires_grad_() for tensor in inputs]
    torch.compiler.reset()
    flex._fa4._compiled_flex_attention_fa4.cache_clear()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            expected = scaled_dot_product_attention(*references, attn_mask=dense, enable_gqa=True).transpose(1, 2)
        if token_major:
            expected = expected.squeeze(0)
        if outer_compile:
            module = torch.nn.Module()

            def outer(q, k, v):
                return flex._flex_attention_fa4(
                    module,
                    q,
                    k,
                    v,
                    mask,
                    dropout=0.0,
                    scaling=128**-0.5,
                    softcap=None,
                    kernel_options={"BACKEND": "FLASH"},
                )

            actual, auxiliary = torch.compile(outer, dynamic=False, fullgraph=True)(*inputs)
        else:
            actual, auxiliary = flex.flex_attention_forward(
                torch.nn.Module(),
                *inputs,
                mask,
                scaling=128**-0.5,
                kernel_options={"BACKEND": "FLASH"},
            )
        assert auxiliary is None
        assert torch.isfinite(actual).all()
        record_property("output_max_abs_error", (actual.float() - expected.float()).abs().max().item())
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=2e-2)
        gradient = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, inputs, gradient)
        expected_grads = torch.autograd.grad(expected, references, gradient)
        for name, actual_grad, expected_grad in zip(("q", "k", "v"), actual_grads, expected_grads, strict=True):
            if token_major:
                expected_grad = expected_grad.squeeze(0).transpose(0, 1)
            relative_l2 = (actual_grad.float() - expected_grad.float()).norm() / expected_grad.float().norm()
            record_property(f"{name}_grad_relative_l2", relative_l2.item())
            assert torch.isfinite(actual_grad).all()
            assert (actual_grad.float() - expected_grad.float()).norm() <= 0.01 * expected_grad.float().norm()
    finally:
        torch.compiler.reset()
        flex._fa4._compiled_flex_attention_fa4.cache_clear()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FA4 requires CUDA")
@pytest.mark.parametrize("token_major", [False, True], ids=["bhsd", "thd"])
@pytest.mark.parametrize("q_length", [128, 129], ids=["complete_q", "padded_q"])
def test_fa4_tail_bounds_keep_ragged_k_protection_for_later_sparse_blocks(
    monkeypatch, token_major, record_property, q_length
):
    test_fa4_tail_bounds_preserve_asymmetric_forward_and_gradients(
        monkeypatch, 7, 1, q_length, 129, token_major, record_property, permute_kv=True
    )


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FA4 requires CUDA")
@pytest.mark.parametrize("broadcast_first", [False, True])
def test_fa4_shared_cache_broadcast_layouts(monkeypatch, record_property, broadcast_first):
    # Reuse the production cache in both orders. The 7/1 case expands K/V to
    # zero-stride views before the adapter materializes them for native FA4.
    head_pairs = [(7, 1), (4, 4), (7, 1)] if broadcast_first else [(4, 4), (7, 1), (4, 4)]
    for query_heads, kv_heads in head_pairs:
        test_fa4_tail_bounds_preserve_asymmetric_forward_and_gradients(
            monkeypatch, query_heads, kv_heads, 129, 259, False, record_property
        )


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FA4 requires CUDA")
def test_fa4_shared_cache_outer_compile_materializes_broadcast(monkeypatch, record_property):
    from functools import wraps

    from veomni.ops.kernels.attention.standard import flex

    if flex.get_gpu_compute_capability(torch.device("cuda")) != 90:
        pytest.skip("requires NVIDIA SM90 for the SM90 compatibility tests")

    from flash_attn.cute import interface

    original = interface._flash_attn_fwd
    observed_strides = []

    @wraps(original)
    def observed(*args, **kwargs):
        tensors = args[:3] if args else (kwargs["q"], kwargs["k"], kwargs["v"])
        observed_strides.append(tuple(tensor.stride() for tensor in tensors))
        assert all(0 not in tensor.stride() for tensor in tensors)
        return original(*args, **kwargs)

    monkeypatch.setattr(interface, "_flash_attn_fwd", observed)
    for query_heads, kv_heads in ((4, 4), (7, 1), (4, 4)):
        test_fa4_tail_bounds_preserve_asymmetric_forward_and_gradients(
            monkeypatch, query_heads, kv_heads, 129, 259, False, record_property, outer_compile=True
        )
    assert observed_strides


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("capability", [80, 89, 100])
def test_fa4_outer_compile_skips_other_architectures_before_optional_import(monkeypatch, capability):
    import builtins

    from veomni.ops.kernels.attention.standard import flex

    original_import = builtins.__import__

    def reject_fa4_import(name, *args, **kwargs):
        if name == "flash_attn.cute":
            raise AssertionError("An unsupported GPU must not import FA4")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(flex, "get_gpu_compute_capability", lambda device: capability)
    monkeypatch.setattr(builtins, "__import__", reject_fa4_import)
    with pytest.raises(pytest.skip.Exception, match="requires NVIDIA SM90"):
        test_fa4_shared_cache_outer_compile_materializes_broadcast(monkeypatch, lambda *args: None)
