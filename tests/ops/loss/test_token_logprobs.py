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

"""Tests for ``token_logprobs`` op implementations and NPU fast-path benchmark."""

from __future__ import annotations

import argparse
import sys
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import veomni.models.loss_utils.chunk_logprobs as cl
from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.loss_utils.chunk_logprobs import (
    _per_token_log_probs_from_logits,
    chunk_logprobs_function,
)
from veomni.ops import OP_REGISTRY, VeomniOp, resolve_op
from veomni.ops.kernels.loss.token_logprobs.chunked import resolve_token_logprobs_op
from veomni.ops.kernels.loss.token_logprobs.standard import eager as tlp_eager
from veomni.ops.kernels.loss.token_logprobs.standard import flash_attn as tlp_flash_attn
from veomni.ops.kernels.loss.token_logprobs.standard import torch_npu as tlp_torch_npu
from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.device import IS_NPU_AVAILABLE


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_token_logprobs_registered_entries():
    """Verify ``token_logprobs`` rows are registered and resolve via ``VeomniOp``."""
    assert set(OP_REGISTRY.list_registered("token_logprobs", "standard")) == {
        "eager",
        "flash_attn",
        "torch_npu",
    }
    eager_entry = resolve_op("token_logprobs", "standard", "eager")
    assert eager_entry.wrapper is tlp_eager.wrapper

    op = resolve_token_logprobs_op(device_type="cpu")
    assert isinstance(op, VeomniOp)
    assert op.impl == "eager"
    assert op.entry.wrapper is tlp_eager.wrapper


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("batch_shape", [(16,), (4, 8), (2, 3, 5)])
@pytest.mark.parametrize("label_dtype", [torch.int32, torch.int64])
def test_token_logprobs_torch_npu_wrapper_contract(monkeypatch, batch_shape, label_dtype):
    """Verify shape flattening/restoration, int64 label cast, and sign negation in ``torch_npu`` row."""
    torch.manual_seed(0)
    vocab_size = 37
    logits = torch.randn(*batch_shape, vocab_size, dtype=torch.float32)
    labels = torch.randint(0, vocab_size, batch_shape, dtype=label_dtype)
    labels.view(-1)[::5] = IGNORE_INDEX

    recorded_calls: list[tuple[torch.Size, torch.Size, torch.dtype, str, int]] = []

    def fake_npu_cross_entropy_loss(
        logits_2d: torch.Tensor,
        labels_1d: torch.Tensor,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        recorded_calls.append((logits_2d.shape, labels_1d.shape, labels_1d.dtype, reduction, ignore_index))
        nll = F.cross_entropy(logits_2d, labels_1d, reduction="none", ignore_index=ignore_index)
        return nll, logits_2d.log_softmax(dim=-1), None, None

    monkeypatch.setattr(tlp_torch_npu, "IS_NPU_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_cross_entropy_loss=fake_npu_cross_entropy_loss))

    out = tlp_torch_npu.wrapper(logits, labels, ignore_index=IGNORE_INDEX)
    ref = tlp_eager.wrapper(logits, labels.to(torch.int64), ignore_index=IGNORE_INDEX)

    num_tokens = logits.numel() // vocab_size
    assert recorded_calls == [
        (torch.Size([num_tokens, vocab_size]), torch.Size([num_tokens]), torch.int64, "none", IGNORE_INDEX)
    ]
    assert out.shape == torch.Size(batch_shape)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
    assert torch.all(out[labels == IGNORE_INDEX] == 0.0)

    # CPU tensor passed to _per_token_log_probs_from_logits(impl=None) selects eager without calling torch_npu.
    recorded_calls.clear()
    cpu_out = _per_token_log_probs_from_logits(logits, labels.to(torch.int64), ignore_index=IGNORE_INDEX)
    assert recorded_calls == []
    torch.testing.assert_close(cpu_out, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("batch_shape", [(16,), (4, 8), (2, 3, 5)])
def test_token_logprobs_flash_attn_wrapper_contract(monkeypatch, batch_shape):
    """Verify shape flattening/restoration and sign negation in ``flash_attn`` row."""
    torch.manual_seed(0)
    vocab_size = 37
    logits = torch.randn(*batch_shape, vocab_size, dtype=torch.float32)
    labels = torch.randint(0, vocab_size, batch_shape, dtype=torch.int64)
    labels.view(-1)[::5] = IGNORE_INDEX

    recorded_calls: list[tuple[torch.Size, torch.Size, int]] = []

    def fake_fa_cross_entropy_loss(
        logits_2d: torch.Tensor,
        labels_1d: torch.Tensor,
        ignore_index: int = -100,
    ):
        recorded_calls.append((logits_2d.shape, labels_1d.shape, ignore_index))
        nll = F.cross_entropy(logits_2d, labels_1d, reduction="none", ignore_index=ignore_index)
        return nll, logits_2d.logsumexp(dim=-1)

    fake_ce_mod = SimpleNamespace(cross_entropy_loss=fake_fa_cross_entropy_loss)
    monkeypatch.setitem(sys.modules, "flash_attn.ops.triton.cross_entropy", fake_ce_mod)

    out = tlp_flash_attn.wrapper(logits, labels, ignore_index=IGNORE_INDEX)
    ref = tlp_eager.wrapper(logits, labels, ignore_index=IGNORE_INDEX)

    num_tokens = logits.numel() // vocab_size
    assert recorded_calls == [(torch.Size([num_tokens, vocab_size]), torch.Size([num_tokens]), IGNORE_INDEX)]
    assert out.shape == torch.Size(batch_shape)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
    assert torch.all(out[labels == IGNORE_INDEX] == 0.0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_token_logprobs_flash_attn_import_error_fallback_vs_strict(monkeypatch):
    """Automatic ``impl=None`` falls back to ``eager`` on ``flash_attn`` ``ImportError``, while explicit ``impl='flash_attn'`` raises."""
    import builtins

    import veomni.models.loss_utils.chunk_topk_distill as ctkd
    import veomni.ops.kernels.loss.token_logprobs.chunked as tlp_chunked

    monkeypatch.setattr(VeomniOp, "_intern", {})
    monkeypatch.setattr(cl, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(ctkd, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    # Simulate a GPU host where OP_REGISTRY discovers flash_attn via find_spec,
    # so VeomniOp("token_logprobs", "standard", None) selects the flash_attn row.
    monkeypatch.setattr("veomni.ops.registry.get_device_type", lambda: "cuda")
    monkeypatch.setattr(
        "veomni.ops.platform.requirement.GpuKernelRequirement.matches",
        lambda self: True,
    )
    monkeypatch.setattr(
        "veomni.ops.platform.requirement.GpuKernelRequirement.check",
        lambda self: None,
    )
    monkeypatch.setattr(
        "veomni.ops.registry.is_package_available",
        lambda pkg: True if pkg == "flash_attn.ops.triton.cross_entropy" else False,
    )

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "flash_attn.ops.triton.cross_entropy" or name.startswith("flash_attn."):
            raise ImportError("simulated flash_attn binary extension ABI mismatch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "flash_attn.ops.triton.cross_entropy", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Automatic selection (impl=None) on non-CPU device_type must fall back to eager.
    auto_op = resolve_token_logprobs_op(None, device_type="cuda")
    assert auto_op.impl == "eager"

    # Explicit selection (impl="flash_attn") must strictly raise ImportError.
    with pytest.raises(ImportError, match="ABI mismatch"):
        resolve_token_logprobs_op("flash_attn", device_type="cuda")

    # Verify both chunk_logprobs_function and chunk_topk_distill_function preserve
    # the eager fallback for impl=None (even when device_type is treated as non-CPU)
    # and raise ImportError for explicit impl="flash_attn".
    orig_resolve = tlp_chunked.resolve_token_logprobs_op
    monkeypatch.setattr(
        cl,
        "resolve_token_logprobs_op",
        lambda impl=None, *, device_type=None: orig_resolve(impl, device_type="cuda"),
    )
    monkeypatch.setattr(
        ctkd,
        "resolve_token_logprobs_op",
        lambda impl=None, *, device_type=None: orig_resolve(impl, device_type="cuda"),
    )

    torch.manual_seed(0)
    h = torch.randn(2, 8, 4, requires_grad=True)
    w = torch.randn(16, 4, requires_grad=True)
    labels = torch.randint(0, 16, (2, 8))
    teacher_logits = torch.randn(2, 8, 16).log_softmax(dim=-1)
    teacher_topk_lp, teacher_topk_ids = teacher_logits.topk(3, dim=-1)

    lp_auto, ent_auto = cl.chunk_logprobs_function(h, w, labels, chunk_size=4)
    lp_eager, ent_eager = cl.chunk_logprobs_function(h, w, labels, chunk_size=4, impl="eager")
    torch.testing.assert_close(lp_auto, lp_eager, rtol=0, atol=0)
    torch.testing.assert_close(ent_auto, ent_eager, rtol=0, atol=0)

    distill_auto = ctkd.chunk_topk_distill_function(h, w, labels, teacher_topk_ids, teacher_topk_lp, chunk_size=4)
    distill_eager = ctkd.chunk_topk_distill_function(
        h, w, labels, teacher_topk_ids, teacher_topk_lp, chunk_size=4, impl="eager"
    )
    for actual, expected in zip(distill_auto, distill_eager):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    with pytest.raises(ImportError, match="ABI mismatch"):
        cl.chunk_logprobs_function(h, w, labels, chunk_size=4, impl="flash_attn")

    with pytest.raises(ImportError, match="ABI mismatch"):
        ctkd.chunk_topk_distill_function(
            h, w, labels, teacher_topk_ids, teacher_topk_lp, chunk_size=4, impl="flash_attn"
        )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("use_shift_labels", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [1.0, 0.8])
def test_chunk_logprobs_topk_ids_forward_and_backward_contract(monkeypatch, use_shift_labels, dtype, temperature):
    """Verify ``topk_ids`` in ``chunk_logprobs_function`` and ``ForCausalLMLoss`` matches dense log-softmax gather."""
    from veomni.models.loss_utils.cross_entropy_loss import ForCausalLMLoss

    monkeypatch.setattr(cl, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    torch.manual_seed(13)

    B, L, H, V, K = 2, 7, 8, 29, 4
    hidden = torch.randn(B, L, H, dtype=dtype, requires_grad=True)
    weight = torch.randn(V, H, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, V, (B, L), dtype=torch.int64)
    labels[0, 3] = IGNORE_INDEX
    labels[1, -2] = IGNORE_INDEX
    shift_labels = torch.roll(labels, shifts=-1, dims=-1) if use_shift_labels else None
    if shift_labels is not None:
        shift_labels[0, 2] = IGNORE_INDEX
    topk_ids = torch.stack([torch.randperm(V)[:K] for _ in range(B * L)]).view(B, L, K)

    _, logits_out, aux = ForCausalLMLoss(
        labels=labels,
        shift_labels=shift_labels,
        hidden_states=hidden,
        weights=weight,
        return_log_probs=True,
        temperature=temperature,
        topk_ids=topk_ids,
        chunk_size=3,
        ignore_index=IGNORE_INDEX,
    )
    assert logits_out is None
    assert aux is not None and aux.topk_log_probs is not None
    assert aux.topk_log_probs.shape == (B, L, K)

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    if use_shift_labels:
        h_used, l_used, ids_used = hidden_ref, shift_labels, topk_ids
    else:
        h_used, l_used, ids_used = hidden_ref[:, :-1, :], labels[:, 1:], topk_ids[:, 1:, :]
    logits_ref = ((h_used @ weight_ref.t()) / temperature).float()
    logp_ref = logits_ref.log_softmax(dim=-1)
    valid = l_used != IGNORE_INDEX
    lp_ref = torch.where(valid, logp_ref.gather(-1, l_used.clamp(min=0).unsqueeze(-1)).squeeze(-1), 0.0)
    ent_ref = torch.where(valid, -(logp_ref.exp() * logp_ref).sum(dim=-1), 0.0)
    topk_lp_ref = torch.where(valid.unsqueeze(-1), logp_ref.gather(-1, ids_used), 0.0)
    if not use_shift_labels:
        lp_ref = F.pad(lp_ref, (0, 1), value=0.0)
        ent_ref = F.pad(ent_ref, (0, 1), value=0.0)
        topk_lp_ref = F.pad(topk_lp_ref, (0, 0, 0, 1), value=0.0)

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(aux.log_probs, lp_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(aux.entropy, ent_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(aux.topk_log_probs, topk_lp_ref, rtol=rtol, atol=atol)

    w_lp = torch.randn_like(aux.log_probs)
    w_ent = torch.randn_like(aux.entropy)
    w_topk = torch.randn_like(aux.topk_log_probs)
    ((aux.log_probs * w_lp).sum() + (aux.entropy * w_ent).sum() + (aux.topk_log_probs * w_topk).sum()).backward()
    ((lp_ref * w_lp).sum() + (ent_ref * w_ent).sum() + (topk_lp_ref * w_topk).sum()).backward()

    torch.testing.assert_close(hidden.grad.float(), hidden_ref.grad.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(weight.grad.float(), weight_ref.grad.float(), rtol=rtol, atol=atol)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_chunk_logprobs_topk_edge_contracts_and_pytree(monkeypatch):
    """Cover pytree leaf exposure, topk-only backward, ignored negative sentinels, duplicates, 2D/SP, and validation."""
    from veomni.models.loss_utils.cross_entropy_loss import ForCausalLMLoss
    from veomni.utils.model_outputs import CausalLMOutputWithLogProbs

    monkeypatch.setattr(cl, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    torch.manual_seed(29)

    # 1. Packed 2D inputs (needs_trailing_pad=True) + duplicate topk_ids + negative sentinels on IGNORE_INDEX rows,
    #    and backward through topk_log_probs ONLY (dlog_probs=None, dentropy=None).
    T, H, V, K = 9, 8, 23, 4
    hidden = torch.randn(T, H, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, V, (T,), dtype=torch.int64)
    labels[4] = IGNORE_INDEX
    topk_ids = torch.randint(0, V, (T, K), dtype=torch.int64)
    topk_ids[2, 1] = topk_ids[2, 0]  # duplicate index in row -> scatter_add_ accumulates
    topk_ids[4] = torch.tensor([-1, IGNORE_INDEX, -5, -99])  # negative sentinels at ignored target position

    _, _, aux_2d = ForCausalLMLoss(
        labels=labels,
        hidden_states=hidden,
        weights=weight,
        return_log_probs=True,
        topk_ids=topk_ids,
        chunk_size=3,
        ignore_index=IGNORE_INDEX,
    )
    out_obj = CausalLMOutputWithLogProbs(fused_linear_aux=aux_2d)
    leaves = torch.utils._pytree.tree_leaves(out_obj)
    assert any(leaf is aux_2d.topk_log_probs for leaf in leaves)

    w_topk = torch.randn_like(aux_2d.topk_log_probs)
    (aux_2d.topk_log_probs * w_topk).sum().backward()

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    logp_ref = (hidden_ref[:-1] @ weight_ref.t()).log_softmax(dim=-1)
    valid = labels[1:] != IGNORE_INDEX
    safe_ids = torch.where(valid.unsqueeze(-1), topk_ids[1:], 0)
    topk_ref = F.pad(torch.where(valid.unsqueeze(-1), logp_ref.gather(-1, safe_ids), 0.0), (0, 0, 0, 1), value=0.0)
    torch.testing.assert_close(aux_2d.topk_log_probs, topk_ref, rtol=1e-5, atol=1e-5)
    (topk_ref * w_topk).sum().backward()
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-5, atol=1e-5)

    # 2. labels=None with shift_labels provided + sp_enabled=True (no internal shift or trailing pad).
    monkeypatch.setattr(cl, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=True))
    _, _, aux_sp = ForCausalLMLoss(
        labels=None,
        shift_labels=labels,
        hidden_states=hidden.detach(),
        weights=weight.detach(),
        return_log_probs=True,
        topk_ids=topk_ids,
        chunk_size=4,
    )
    assert aux_sp.topk_log_probs.shape == (T, K)
    assert torch.all(aux_sp.topk_log_probs[4] == 0.0)

    # 3. Validation guards: mutual exclusion with teacher_topk_ids and mismatched shapes / K <= 0.
    with pytest.raises(ValueError, match="cannot be combined"):
        ForCausalLMLoss(
            labels=labels,
            hidden_states=hidden.detach(),
            weights=weight.detach(),
            return_log_probs=True,
            topk_ids=topk_ids,
            teacher_topk_ids=topk_ids,
            teacher_topk_log_probs=torch.zeros_like(topk_ids, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="leading shape"):
        cl.chunk_logprobs_function(
            hidden.detach(), weight.detach(), labels, shift_labels=labels, topk_ids=topk_ids[:-1]
        )
    with pytest.raises(ValueError, match="positive"):
        cl.chunk_logprobs_function(
            hidden.detach(), weight.detach(), labels, shift_labels=labels, topk_ids=topk_ids[:, :0]
        )


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend NPU required")
def test_npu_token_logprobs_matches_eager():
    """On Ascend NPU, default ``token_logprobs`` resolves to ``torch_npu`` and matches ``eager``."""
    assert OP_REGISTRY.default_impl("token_logprobs", "standard", device="npu") == "torch_npu"
    device = torch.device("npu")
    torch.manual_seed(42)

    tokens, vocab_size = 512, 4096
    logits = torch.randn(tokens, vocab_size, device=device, dtype=torch.float32)
    labels = torch.randint(0, vocab_size, (tokens,), device=device, dtype=torch.int64)
    labels[::11] = IGNORE_INDEX

    ref = tlp_eager.wrapper(logits, labels, ignore_index=IGNORE_INDEX)
    out = _per_token_log_probs_from_logits(logits, labels, ignore_index=IGNORE_INDEX)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert torch.all(out[labels == IGNORE_INDEX] == 0.0)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend NPU required")
def test_npu_chunk_logprobs_function_forward_and_backward(monkeypatch):
    """End-to-end ``chunk_logprobs_function`` forward and backward on Ascend NPU."""
    monkeypatch.setattr(cl, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    device = torch.device("npu")
    torch.manual_seed(7)

    B, L, H, V = 2, 32, 64, 512
    temperature = 1.5
    hidden = torch.randn(B, L, H, device=device, dtype=torch.float32, requires_grad=True)
    weights = torch.randn(V, H, device=device, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, V, (B, L), device=device, dtype=torch.int64)
    labels[0, ::5] = IGNORE_INDEX
    labels[1, 0] = IGNORE_INDEX

    log_probs, entropy = chunk_logprobs_function(
        hidden, weights, labels, chunk_size=16, ignore_index=IGNORE_INDEX, temperature=temperature
    )
    loss = log_probs.sum() + 0.1 * entropy.sum()
    loss.backward()

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weights_ref = weights.detach().clone().requires_grad_(True)
    log_probs_ref, entropy_ref = chunk_logprobs_function(
        hidden_ref,
        weights_ref,
        labels,
        chunk_size=16,
        ignore_index=IGNORE_INDEX,
        temperature=temperature,
        impl="eager",
    )
    (log_probs_ref.sum() + 0.1 * entropy_ref.sum()).backward()

    torch.testing.assert_close(log_probs, log_probs_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, entropy_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(weights.grad, weights_ref.grad, rtol=1e-4, atol=1e-4)


def _measure_ms(fn, warmup: int, iters: int) -> float:
    """Measure average execution latency of ``fn`` in milliseconds after warmup."""
    for _ in range(warmup):
        _ = fn()
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    torch.npu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def run_npu_cross_entropy_fast_path_check(
    tokens: int = 2048,
    vocab_size: int = 151936,
    dtype_str: str = "float32",
    ignore_index: int = -100,
    ignore_ratio: float = 0.01,
    warmup: int = 20,
    iters: int = 100,
    seed: int = 2026,
) -> None:
    """Run precision and latency benchmark comparing NPU CE fast path against eager log_softmax."""
    if not IS_NPU_AVAILABLE:
        raise RuntimeError("NPU is not available in current environment.")

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from {list(dtype_map.keys())}.")

    torch.manual_seed(seed)
    device = torch.device("npu")
    dtype = dtype_map[dtype_str]

    logits = torch.randn(tokens, vocab_size, device=device, dtype=dtype)
    labels = torch.randint(0, vocab_size, (tokens,), device=device, dtype=torch.int64)
    if ignore_ratio > 0:
        ignore_mask = torch.rand(tokens, device=device) < ignore_ratio
        labels = torch.where(ignore_mask, torch.full_like(labels, ignore_index), labels)

    with torch.no_grad():
        old_out = tlp_eager.wrapper(logits, labels, ignore_index=ignore_index)
        npu_out = tlp_torch_npu.wrapper(logits, labels, ignore_index=ignore_index)
        dispatch_out = _per_token_log_probs_from_logits(logits, labels, ignore_index=ignore_index)
    torch.npu.synchronize()

    diff_npu = (old_out.float() - npu_out.float()).abs()
    diff_dispatch = (npu_out.float() - dispatch_out.float()).abs()
    ign_mask = labels == ignore_index
    ign_old_abs = old_out[ign_mask].float().abs().max().item() if ign_mask.any() else 0.0
    ign_new_abs = npu_out[ign_mask].float().abs().max().item() if ign_mask.any() else 0.0

    old_ms = _measure_ms(
        lambda: tlp_eager.wrapper(logits, labels, ignore_index=ignore_index),
        warmup=warmup,
        iters=iters,
    )
    new_ms = _measure_ms(
        lambda: tlp_torch_npu.wrapper(logits, labels, ignore_index=ignore_index),
        warmup=warmup,
        iters=iters,
    )
    speedup = old_ms / new_ms if new_ms > 0 else float("inf")

    print("=== NPU Cross-Entropy Fast Path Check ===")
    print(
        f"shape=({tokens}, {vocab_size}), dtype={dtype_str}, "
        f"ignore_index={ignore_index}, ignore_ratio={ignore_ratio}, warmup={warmup}, iters={iters}"
    )
    print("--- Precision ---")
    print(f"max_abs_diff(old vs npu_fast):      {diff_npu.max().item():.8e}")
    print(f"mean_abs_diff(old vs npu_fast):     {diff_npu.mean().item():.8e}")
    print(f"max_abs_diff(npu_fast vs dispatch): {diff_dispatch.max().item():.8e}")
    print(f"ignored_count:                      {int(ign_mask.sum().item())}")
    print(f"ignored_old_max_abs:                {ign_old_abs:.8e}")
    print(f"ignored_new_max_abs:                {ign_new_abs:.8e}")
    print("--- Performance ---")
    print(f"old (log_softmax + gather): {old_ms:.3f} ms/iter")
    print(f"new (npu_cross_entropy):    {new_ms:.3f} ms/iter")
    print(f"speedup (old / new):        {speedup:.2f}x")


def main() -> None:
    """CLI entrypoint for NPU cross-entropy fast-path precision and speed check."""
    parser = argparse.ArgumentParser(description="Check NPU CE fast path precision and speed for chunk_logprobs.")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument("--ignore-ratio", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    run_npu_cross_entropy_fast_path_check(
        tokens=args.tokens,
        vocab_size=args.vocab_size,
        dtype_str=args.dtype,
        ignore_index=args.ignore_index,
        ignore_ratio=args.ignore_ratio,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
