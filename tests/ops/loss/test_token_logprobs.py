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
