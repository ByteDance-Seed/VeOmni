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

"""Manual check for the NPU fused CE fast path.

Run with:
    python tests/ops/test_npu_chunk_logprobs.py --device npu

This script compares the NPU fused CE helper against the old
log_softmax+gather fallback. It is kept as a manual check because reliable
timing requires an NPU runtime and explicit synchronization.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from veomni.ops.kernels.cross_entropy.chunk_logprobs import (  # noqa: E402
    _NPU_CE_AVAILABLE,
    _per_token_log_probs_from_logits_npu,
)


def old_log_softmax_gather(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    # This is the pre-optimization fallback path: it materializes a full
    # [tokens, vocab] log-prob tensor, then gathers target-token logprobs.
    mask = labels != ignore_index
    safe_labels = labels.clamp(min=0).unsqueeze(-1)
    log_probs = logits.log_softmax(dim=-1).gather(-1, safe_labels).squeeze(-1)
    return torch.where(mask, log_probs, torch.zeros_like(log_probs))


def run_npu_cross_entropy_fast_path_check() -> None:
    """Compare the NPU fused CE path with the original PyTorch fallback."""
    parser = argparse.ArgumentParser(description="Compare NPU CE fast path against log_softmax+gather.")
    parser.add_argument("--device", default="npu", choices=["npu", "cpu"], help="Device used for generated tensors.")
    parser.add_argument("--tokens", type=int, default=2048, help="Number of token rows, i.e. N in [N, V].")
    parser.add_argument("--vocab", type=int, default=151936, help="Vocabulary size, i.e. V in [N, V].")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Logits dtype.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations.")
    parser.add_argument("--ignore-index", type=int, default=-100, help="Ignored label value.")
    parser.add_argument("--ignore-ratio", type=float, default=0.01, help="Fraction of labels set to ignore-index.")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    if args.device == "npu" and (not _NPU_CE_AVAILABLE or not hasattr(torch, "npu") or not torch.npu.is_available()):
        raise RuntimeError(
            "NPU manual check requested, but torch_npu.npu_cross_entropy_loss or an NPU device is unavailable."
        )

    def synchronize() -> None:
        if device.type == "npu":
            torch.npu.synchronize()

    def new_npu_fused_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return _per_token_log_probs_from_logits_npu(logits, labels, args.ignore_index)

    torch.manual_seed(2026)
    logits = torch.randn(args.tokens, args.vocab, device=device, dtype=dtype)
    labels = torch.randint(0, args.vocab, (args.tokens,), device=device, dtype=torch.long)
    if args.ignore_ratio > 0:
        ignore_mask = torch.rand(args.tokens, device=device) < args.ignore_ratio
        labels = labels.masked_fill(ignore_mask, args.ignore_index)

    # Precision check uses fp32 comparison. For fp16/bf16 inputs, the two kernels
    # may differ slightly because their internal reductions are implemented
    # differently, so report both absolute and relative error.
    with torch.no_grad():
        old_out = old_log_softmax_gather(logits, labels, args.ignore_index)
        if device.type == "npu":
            new_out = new_npu_fused_ce(logits, labels)
        else:
            print("[WARN] CPU has no torch_npu fast path; comparing the fallback against itself.")
            new_out = old_log_softmax_gather(logits, labels, args.ignore_index)
        synchronize()

        old_f32 = old_out.float()
        new_f32 = new_out.float()
        abs_diff = (new_f32 - old_f32).abs()
        rel_diff = abs_diff / old_f32.abs().clamp_min(1e-12)
        ignored = labels == args.ignore_index

    def time_path(name: str, fn) -> float:
        with torch.no_grad():
            for _ in range(args.warmup):
                fn(logits, labels)
            synchronize()

            start = time.perf_counter()
            for _ in range(args.iters):
                fn(logits, labels)
            synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / args.iters
            print(f"[TIME] {name}: {elapsed_ms:.3f} ms/iter")
            return elapsed_ms

    print("========== NPU CE Fast Path Check ==========")
    print(f"device={device}, dtype={args.dtype}, tokens={args.tokens}, vocab={args.vocab}")
    print(f"ignore_index={args.ignore_index}, ignore_ratio={args.ignore_ratio}, ignored_tokens={ignored.sum().item()}")
    print(f"[PRECISION] max_abs_diff={abs_diff.max().item():.8e}")
    print(f"[PRECISION] mean_abs_diff={abs_diff.mean().item():.8e}")
    print(f"[PRECISION] max_rel_diff={rel_diff.max().item():.8e}")
    print(f"[PRECISION] mean_rel_diff={rel_diff.mean().item():.8e}")
    if ignored.any():
        print(f"[PRECISION] ignored_old_max_abs={old_f32[ignored].abs().max().item():.8e}")
        print(f"[PRECISION] ignored_new_max_abs={new_f32[ignored].abs().max().item():.8e}")

    old_ms = time_path(
        "old log_softmax+gather",
        lambda logits, labels: old_log_softmax_gather(logits, labels, args.ignore_index),
    )
    if device.type == "npu":
        new_ms = time_path("new torch_npu.npu_cross_entropy_loss", new_npu_fused_ce)
        print(f"[SPEEDUP] old/new = {old_ms / new_ms:.3f}x")
    else:
        print("[SPEEDUP] skipped because CPU cannot run the NPU fused CE kernel.")


if __name__ == "__main__":
    run_npu_cross_entropy_fast_path_check()
