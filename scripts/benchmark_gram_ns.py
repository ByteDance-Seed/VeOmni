#!/usr/bin/env python3
"""Microbenchmark Muon Newton-Schulz backends used by VeOmni.

Compares:
  1) veomni batched_newton_schulz (current default)
  2) veomni batched_gram_newton_schulz (pure torch Gram-NS)
  3) Dao-AILab GramNewtonSchulz torch / kernels (if installed)

Example:
  PYTHONPATH=/home/tiger/.local/lib/python3.11/site-packages \\
  python scripts/benchmark_gram_ns.py --batch 8 --rows 512 --cols 2048
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List

import torch

from veomni.optim.muon import (
    DEFAULT_NS_COEFFICIENTS,
    batched_gram_newton_schulz,
    batched_newton_schulz,
    run_newton_schulz,
)


def _bench(fn: Callable[[], None], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = "cuda"
    x = torch.randn(args.batch, args.rows, args.cols, device=device, dtype=dtype)
    print(
        f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability()} "
        f"shape=({args.batch},{args.rows},{args.cols}) dtype={args.dtype}"
    )

    variants = []

    def std():
        batched_newton_schulz(x, DEFAULT_NS_COEFFICIENTS, 5, eps=1e-7)

    variants.append(("veomni_standard_ns", std))

    def gram_torch():
        batched_gram_newton_schulz(
            x,
            ns_coefficients=DEFAULT_NS_COEFFICIENTS,
            ns_steps=5,
            eps=1e-7,
            reset_iterations=(2,),
            compute_dtype=torch.float16,
        )

    variants.append(("veomni_gram_ns_torch", gram_torch))

    def gram_dispatch_no_kernel():
        run_newton_schulz(
            x,
            ns_coefficients=DEFAULT_NS_COEFFICIENTS,
            ns_steps=5,
            ns_implementation="gram",
            gram_ns_reset_iterations=(2,),
        )

    variants.append(("veomni_run_gram_torch", gram_dispatch_no_kernel))

    try:
        import gram_newton_schulz  # noqa: F401

        def gram_pkg_torch():
            run_newton_schulz(
                x,
                ns_coefficients=DEFAULT_NS_COEFFICIENTS,
                ns_steps=5,
                ns_implementation="gram",
                gram_ns_reset_iterations=(2,),
            )

        # pure package object for kernel path
        from gram_newton_schulz import GramNewtonSchulz, YOU_COEFFICIENTS  # noqa: I001

        pkg_torch = GramNewtonSchulz(
            ns_use_kernels=False,
            ns_coefficients=[[*DEFAULT_NS_COEFFICIENTS]] * 5,
            gram_newton_schulz_reset_iterations=[2],
            compile_kwargs=None,
        )
        pkg_kernel = GramNewtonSchulz(
            ns_use_kernels=True,
            ns_coefficients=[[*DEFAULT_NS_COEFFICIENTS]] * 5,
            gram_newton_schulz_reset_iterations=[2],
            compile_kwargs=None,
        )
        pkg_you_kernel = GramNewtonSchulz(
            ns_use_kernels=True,
            ns_coefficients=YOU_COEFFICIENTS,
            gram_newton_schulz_reset_iterations=[2],
            compile_kwargs=None,
        )

        variants.append(("package_gram_torch", lambda: pkg_torch(x)))
        variants.append(("package_gram_kernels", lambda: pkg_kernel(x)))
        variants.append(("package_gram_you_kernels", lambda: pkg_you_kernel(x)))

        def veomni_kernel():
            run_newton_schulz(
                x,
                ns_coefficients=DEFAULT_NS_COEFFICIENTS,
                ns_steps=5,
                ns_implementation="gram_quack",
                gram_ns_reset_iterations=(2,),
            )

        variants.append(("veomni_run_gram_kernels", veomni_kernel))
    except Exception as exc:
        print(f"[skip package/kernels] {type(exc).__name__}: {exc}")

    # correctness peek
    with torch.no_grad():
        y_std = batched_newton_schulz(x, DEFAULT_NS_COEFFICIENTS, 5)
        y_gram = batched_gram_newton_schulz(
            x, DEFAULT_NS_COEFFICIENTS, 5, reset_iterations=(), compute_dtype=torch.float32
        )
        # compare on same device float
        diff = (y_std.float() - y_gram.float()).abs().max().item()
        print(f"max|standard - gram(no-restart,fp32-compute)| = {diff:.6g}")

    print(f"{'variant':32s}  median_ms")
    for name, fn in variants:
        # first call may compile kernels
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = _bench(fn, warmup=args.warmup, repeats=args.repeats)
        print(f"{name:32s}  {ms:8.3f}")


if __name__ == "__main__":
    main()
