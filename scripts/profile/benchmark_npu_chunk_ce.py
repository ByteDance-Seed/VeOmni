"""Compare full/compact eager and native CE with first-order grads.

Run from the repository root on a native NPU. This measures synchronous chunk
math, not training throughput, distributed communication or activation offload.
"""

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from veomni.ops.kernels.loss.cross_entropy_loss.standard.eager import cross_entropy_from_logits
from veomni.ops.kernels.loss.cross_entropy_loss.standard.npu import _NPUCrossEntropy, fused_linear_ce


def main():
    """Measure chunk loss and gradients, then export timings, memory and parity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4097)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=32768)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cann-version", default="unknown", help="Record the installed CANN version")
    parser.add_argument("--valid-fractions", type=float, nargs="+", default=[0.25, 1.0])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["full_eager", "compact_eager", "compact_native", "full_native"],
        default=["full_eager", "compact_eager", "compact_native"],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if "full_native" in args.modes and any(fraction != 1.0 for fraction in args.valid_fractions):
        parser.error("full_native requires --valid-fractions 1.0 (native CE cannot consume ignored rows)")
    torch.manual_seed(127)
    hidden = torch.randn(2, args.sequence, args.hidden, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(args.vocab, args.hidden, device="npu", dtype=torch.bfloat16)
    results = []
    for fraction in args.valid_fractions:
        labels = torch.randint(0, args.vocab, hidden.shape[:-1], device="npu")
        labels[torch.rand(labels.shape, device="npu") >= fraction] = -100
        denominator = (labels != -100).sum().clamp_min(1)

        def loss_func(x, w, targets, mode, denominator=denominator):
            """Evaluate one chunk with the selected path and shared denominator."""
            x = x.reshape(-1, x.shape[-1])
            targets = targets.reshape(-1)
            if mode == "compact_native":
                return fused_linear_ce(x, w, targets, denominator, -100)
            if mode == "full_native":
                # The B x 1 final chunk can retain a non-contiguous target
                # stride after reshape; native CE requires contiguous targets.
                loss, _ = _NPUCrossEntropy.apply(F.linear(x, w).float(), targets.contiguous())
                return loss / denominator
            if mode == "compact_eager":
                valid = targets != -100
                x, targets = x[valid], targets[valid]
            return cross_entropy_from_logits(
                F.linear(x, w).float(), targets, ignore_index=-100, num_items_in_batch=denominator
            )

        def run(mode, labels=labels, loss_func=loss_func):
            """Compute batch loss and hidden/weight gradients across all chunks."""
            total = torch.zeros((), device="npu")
            grad_hidden, grad_weight = torch.empty_like(hidden), torch.zeros_like(weight)
            chunks = zip(
                hidden.split(args.chunk_size, dim=1),
                labels.split(args.chunk_size, dim=1),
                grad_hidden.split(args.chunk_size, dim=1),
                strict=True,
            )
            for x, targets, grad_chunk in chunks:
                grads, loss = torch.func.grad_and_value(loss_func, argnums=(0, 1))(x, weight, targets, mode)
                grad_chunk.copy_(grads[0])
                grad_weight.add_(grads[1])
                total.add_(loss)
            return total, grad_hidden, grad_weight

        reference = tuple(value.detach() for value in run("full_eager"))
        torch.npu.synchronize()
        for mode in args.modes:
            for _ in range(args.warmup):
                run(mode)
            torch.npu.synchronize()
            base_memory = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            durations = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                actual = run(mode)
                torch.npu.synchronize()
                durations.append((time.perf_counter() - start) * 1000)
                del actual
            peak_extra = torch.npu.max_memory_allocated() - base_memory
            actual = run(mode)
            errors = {}
            for name, value, target in zip(("loss", "grad_hidden", "grad_weight"), actual, reference, strict=True):
                difference = (value.float() - target.float()).abs()
                errors[name + "_max_abs"] = difference.max().item()
                errors[name + "_relative_l2"] = (difference.norm() / target.float().norm().clamp_min(1e-12)).item()
            results.append(
                {
                    "mode": mode,
                    "requested_valid_fraction": fraction,
                    "actual_valid_fraction": denominator.item() / labels.numel(),
                    "median_ms": statistics.median(durations),
                    "min_ms": min(durations),
                    "total_measured_ms": sum(durations),
                    "durations_ms": durations,
                    "ce_equivalent_billion_tokens_per_day": (
                        labels.numel() * args.iterations * 86400 / (sum(durations) / 1000) / 1e9
                    ),
                    "extra_peak_allocated_mib": peak_extra / 1024**2,
                    **errors,
                }
            )
            del actual
        del reference
    import torch_npu

    report = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(),
        "dtype": "bfloat16",
        "batch": 2,
        "execution": "direct torch.func chunk math; no registry or offload adapter",
        **vars(args),
        "results": results,
    }
    with open(args.output, "w") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
