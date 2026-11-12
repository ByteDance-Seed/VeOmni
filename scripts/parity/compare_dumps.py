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
"""Diff two ``dump_forward.py`` files and say where, not just whether, they differ.

A single max-abs number is the wrong report here. fp32 accumulation over a few
hundred terms disagrees in the last couple of bits between any two orderings, so the
question is never "are they equal" but "is the gap the arithmetic's or the model's".
Relative error against the tensor's own scale separates those, and printing where the
worst element sits is what makes a real failure debuggable.

    python scripts/parity/compare_dumps.py /tmp/dg_parity/veomni.pt /tmp/dg_parity/nemo.pt

Gradients are matched through ``VEOMNI_TO_NEMO``, a mechanical rewrite of the parameter
names. Anything it fails to match is listed rather than dropped: a rename on either
side would otherwise quietly shrink the comparison instead of failing it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


# VeOmni names the shared trunk under the decoder and the encoder aliases into it;
# NeMo has one trunk and no such split. The MoE differs in wording only -- except for
# `per_expert_scale`, which NeMo does not have at all because it folds the scale into
# the expert down-projection on load. That is a real parameterization difference, and
# it is why the gradient comparison needs a checkpoint written with
# `--unit-expert-scale`: at 1 the fold is the identity and the two tensors coincide.
VEOMNI_TO_NEMO = (
    ("model.decoder.", "model."),
    (".router.proj.weight", ".moe.gate.proj.weight"),
    (".router.scale", ".moe.gate.scale"),
    (".experts.gate_up_proj", ".moe.experts.gate_and_up_projs"),
    (".experts.down_proj", ".moe.experts.down_projs"),
)

NO_COUNTERPART = ("router.per_expert_scale",)

# NeMo's state-dict adapter stores the stacked expert weights as ``(experts, out, in)``;
# VeOmni keeps the HF ``(experts, in, out)``. A storage convention, not a different
# function -- the forward agrees to fp32 noise either way -- but the gradients are
# indexed the same way as the weights, so they have to be brought into one frame before
# being subtracted. Transposing here rather than accepting a "permutation of each other"
# verdict is what keeps a genuinely scrambled expert weight detectable.
TRANSPOSED_EXPERTS = (".experts.gate_up_proj", ".experts.down_proj")

# At the 26B checkpoint `dump_forward.py` summarises the stacked expert gradients
# rather than writing 91 GiB of them. A norm along one inner axis on this side is a
# norm along the other on NeMo's, so the parts are paired through the transpose the
# same way the whole tensors are.
SIGNATURE_PARTS = ("scalars", "norm_rows", "norm_cols")
TRANSPOSED_SIGNATURE = {"scalars": "scalars", "norm_rows": "norm_cols", "norm_cols": "norm_rows"}


def translate(name: str) -> str:
    for before, after in VEOMNI_TO_NEMO:
        name = name.replace(before, after)
    return name


def split_gradient(name: str) -> tuple[str, str | None]:
    """``grad::<param>[::<part>]`` -> ``(param, part)``, ``part`` None for whole tensors."""
    body = name[len("grad::") :]
    head, _, tail = body.rpartition("::")
    return (head, tail) if head and tail in SIGNATURE_PARTS else (body, None)


# Module paths for the `sub::` traces. Same two renames as the parameters, minus the
# tensor suffix: VeOmni hangs the MoE branch off the block directly, NeMo puts it under
# a `moe` container and calls the router a gate.
VEOMNI_TO_NEMO_MODULES = (
    ("experts", "moe.experts"),
    ("router.proj", "moe.gate.proj"),
    ("router", "moe.gate"),
)


def counterpart(name: str) -> str:
    """The name this tensor should be compared against in the other stack's dump."""
    if name.startswith("sub::"):
        path = name[len("sub::") :]
        for before, after in VEOMNI_TO_NEMO_MODULES:
            if path == before or path.startswith(f"{before}."):
                return f"sub::{after}{path[len(before) :]}"
        return name
    if not name.startswith("grad::"):
        return name
    param, part = split_gradient(name)
    if part is None:
        return f"grad::{translate(param)}"
    if param.endswith(TRANSPOSED_EXPERTS):
        part = TRANSPOSED_SIGNATURE[part]
    return f"grad::{translate(param)}::{part}"


def to_common_frame(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Put a VeOmni tensor in the layout its NeMo counterpart is stored in."""
    if name.endswith(TRANSPOSED_EXPERTS) and tensor.dim() == 3:
        return tensor.transpose(1, 2)
    return tensor


def relative_error(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float, tuple[int, ...]]:
    """``(max_abs, rel_norm, argmax_index)``.

    ``rel_norm`` is ``||a - b|| / max(||a||, ||b||)``: scale-free, so one threshold
    reads the same on a tiny model's 1e-8 gradients and the 26B model's, where the
    absolute numbers are orders of magnitude larger without anything being wrong.
    The argmax index is kept because knowing *where* the worst element sits is what
    makes a real failure debuggable.
    """
    a, b = left.double(), right.double()
    difference = (a - b).abs()
    scale = max(float(a.norm()), float(b.norm()))
    flat = int(difference.argmax())
    return (
        float(difference.max()),
        float(difference.norm()) / scale if scale > 0 else 0.0,
        tuple(torch.unravel_index(torch.tensor(flat), difference.shape)),
    )


def elementwise(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    """Per-element diff statistics, for when the norm ratio is the wrong summary.

    ``rel_norm`` answers "how far apart are these two tensors"; these answer "how far
    apart is a typical element, and the worst one". They disagree usefully: a single
    bad element is diluted by the norm's denominator but dominates ``max``, and a
    uniform small bias shows up in ``mean`` while the norm ratio reports it as one
    unremarkable number.

    The relative difference is ``|a-b| / max(|a|,|b|)``, symmetric like the norm
    ratio's denominator, zero where both sides are, and needing no epsilon. It is
    still dominated by near-zero elements, where a difference far below anything
    meaningful divides by something equally small -- so the same pair of statistics
    is reported again over the elements carrying at least 1% of the tensor's RMS,
    rather than that population being quietly smoothed away.
    """
    a, b = left.double(), right.double()
    difference = (a - b).abs()
    scale = torch.maximum(a.abs(), b.abs())
    relative = torch.where(scale > 0, difference / scale, torch.zeros_like(difference))
    carrying = scale >= 0.01 * float((b * b).mean().sqrt())
    return {
        "mean_abs": float(difference.mean()),
        "max_abs": float(difference.max()),
        "mean_rel": float(relative.mean()),
        "max_rel": float(relative.max()),
        "mean_rel_carrying": float(relative[carrying].mean()) if carrying.any() else float("nan"),
        "max_rel_carrying": float(relative[carrying].max()) if carrying.any() else float("nan"),
        "carrying": float(carrying.double().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--tol", type=float, default=1e-5, help="max absolute difference treated as agreement")
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-5,
        help="norm-relative difference treated as agreement; either threshold passing is enough",
    )
    parser.add_argument("--quiet", action="store_true", help="print only the tensors that differ")
    parser.add_argument(
        "--elementwise",
        action="store_true",
        help="report mean/max of the per-element absolute and relative difference instead of the norm ratio",
    )
    args = parser.parse_args()

    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    if left["seed"] != right["seed"]:
        raise SystemExit(f"seeds differ ({left['seed']} vs {right['seed']}); the two runs were not fed one input")

    left_tensors, right_tensors = left["tensors"], right["tensors"]
    activations = sorted(n for n in left_tensors if not n.startswith("grad::"))
    gradients = sorted(n for n in left_tensors if n.startswith("grad::"))

    for side in (left, right):
        if side.get("from"):
            print(f"{side['stack']:<7} {side['from']['argv']}")

    print(f"{left['stack']} vs {right['stack']}   (abs tol {args.tol:g}, rel tol {args.rel_tol:g})")
    if args.elementwise:
        header = (
            f"{'tensor':<40} {'mean|Δ|':>10} {'max|Δ|':>10} {'mean rel':>10} "
            f"{'max rel':>10} {'mean rel*':>10} {'max rel*':>10} {'*share':>8}"
        )
    else:
        header = f"{'tensor':<58} {'shape':>16} {'max|Δ|':>11} {'rel norm':>10}  verdict"

    failures, compared, skipped, worst = 0, 0, [], 0.0
    matched_right: set[str] = set()
    printed_header = False
    for left_name in activations + gradients:
        right_name = counterpart(left_name)
        if right_name not in right_tensors:
            base = split_gradient(left_name)[0] if left_name.startswith("grad::") else left_name
            reason = "folded into the expert down-projection" if base.endswith(NO_COUNTERPART) else "UNMATCHED"
            skipped.append((left_name, reason))
            failures += reason == "UNMATCHED"
            continue
        matched_right.add(right_name)

        a = to_common_frame(left_name, left_tensors[left_name])
        b = right_tensors[right_name]
        max_abs, rel_norm, index = (
            relative_error(a, b) if a.shape == b.shape else (float("inf"), float("inf"), tuple(b.shape))
        )
        ok = a.shape == b.shape and (max_abs <= args.tol or rel_norm <= args.rel_tol)
        failures += not ok
        compared += 1
        worst = max(worst, rel_norm if a.shape == b.shape else float("inf"))
        if args.quiet and ok:
            continue
        if not printed_header:
            print(header)
            printed_header = True
        if args.elementwise and a.shape == b.shape:
            s = elementwise(a, b)
            print(
                f"{left_name:<40} {s['mean_abs']:>10.3e} {s['max_abs']:>10.3e} "
                f"{s['mean_rel']:>10.3e} {s['max_rel']:>10.3e} "
                f"{s['mean_rel_carrying']:>10.3e} {s['max_rel_carrying']:>10.3e} {s['carrying']:>7.1%}"
            )
            continue
        verdict = "ok" if ok else (f"SHAPE {tuple(b.shape)}" if a.shape != b.shape else f"DIFFERS at {index}")
        print(f"{left_name:<58} {str(tuple(a.shape)):>16} {max_abs:>11.3e} {rel_norm:>10.3e}  {verdict}")

    # A tensor only the right stack dumped is as much a mismatch as one only the left
    # stack dumped: `dump_forward.py` omits a gradient that was never produced, so a
    # dropped backward in one stack would otherwise pass with exit status 0.
    for right_name in sorted(set(right_tensors) - matched_right):
        base = split_gradient(right_name)[0] if right_name.startswith("grad::") else right_name
        reason = "folded into the expert down-projection" if base.endswith(NO_COUNTERPART) else "UNMATCHED"
        skipped.append((right_name, reason))
        failures += reason == "UNMATCHED"

    if args.elementwise:
        print("\n* over elements at or above 1% of the tensor's RMS; `*share` is how many those are.")
    print(f"\ncompared {compared} tensors, {failures} outside tolerance, worst rel norm {worst:.3e}")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
