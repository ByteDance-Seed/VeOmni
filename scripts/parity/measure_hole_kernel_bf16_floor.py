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
"""How far the two decoder attention paths sit from exact, and from each other, in bf16.

The CPU/fp32 parity steps establish that VeOmni's DiffusionGemma computes the same
function NeMo's does. They say nothing about training, which runs bf16 on a GPU
through a Triton kernel that never appears in those runs. This measures the floor
that sits under any bf16 comparison, so a difference observed between two training
runs can be read against the size of the arithmetic noise rather than assumed to be
semantic.

Three things are measured at the shipped geometry, all against an **fp32** oracle:

* the Triton hole kernel in bf16,
* ``F.scaled_dot_product_attention`` over a materialized hole mask in bf16 -- the
  fallback the non-kernel path actually runs,
* and the two against each other.

The oracle runs in fp32 on purpose. ``hole_attention_reference`` multiplies in
whatever dtype it is handed, so running it in bf16 makes it err at 1e-2 and would
flatter whichever path is being scored against it -- crediting a kernel with beating
an implementation nobody runs.

Needs a GPU: the kernel is Triton.

    python scripts/parity/measure_hole_kernel_bf16_floor.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.ops.test_dllm_hole_attention import build_case, worst_relative  # noqa: E402
from veomni.models.transformers.diffusion_gemma.dg_hole import dense_hole_mask  # noqa: E402
from veomni.ops.kernels.dllm_hole_attention import dllm_hole_attention, hole_attention_reference  # noqa: E402


# What the recipe runs: canvas 256, a 221-key visible prompt for canvas 0, and
# `sm_scale` 1.0 (the model folds the usual 1/sqrt(d) into the query norm). The two
# head dims are the sliding and full-attention layers.
SHIPPED = {"canvas_length": 256, "canvases": 2, "clean_length": 512, "prompt": 221}
HEAD_DIMS = (256, 512)
SM_SCALE = 1.0
GRAD_NAMES = ("query", "clean_key", "clean_value", "noise_key", "noise_value")


def sdpa_over_hole_mask(inputs, sm_scale: float) -> torch.Tensor:
    """The dense fallback: one mask over ``[clean | noise]`` keys, then plain sdpa.

    This is what ``flat_seq`` runs when the kernel is off, so it is the right second
    path to measure -- not an idealized dense attention, but the one in the codebase.
    """
    query, clean_key, clean_value, noise_key, noise_value, lo, hi = inputs
    keys = torch.cat([clean_key, noise_key], dim=2)
    values = torch.cat([clean_value, noise_value], dim=2)

    repeat = query.shape[1] // keys.shape[1]
    keys = keys.repeat_interleave(repeat, dim=1)
    values = values.repeat_interleave(repeat, dim=1)

    mask = dense_hole_mask(lo, hi, clean_key.shape[2], query.shape[2] // lo.shape[1])
    return F.scaled_dot_product_attention(query, keys, values, attn_mask=mask, scale=sm_scale)


def measure(head_dim: int, seed: int) -> dict[str, dict[str, float]]:
    """Separate the two things bf16 costs, because they are different sizes.

    Rounding the *inputs* to bf16 changes the answer before any path has run, and that
    contribution belongs to neither implementation. So the oracle is run twice: once on
    the original fp32 draw and once on those same values rounded to bf16 and read back
    as fp32. The gap between the two oracles is the input floor; each path is scored
    against the *rounded* oracle, which is the only comparison that isolates its own
    arithmetic.

    Scoring the paths against the unrounded oracle instead -- the obvious thing to do --
    reports the input floor for both and buries the difference between them under it.
    """
    drawn = build_case(head_dim=head_dim, dtype=torch.float32, seed=seed, **SHIPPED)
    # Both paths must be handed bit-identical inputs, or the comparison measures the
    # draw rather than the arithmetic.
    rounded = [
        t.detach().to(torch.bfloat16).requires_grad_(t.requires_grad) if t.is_floating_point() else t for t in drawn
    ]
    upcast = [t.detach().float().requires_grad_(t.requires_grad) if t.is_floating_point() else t for t in rounded]

    reference = hole_attention_reference(*upcast, SM_SCALE)
    upstream = torch.randn_like(reference)
    reference_grads = torch.autograd.grad(reference, [t for t in upcast if t.requires_grad], upstream)

    drawn_out = hole_attention_reference(*drawn, SM_SCALE)
    drawn_grads = torch.autograd.grad(drawn_out, [t for t in drawn if t.requires_grad], upstream)

    results = {
        "bf16 inputs": {"forward": worst_relative(reference, drawn_out)}
        | {key: worst_relative(a, b) for key, a, b in zip(GRAD_NAMES, reference_grads, drawn_grads)}
    }

    outputs, gradients = {}, {}
    for name, run in (("kernel", lambda i: dllm_hole_attention(*i, SM_SCALE)), ("sdpa", sdpa_over_hole_mask)):
        leaves = [t for t in rounded if t.is_floating_point()]
        out = run(rounded, SM_SCALE) if name == "sdpa" else run(rounded)
        grads = torch.autograd.grad(out, leaves, upstream.to(torch.bfloat16))
        outputs[name], gradients[name] = out, grads
        results[name] = {"forward": worst_relative(out.float(), reference)} | {
            key: worst_relative(got.float(), want) for key, got, want in zip(GRAD_NAMES, grads, reference_grads)
        }

    results["kernel vs sdpa"] = {"forward": worst_relative(outputs["kernel"].float(), outputs["sdpa"].float())} | {
        key: worst_relative(a.float(), b.float())
        for key, a, b in zip(GRAD_NAMES, gradients["kernel"], gradients["sdpa"])
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("the hole kernel is Triton, so this needs a GPU")

    columns = ("forward", *GRAD_NAMES)
    for head_dim in HEAD_DIMS:
        print(f"\nhead_dim {head_dim}, canvas {SHIPPED['canvas_length']}, prompt {SHIPPED['prompt']}, bf16 vs fp32")
        print(f"{'path':<16}" + "".join(f"{name:>13}" for name in columns))
        for path, errors in measure(head_dim, args.seed).items():
            print(f"{path:<16}" + "".join(f"{errors[name]:>13.2e}" for name in columns))

    print(
        "\nRelative to the tensor's own scale.\n"
        "  `bf16 inputs`    -- the fp32 oracle on rounded vs unrounded inputs. Belongs to neither\n"
        "                      path, and is the larger effect; both rows below are scored against\n"
        "                      the rounded oracle so it is not counted twice.\n"
        "  `kernel`/`sdpa`  -- each path's own arithmetic error given those inputs.\n"
        "  `kernel vs sdpa` -- the number that matters: it bounds how much of a bf16 train-time\n"
        "                      difference between the two paths can be arithmetic alone."
    )


if __name__ == "__main__":
    main()
