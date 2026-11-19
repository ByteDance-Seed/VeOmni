#!/usr/bin/env python
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
"""In which precision does NeMo's loader fold ``per_expert_scale``?

``DiffusionGemmaStateDictAdapter._fold_per_expert_scale`` is called on the raw
state-dict values and ``.to(self.dtype)`` is applied to the result. Both operands come
from a bf16 checkpoint, so the product is rounded to 8 mantissa bits and the widening
that follows preserves the rounding: a run asking for fp32 weights gets an expert
down-projection carrying ~1.5e-3 of relative error, on ~90% of the model's parameters.

That single rounding is the whole of the 8% forward gap between the two stacks at the
shipped 26B checkpoint, and it destroys the gradient comparison outright (719 of 723
tensors, because a perturbed expert output moves top-k routing). It is invisible on the
tiny parity artifact, which is written in fp32 and whose gradient variant has a unit
scale -- the fold is exact either way. See ``README.md`` and constraint 44.

This calls the real adapter on the real checkpoint's tensors, so it reports what the
loader does rather than what it is read to do, and exits non-zero while the fold is
lossy. Runs under the NeMo venv:

    $AM/.venv/bin/python scripts/parity/check_expert_scale_fold.py \\
        --ckpt /mnt/hdfs/user_dir/omni_v2/ckpt/diffusiongemma-26B-A4B-it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def read_tensor(ckpt: Path, key: str) -> torch.Tensor:
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(ckpt / index[key]), framework="pt") as handle:
        return handle.get_tensor(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="the fold is lossless below this; bf16 rounding lands ~1.5e-3 above it",
    )
    args = parser.parse_args()

    from nemo_automodel.components.models.diffusion_gemma.state_dict_adapter import (
        DiffusionGemmaStateDictAdapter,
    )

    down = read_tensor(args.ckpt, f"model.decoder.layers.{args.layer}.experts.down_proj")
    scale = read_tensor(args.ckpt, f"model.decoder.layers.{args.layer}.router.per_expert_scale")
    print(f"checkpoint: down_proj {down.dtype} {tuple(down.shape)}, per_expert_scale {scale.dtype}")
    print(f"per_expert_scale spans [{scale.float().min():.4f}, {scale.float().max():.4f}]")

    transposed = down.transpose(-2, -1)
    # What the loader produces, then widened exactly as `_from_hf_w_merged_qkv_and_experts` does.
    as_shipped = DiffusionGemmaStateDictAdapter._fold_per_expert_scale(transposed, scale).to(torch.float32)
    # The same fold with the rounding removed: widen the operands, not the product.
    lossless = transposed.to(torch.float32) * scale[:, None, None].to(torch.float32)

    error = ((as_shipped - lossless).norm() / lossless.norm()).item()
    print(f"\nthe loader's fp32 down-projection differs from an exact fold by {error:.3e} (relative)")

    if error > args.tol:
        print(
            f"\nLOSSY: the fold rounds the product to the operands' {down.dtype} before widening.\n"
            "Fix in `_from_hf_w_merged_qkv_and_experts`: upcast to `self.dtype` before the\n"
            "multiply, which makes the existing `.to(self.dtype)` a no-op."
        )
        return 1

    print("\nOK: the fold carries the target precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
