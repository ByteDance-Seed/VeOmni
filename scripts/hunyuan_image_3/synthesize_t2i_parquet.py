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
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI to write a P7-smoke synthetic T2I parquet.

Examples:

    # Stage A: single 1024px bucket, single prompt length.
    python3 scripts/hunyuan_image_3/synthesize_t2i_parquet.py \
        --out /tmp/hy3_p7_stage_A.parquet --mode online_vae \
        --n_rows 128 --image_sizes 1024,1024 --prompt_lengths 128

    # Stage B: multi-resolution + multi-length prompts.
    python3 scripts/hunyuan_image_3/synthesize_t2i_parquet.py \
        --out /tmp/hy3_p7_stage_B.parquet --mode online_vae \
        --n_rows 192 --image_sizes 512,512:768,768:1024,1024 \
        --prompt_lengths 32,128,256

    # Overfit set (P7-2/P7-3): identical rows for memorization.
    python3 scripts/hunyuan_image_3/synthesize_t2i_parquet.py \
        --out /tmp/hy3_overfit.parquet --mode online_vae \
        --n_rows 8 --image_sizes 512,512 --prompt_lengths 32 --fixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow ``python3 scripts/hunyuan_image_3/synthesize_t2i_parquet.py ...`` without setting PYTHONPATH.
sys.path.insert(0, str(Path(__file__).parent))
from synthetic_t2i import write_synthetic_parquet  # noqa: E402


def _parse_sizes(spec: str) -> list[tuple[int, int]]:
    """``512,512:768,768`` -> ``[(512,512),(768,768)]``. Colon separates pairs."""
    pairs: list[tuple[int, int]] = []
    for chunk in spec.split(":"):
        h_str, w_str = chunk.split(",")
        pairs.append((int(h_str), int(w_str)))
    return pairs


def _parse_ints_csv(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic T2I parquet for P7 smoke.")
    parser.add_argument("--out", required=True, help="Output parquet path.")
    parser.add_argument("--mode", choices=["online_vae", "posterior_cache"], default="online_vae")
    parser.add_argument("--n_rows", type=int, default=128)
    parser.add_argument(
        "--image_sizes",
        type=_parse_sizes,
        default=[(1024, 1024)],
        help="H,W pairs; colon-separated for multiple (e.g. '512,512:768,768:1024,1024').",
    )
    parser.add_argument(
        "--prompt_lengths",
        type=_parse_ints_csv,
        default=[128],
        help="Word counts (comma-separated) cycled across rows.",
    )
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--vae_spatial_factor", type=int, default=16)
    parser.add_argument("--fixed", action="store_true", help="Deterministic identical rows (overfit).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = write_synthetic_parquet(
        args.out,
        n_rows=args.n_rows,
        mode=args.mode,
        image_sizes=args.image_sizes,
        prompt_word_counts=args.prompt_lengths,
        latent_channels=args.latent_channels,
        vae_spatial_factor=args.vae_spatial_factor,
        fixed=args.fixed,
        seed=args.seed,
    )
    print(
        f"WROTE {path} rows={args.n_rows} mode={args.mode} "
        f"sizes={args.image_sizes} lengths={args.prompt_lengths} "
        f"fixed={args.fixed} seed={args.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
