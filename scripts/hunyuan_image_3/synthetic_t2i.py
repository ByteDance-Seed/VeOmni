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

"""Synthetic T2I dataset writer for P7 smoke / overfit runs.

Two row modes matching :mod:`veomni.models.transformers.hunyuan_image_3.data_transform`:

- ``online_vae``: rows carry ``(id, prompt, image, width, height)``; the real
  ``HunyuanImage3ImageProcessor`` resizes/normalizes them and the online VAE
  encoder produces the latent posterior.
- ``posterior_cache``: rows carry ``(id, prompt, latent_mean, latent_logvar)``;
  the VAE is bypassed. Useful for isolating topology bugs from VAE-encode memory.

Data shape can be perturbed along two independent axes:
- ``image_sizes``: cycled ``(H, W)`` tuples per row -- feeds per-sample bucket
  select or the scheduled multi-resolution resize (Stage B).
- ``prompt_word_counts``: cycled word counts -- verifies the packed varlen path
  and its cu_seqlens tolerate heterogeneous text lengths.

``fixed=True`` uses one seed for every row, producing byte-identical content
regardless of ``n_rows`` -- required for the P7-2 / P7-3 memorization gates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from PIL import Image


# A small closed vocabulary of ASCII words. The tokenizer will yield a small,
# deterministic number of subword tokens per word, so ``prompt_word_counts``
# roughly controls the text-token count (exact count depends on the tokenizer).
_VOCAB: tuple[str, ...] = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
    "red",
    "blue",
    "green",
    "yellow",
    "cyan",
    "magenta",
    "orange",
    "violet",
    "cat",
    "dog",
    "bird",
    "fish",
    "tree",
    "house",
    "car",
    "boat",
    "sun",
    "moon",
    "cloud",
    "star",
    "river",
    "mountain",
    "field",
    "sky",
)


def _make_pil_image(height: int, width: int, seed: int) -> Image.Image:
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 255, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_prompt(word_count: int, seed: int) -> str:
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(_VOCAB), size=max(word_count, 1))
    return " ".join(_VOCAB[i] for i in idx)


def build_synthetic_rows(
    *,
    n_rows: int,
    mode: Literal["online_vae", "posterior_cache"] = "online_vae",
    image_sizes: Sequence[tuple[int, int]] = ((1024, 1024),),
    prompt_word_counts: Sequence[int] = (32,),
    latent_channels: int = 32,
    vae_spatial_factor: int = 16,
    fixed: bool = False,
    seed: int = 0,
) -> list[dict]:
    """Build the raw list of row dicts (unwritten). See module docstring for schema."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    if not image_sizes:
        raise ValueError("image_sizes must not be empty.")
    if not prompt_word_counts:
        raise ValueError("prompt_word_counts must not be empty.")
    if mode not in ("online_vae", "posterior_cache"):
        raise ValueError(f"unknown mode: {mode!r}")

    rows: list[dict] = []
    for i in range(n_rows):
        # ``fixed`` collapses per-row seeds to a single seed so every row is the
        # SAME. Keep image_size / prompt_len cycling even under ``fixed`` so a
        # multi-bucket overfit set is still legal; ``n_rows`` then just controls
        # the epoch length, not the content variety.
        row_seed = seed if fixed else seed + i
        height, width = image_sizes[i % len(image_sizes)]
        word_count = prompt_word_counts[i % len(prompt_word_counts)]

        row: dict = {
            "id": f"synth_{i:05d}",
            "prompt": _make_prompt(word_count, row_seed),
        }

        if mode == "online_vae":
            row["image"] = _make_pil_image(height, width, row_seed)
            row["width"] = int(width)
            row["height"] = int(height)
        else:  # posterior_cache
            if height % vae_spatial_factor or width % vae_spatial_factor:
                raise ValueError(
                    f"posterior_cache mode needs (H, W) divisible by vae_spatial_factor="
                    f"{vae_spatial_factor}; got ({height}, {width})."
                )
            grid_h = height // vae_spatial_factor
            grid_w = width // vae_spatial_factor
            rng = np.random.RandomState(row_seed)
            mean = rng.standard_normal(size=(latent_channels, grid_h, grid_w)).astype("float32")
            logvar = np.zeros_like(mean)
            row["latent_mean"] = mean.tolist()
            row["latent_logvar"] = logvar.tolist()

        rows.append(row)
    return rows


def write_synthetic_parquet(
    path: str | Path,
    *,
    n_rows: int,
    mode: Literal["online_vae", "posterior_cache"] = "online_vae",
    image_sizes: Sequence[tuple[int, int]] = ((1024, 1024),),
    prompt_word_counts: Sequence[int] = (32,),
    latent_channels: int = 32,
    vae_spatial_factor: int = 16,
    fixed: bool = False,
    seed: int = 0,
) -> Path:
    """Materialize a synthetic T2I parquet at ``path`` (HuggingFace ``datasets`` format)."""
    from datasets import Dataset

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_synthetic_rows(
        n_rows=n_rows,
        mode=mode,
        image_sizes=image_sizes,
        prompt_word_counts=prompt_word_counts,
        latent_channels=latent_channels,
        vae_spatial_factor=vae_spatial_factor,
        fixed=fixed,
        seed=seed,
    )
    Dataset.from_list(rows).to_parquet(str(path))
    return path


__all__ = ["build_synthetic_rows", "write_synthetic_parquet"]
