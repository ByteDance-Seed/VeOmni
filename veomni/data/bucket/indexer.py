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

"""Per-sample bucket precomputation for the bucket-aware data plane.

Startup-time scan of a dataset's ``(width, height)`` metadata columns to
produce ``bucket_ids[N]`` -- the input consumed by
:class:`~veomni.data.bucket.batch_sampler.BucketBatchSampler`. The fast path
only reads two columns (no PIL decode); the slow path falls back to opening
each image via PIL when the columns are missing.

The indexer also computes a stable fingerprint over ``(policy_fingerprint,
dataset identity, keys)`` for the DCP manifest -- resume validates that the
dataset + resolution policy + column mapping did not drift under the sampler's
cursors.
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ...utils import logging


if TYPE_CHECKING:
    from torch.utils.data import Dataset


logger = logging.get_logger(__name__)

_SLOW_PATH_WARNED: set[str] = set()


class BucketIndexer:
    """Precompute ``bucket_id`` per sample from a dataset's metadata columns.

    Args:
        bucket_of: Callable ``(width, height) -> bucket_id``; supplied by the
            per-model resolution policy so this indexer stays model-agnostic.
        policy_fingerprint: Stable hash of the resolution policy's bucket
            table (geometry + ordering). Mixed into :meth:`fingerprint` so the
            DCP manifest gate rejects a resume where the policy drifted.
        width_key / height_key: Dataset column names carrying the raw
            ``(width, height)`` metadata for the fast path.
        image_key: Fallback column for the PIL slow path when the width/height
            columns are missing.
    """

    def __init__(
        self,
        *,
        bucket_of: Callable[[int, int], int],
        policy_fingerprint: str,
        width_key: str = "width",
        height_key: str = "height",
        image_key: str = "image",
    ) -> None:
        if not callable(bucket_of):
            raise TypeError("bucket_of must be a callable (width, height) -> bucket_id.")
        if not isinstance(policy_fingerprint, str) or not policy_fingerprint:
            raise TypeError("policy_fingerprint must be a non-empty string.")
        if not width_key or not height_key:
            raise ValueError("width_key and height_key must be non-empty strings.")
        self._bucket_of = bucket_of
        self._policy_fingerprint = policy_fingerprint
        self._width_key = width_key
        self._height_key = height_key
        self._image_key = image_key

    # ---- public API ----

    def index(self, dataset: "Dataset") -> np.ndarray:
        """Return ``bucket_ids[N] : int32`` -- one per dataset sample.

        Fast path: read the ``width_key`` / ``height_key`` columns directly (no
        image decode). Slow path (missing columns): open PIL per sample once at
        startup and warn -- recommend adding the metadata columns to the
        dataset.
        """
        if hasattr(dataset, "__len__"):
            n = len(dataset)
        else:
            raise TypeError(
                "BucketIndexer requires a map-style dataset with __len__; iterable datasets "
                "are unsupported (the batch sampler needs random-access indices)."
            )
        if n == 0:
            return np.zeros(0, dtype=np.int32)

        widths, heights, used_slow_path = self._read_wh_columns(dataset, n)
        if used_slow_path:
            fp = self.fingerprint(dataset)
            if fp not in _SLOW_PATH_WARNED:
                _SLOW_PATH_WARNED.add(fp)
                logger.warning_rank0(
                    "BucketIndexer slow path: dataset does not expose "
                    f"'{self._width_key}' / '{self._height_key}' columns, opened {n} images via PIL "
                    "at startup. Add the columns during dataset preprocessing to skip this scan on "
                    "the next run."
                )

        bucket_ids = np.empty(n, dtype=np.int32)
        # Cache per-(w, h) -> bucket_id to avoid repeated selection work on datasets
        # with a small number of distinct aspect ratios.
        cache: dict[tuple[int, int], int] = {}
        for i in range(n):
            key = (int(widths[i]), int(heights[i]))
            bucket_id = cache.get(key)
            if bucket_id is None:
                bucket_id = int(self._bucket_of(key[0], key[1]))
                cache[key] = bucket_id
            bucket_ids[i] = bucket_id
        return bucket_ids

    def fingerprint(self, dataset: "Dataset") -> str:
        """Stable fingerprint of ``(policy, dataset identity, keys)``.

        Used by the DCP manifest to hard-gate resume: dataset / policy / column
        mapping drift -> the sampler's cursors would point at the wrong
        samples, so we refuse to reload the cursor state and force a clean
        start.

        Dataset identity is best-effort: uses HuggingFace ``datasets.
        _fingerprint`` when available, else falls back to ``(len(dataset),
        type(dataset).__name__)``.
        """
        parts: dict[str, Any] = {
            "policy_fingerprint": self._policy_fingerprint,
            "width_key": self._width_key,
            "height_key": self._height_key,
            "dataset_len": int(len(dataset)) if hasattr(dataset, "__len__") else -1,
        }
        hf_fp = getattr(dataset, "_fingerprint", None)
        if isinstance(hf_fp, str):
            parts["hf_fingerprint"] = hf_fp
        else:
            parts["dataset_type"] = type(dataset).__name__
        payload = json.dumps(parts, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    # ---- internals ----

    def _read_wh_columns(self, dataset: "Dataset", n: int) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return ``(widths, heights, used_slow_path)`` for all N samples."""
        # HuggingFace ``Dataset`` supports column access without materializing rows.
        hf_columns = None
        if hasattr(dataset, "column_names") and hasattr(dataset, "__getitem__"):
            column_names = list(dataset.column_names or ())
            if self._width_key in column_names and self._height_key in column_names:
                try:
                    widths_col = dataset[self._width_key]
                    heights_col = dataset[self._height_key]
                except Exception:  # noqa: BLE001 -- best-effort HF path
                    hf_columns = None
                else:
                    hf_columns = (widths_col, heights_col)
        if hf_columns is not None:
            widths = np.asarray(hf_columns[0], dtype=np.int64)
            heights = np.asarray(hf_columns[1], dtype=np.int64)
            if widths.shape != (n,) or heights.shape != (n,):
                raise ValueError(
                    f"Column '{self._width_key}' / '{self._height_key}' shapes do not match "
                    f"dataset length {n}: got {widths.shape} / {heights.shape}."
                )
            return widths, heights, False

        # Slow path -- read each sample and either pick columns off the dict or
        # PIL-decode the image field.
        widths = np.empty(n, dtype=np.int64)
        heights = np.empty(n, dtype=np.int64)
        for i in range(n):
            sample = dataset[i]
            if isinstance(sample, dict) and self._width_key in sample and self._height_key in sample:
                widths[i] = int(sample[self._width_key])
                heights[i] = int(sample[self._height_key])
                continue
            image = sample.get(self._image_key) if isinstance(sample, dict) else None
            if image is None:
                raise KeyError(
                    f"Sample {i} exposes neither '{self._width_key}'/'{self._height_key}' columns "
                    f"nor '{self._image_key}' for PIL fallback. Add width/height metadata to the "
                    "dataset or rename the image field via the model-side configuration."
                )
            w, h = extract_image_size(image)
            widths[i] = w
            heights[i] = h
        return widths, heights, True


def extract_image_size(image: Any) -> tuple[int, int]:
    """Extract ``(width, height)`` from a PIL image, raw bytes, or a path-like value."""
    # Import lazily so environments without PIL don't hard-crash at import.
    from PIL import Image

    if hasattr(image, "size") and hasattr(image, "mode"):
        return int(image.size[0]), int(image.size[1])
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(image)) as pil_image:
            return int(pil_image.size[0]), int(pil_image.size[1])
    if isinstance(image, str):
        with Image.open(image) as pil_image:
            return int(pil_image.size[0]), int(pil_image.size[1])
    if isinstance(image, dict) and "bytes" in image:
        with Image.open(io.BytesIO(image["bytes"])) as pil_image:
            return int(pil_image.size[0]), int(pil_image.size[1])
    raise TypeError(f"Unsupported image type for slow-path bucket indexing: {type(image).__name__}.")


def _reset_slow_path_warning_cache_for_tests() -> None:
    """Only for tests -- restore fresh state for the slow-path warn-once cache."""
    _SLOW_PATH_WARNED.clear()


__all__ = ["BucketIndexer", "extract_image_size"]
