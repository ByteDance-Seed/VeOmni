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

"""Generic CPU-side helpers for multimodal forward metadata.

See `.agents/knowledge/multimodal_metadata.md` for the design.

This module holds **only the model-agnostic** pieces of the precompute
pipeline — the bits every Qwen multimodal transform shares:

  * `per_sample_metadata` — called by the data transforms; emits Python-list
    mirrors of the per-sample `grid_thw` tensors.
  * `merge_position_id_returns` — copies the `position_id_func` output into
    the per-sample feature dict.

The **batch-level** derivation (ViT `cu_seqlens` / `max_seqlen`, window
`cu_seqlens`, the SP-pad tail extension, and bundling into the
`multimodal_metadata` dict) is intentionally **not** here: those formulas
depend on per-model config (`spatial_merge_size`, `window_size`, …) the
collator does not have. Each model owns that logic via a picklable
`get_metadata_collate_func()` hook — a `partial` over a patchgen-generated
helper — which the collator invokes after SP padding. See the per-model
`*_patch_gen_config.py` files and `data_collator.MetadataCollateFunc`.
"""

from typing import Any, Dict


# Keys this module reads from per-sample dicts (grid_thw tensors emitted by HF processors).
_PER_SAMPLE_GRID_KEYS = ("image_grid_thw", "video_grid_thw")

# Mapping from grid_thw key to the list-suffixed key the model consumes.
_GRID_TO_LIST_KEY = {
    "image_grid_thw": "image_grid_thw_list",
    "video_grid_thw": "video_grid_thw_list",
}


def per_sample_metadata(model_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-sample CPU metadata to merge into a feature dict.

    Currently emits ``image_grid_thw_list`` / ``video_grid_thw_list`` —
    Python lists of ``[t, h, w]`` triplets. ``PackingCollator`` concatenates
    these across samples; the model's ``metadata_collate_func`` hook then
    consumes them to derive batch-level ViT metadata.
    """
    out: Dict[str, Any] = {}
    for key in _PER_SAMPLE_GRID_KEYS:
        grid = model_inputs.get(key)
        if grid is None:
            continue
        # ``grid`` is a CPU int tensor from the HF image/video processor — no sync.
        out[_GRID_TO_LIST_KEY[key]] = grid.tolist()
    return out


def merge_position_id_returns(
    model_inputs: Dict[str, Any],
    position_id_returns: Dict[str, Any],
) -> None:
    """Copy the ``position_id_func`` output into ``model_inputs`` in place.

    ``position_id_func`` for Qwen multimodal models returns a dict; only
    ``position_ids`` is propagated into the training feature dict. The
    ``rope_deltas`` it also returns is generation-only (KV-cache decode) and
    is intentionally dropped — the training forward always receives a
    precomputed ``position_ids`` so it never derives or reads ``rope_deltas``.
    """
    if "position_ids" not in position_id_returns:
        raise KeyError(
            f"position_id_func returned dict without 'position_ids'; got keys: {sorted(position_id_returns)}"
        )
    model_inputs["position_ids"] = position_id_returns["position_ids"]
