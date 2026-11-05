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

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from torch.utils.data._utils.collate import default_collate

from ..data_collator import DataCollator


# Which axis of a per-sample packed tensor continues the packed sequence. Same
# idea as ``DEFAULT_DATA_COLLATE_INFO`` in ``data_collator`` — a declarative table
# keyed by literal field name, not a per-key branch — only these sample tensors
# already carry the batch axis the base collator appends at the end:
#
#   packed_input_ids      (1, T)      -> (1, sum T)
#   packed_position_ids   (3, 1, T)   -> (3, 1, sum T)
#   pixel_values          (N, D)      -> (sum N, D)      (one row per visual)
#   image_grid_thw        (N, 3)      -> (sum N, 3)
#
# A key that is not listed is not part of the packed sequence: it is batched by
# ``default_collate``, exactly as the base ``PackingCollator`` treats a field the
# model did not declare. That is also how the framework's own metadata
# (``ds_idx`` / ``source_name`` / ``cur_token_num``, appended to a sample by the
# dataset *after* the transform) reaches the trainer.
_PACKED_CONCAT_DIM: Dict[str, int] = {
    "packed_input_ids": 1,
    "packed_labels": 1,
    "packed_attention_mask": 1,
    "packed_position_ids": 2,
    "visual_pos_mask": 1,
    "pixel_values": 0,
    "image_grid_thw": 0,
}

# Summed across the batch rather than joined: a count of visuals, not a tensor.
_PACKED_SUM_KEYS = ("visual_num_real",)

# Re-derived from the joined sequence, so the per-sample copies are ignored.
_PACKED_REBUILT_KEYS = ("packed_cu_seqlens", "packed_max_length")


@dataclass
class SeedOmniCollator(DataCollator):
    """List-only collator for the SeedOmni ``conversation_list`` schema.

    Pairs with ``data_type: seedomni`` (see
    ``veomni/data/seed_omni/seedomni_transform.py``) which emits
    preprocessed per-sample ``{"conversation_list": [ConversationItem, ...]}``.
    This collator turns ``N`` such dicts into a single batched dict
    ``{"conversation_list": [[ConversationItem, ...], ...]}`` plus any extra
    keys present in the samples (passed through unchanged as a per-sample list).

    No stacking, no padding, no sequence-parallel slicing — the SeedOmni design
    contract puts all of that inside model modules' ``pre_forward``,
    where each module knows which fields it owns and how to slice them.
    Per-module CPU input preparation (chat template, tokenization, image/video
    patchification, per-sample dummy anchors, and ``_omni_token_len`` metadata)
    runs earlier in ``seedomni_transform`` via :class:`~veomni.models.seed_omni.processing_omni.OmniProcessor`,
    so dynamic batching (``dyn_bsz``) can measure each sample's total and
    effective token lengths before collation.

    The collator is intentionally **agnostic** about the exact set of
    keys present: any sample-level key is gathered into a list of length
    ``batch_size``.  This keeps the schema open for later extensions
    (precomputed features, offline-embedding payloads, etc.) without
    needing to update the collator.

    Image / audio tensors inside ``conversation_list`` items keep their
    original shapes — different samples may carry images of different
    resolutions, so the values **must not** be torch.stacked here.
    Vision encoders (e.g. ``JanusSiglip``) collate them in their own
    ``pre_forward`` after their image processor has matched shapes.

    Example::

        collator = SeedOmniCollator()
        batch = collator([
            {"conversation_list": [ConversationItem(type="text", value=<Tensor>, ...)]},
            {"conversation_list": [ConversationItem(type="image", value=<Tensor>, ...)]},
        ])
        # batch == {"conversation_list": [[ConversationItem(...)], ...]}
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("SeedOmniCollator received an empty feature list")

        # Gather keys across all samples.  Strict alignment (every sample
        # must share the same key set) — silent dropping of fields would
        # be a debugging nightmare.
        first_keys = set(features[0].keys())
        for i, f in enumerate(features[1:], start=1):
            if set(f.keys()) != first_keys:
                raise ValueError(
                    f"SeedOmniCollator: sample {i} has keys {set(f.keys())} but sample 0 has {first_keys}; "
                    f"all samples in a batch must carry the same key set."
                )
        if "conversation_list" not in first_keys:
            if "packed_input_ids" in first_keys:
                return _collate_packed(features, first_keys)
            raise KeyError(
                "SeedOmniCollator: every sample must have a 'conversation_list' key "
                "(produced by the seedomni data_transform) or the packed keys written by a "
                f"packed preprocessor; got {first_keys}"
            )

        return {key: [f[key] for f in features] for key in first_keys}


def _collate_packed(features: Sequence[Dict[str, Any]], keys: set[str]) -> Dict[str, Any]:
    """Batch already-packed per-sample dicts into one packed batch.

    A packed preprocessor consumes ``conversation_list`` and emits one packed
    sequence per sample, so the tensors describing that sequence join into one
    longer sequence instead of staying a per-sample list: every packed node
    indexes into the batch directly.

    Only the axes in :data:`_PACKED_CONCAT_DIM` are known here; everything else
    goes through ``default_collate``, the same rule the base ``PackingCollator``
    applies to a key the model did not declare. That is what carries the
    framework's own metadata — ``ds_idx`` / ``source_name`` / ``cur_token_num``
    are appended to a sample by the dataset *after* the transform, and
    ``default_collate`` joins them the way the meter expects (a tensor of
    per-sample indices, or a list when they are strings).
    """
    lengths = [int(f["packed_input_ids"].shape[-1]) for f in features]
    batch: Dict[str, Any] = {}
    for key in keys:
        if key in _PACKED_REBUILT_KEYS:
            continue
        values = [f[key] for f in features]
        if key in ("pixel_values", "image_grid_thw"):
            # Reordered below, so the batch-wide real/dummy split stays contiguous.
            continue
        if key in _PACKED_CONCAT_DIM:
            batch[key] = torch.cat(values, dim=_PACKED_CONCAT_DIM[key])
        elif key in _PACKED_SUM_KEYS:
            batch[key] = sum(int(v) for v in values)
        else:
            # Not part of the packed sequence — framework metadata, and whatever a
            # later preprocessor adds. Batch it the way every other trainer field
            # is batched.
            batch[key] = default_collate(values)

    _collate_visual_rows(features, batch)

    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    batch["packed_cu_seqlens"] = torch.tensor(cu, dtype=torch.int32)
    batch["packed_max_length"] = max(lengths)
    return batch


def _collate_visual_rows(features: Sequence[Dict[str, Any]], batch: Dict[str, Any]) -> None:
    """Join ``pixel_values`` / ``image_grid_thw`` with all real visuals first.

    Each sample packs its own visuals as ``[real..., dummy...]`` and reports how
    many of them are real in ``visual_num_real``. A plain concatenation would then
    order the batch as ``[s0 real, s0 dummy, s1 real, s1 dummy]`` while the summed
    ``visual_num_real`` stays ``Σ real`` — and the vision hook reads the first
    ``visual_num_real`` **rows** as the real ones, which would count ``s0``'s
    dummies and then either scatter them into the real mask or crash on a shape
    mismatch. Reorder so every real row precedes every dummy row.
    """
    has_pixels = "pixel_values" in features[0]
    has_grids = "image_grid_thw" in features[0]
    if not (has_pixels and has_grids):
        # Both keys are skipped by the main loop, so a half-present pair would be
        # dropped silently. Say so instead: the graph node needs both.
        if has_pixels or has_grids:
            missing = "image_grid_thw" if has_pixels else "pixel_values"
            raise KeyError(
                f"SeedOmniCollator: a packed sample carries visual patches but no {missing!r}; "
                "the vision node needs both."
            )
        return

    # One ordered list of blocks per sample, concatenated once: real blocks for
    # every sample, then dummy blocks for every sample. Copying per sample and
    # again for the join would copy every pixel twice.
    blocks_by_phase: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
    grids_by_phase: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
    for feature in features:
        grids = feature["image_grid_thw"]
        n_real = int(feature.get("visual_num_real", 0))
        # One patch block per visual row, so pixels and grids reorder together.
        blocks = torch.split(feature["pixel_values"], grids.prod(dim=1).tolist(), dim=0)
        blocks_by_phase[0].extend(blocks[:n_real])
        blocks_by_phase[1].extend(blocks[n_real:])
        grids_by_phase[0].append(grids[:n_real])
        grids_by_phase[1].append(grids[n_real:])
    pixel_blocks = blocks_by_phase[0] + blocks_by_phase[1]
    grid_rows = grids_by_phase[0] + grids_by_phase[1]
    if not pixel_blocks:
        # Both tensors present but empty: `torch.cat` on an empty list raises an
        # opaque shape error, so say what actually went wrong.
        raise ValueError(
            "SeedOmniCollator: a packed sample carries an empty visual pair "
            "(no `image_grid_thw` rows); the vision node needs at least one."
        )
    batch["pixel_values"] = torch.cat(pixel_blocks, dim=0)
    batch["image_grid_thw"] = torch.cat(grid_rows, dim=0)
