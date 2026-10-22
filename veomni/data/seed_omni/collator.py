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

from ..data_collator import DataCollator


def _collate_tensor_field(key: str, values: Sequence[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    """Assemble per-sample top-level tensors written by a packed preprocessor into a batch tensor."""
    if len(values) == 1:
        return values[0]
    first = values[0]
    if first.dim() == 0 and all(v.dim() == 0 for v in values):
        return torch.stack(list(values), dim=0)
    if key in {"cu_seqlens", "packed_cu_seqlens"} and all(
        v.dim() == 1 and v.numel() >= 2 and int(v[0].item()) == 0 for v in values
    ):
        parts = [first]
        offset = first[-1]
        for v in values[1:]:
            parts.append(v[1:] + offset)
            offset = offset + v[-1]
        return torch.cat(parts, dim=0)
    if first.dim() == 1 and all(v.dim() == 1 for v in values):
        return torch.cat(list(values), dim=0)
    if first.dim() >= 2 and all(v.dim() == first.dim() for v in values):
        if first.shape[0] == 1 and all(v.shape[:-1] == first.shape[:-1] for v in values):
            return torch.cat(list(values), dim=-1)
        if first.dim() == 3 and first.shape[:2] == (3, 1) and all(v.shape[:2] == (3, 1) for v in values):
            return torch.cat(list(values), dim=-1)
        if key in {"position_ids", "packed_position_ids"} and all(v.shape[:-1] == first.shape[:-1] for v in values):
            return torch.cat(list(values), dim=-1)
        if all(v.shape[1:] == first.shape[1:] for v in values):
            return torch.cat(list(values), dim=0)
        if all(v.shape[:-1] == first.shape[:-1] for v in values):
            return torch.cat(list(values), dim=-1)
    return list(values)


@dataclass
class SeedOmniCollator(DataCollator):
    """Collator for the SeedOmni ``conversation_list`` and packed-tensor schemas.

    Pairs with ``data_type: seedomni`` (see
    ``veomni/data/seed_omni/seedomni_transform.py``) which emits
    preprocessed per-sample ``{"conversation_list": [ConversationItem, ...]}``
    (plus any top-level packed tensors written by a ``ModulePreprocessorBase.__call__``
    override). This collator turns ``N`` such dicts into a single batched dict:
    ``"conversation_list"`` and non-tensor metadata keys are gathered into
    per-sample lists, while top-level ``torch.Tensor`` fields are concatenated
    into packed batch tensors.

    No stacking or sequence-parallel slicing of ``conversation_list`` items —
    the SeedOmni design contract puts all of that inside model modules'
    ``pre_forward``, where each module knows which fields it owns and how to
    slice them. Per-module CPU input preparation (chat template, tokenization,
    image/video patchification, per-sample dummy anchors, and ``_omni_token_len``
    metadata) runs earlier in ``seedomni_transform`` via
    :class:`~veomni.models.seed_omni.processing_omni.OmniProcessor`, so dynamic
    batching (``dyn_bsz``) can measure each sample's total and effective token
    lengths before collation.

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
        if "conversation_list" not in first_keys and not any(
            isinstance(features[0][k], torch.Tensor) for k in first_keys
        ):
            raise KeyError(
                "SeedOmniCollator: every sample must have a 'conversation_list' key "
                "(produced by the seedomni data_transform); "
                f"got {first_keys}"
            )

        batch: Dict[str, Any] = {}
        for key in first_keys:
            values = [f[key] for f in features]
            if key != "conversation_list" and all(isinstance(v, torch.Tensor) for v in values):
                batch[key] = _collate_tensor_field(key, values)
            else:
                batch[key] = values
        return batch
