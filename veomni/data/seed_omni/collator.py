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
from typing import TYPE_CHECKING, Any, Dict, Sequence

from ..data_collator import DataCollator


if TYPE_CHECKING:
    from ...models.seed_omni.processing_omni import OmniProcessor


@dataclass
class SeedOmniCollator(DataCollator):
    """List-only collator for the SeedOmni ``conversation_list`` schema.

    Pairs with ``data_type: seedomni`` (see
    ``veomni/data/seed_omni/seedomni_transform.py``) which emits
    per-sample ``{"conversation_list": [ConversationItem, ...]}``.
    This collator turns ``N`` such dicts into a single batched dict
    ``{"conversation_list": [[ConversationItem, ...], ...]}`` plus any extra
    keys present in the samples (passed through unchanged as a per-sample list).

    No stacking, no padding, no sequence-parallel slicing — the SeedOmni design
    contract puts all of that inside model modules' ``pre_forward``,
    where each module knows which fields it owns and how to slice them.
    The data layer's job stops at "list of per-sample dicts → dict of
    per-sample lists".

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

        collator = SeedOmniCollator(processor=processor)
        batch = collator([
            {"conversation_list": [{"type": "text", "value": "hi", ...}]},
            {"conversation_list": [{"type": "image", "value": <Tensor>, ...}]},
        ])
        # batch == {"conversation_list": [[ConversationItem(...)], ...]}

    ``processor`` is an :class:`~veomni.models.seed_omni.processing_omni.OmniProcessor` built
    via :meth:`~veomni.models.seed_omni.processing_omni.OmniProcessor.from_config` from the
    active graph modules' config (see ``OmniTrainer._build_collate_fn``). Its preprocessor
    chain runs, in order, over the grouped batch (``conversation_list`` plus any extra
    keys) so the heavy per-module CPU input-prep (tokenize / image normalize)
    executes inside the DataLoader worker and overlaps with GPU compute via prefetch, instead of
    blocking the main process inside each module's ``pre_forward``.
    """

    processor: "OmniProcessor"

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
            raise KeyError(
                "SeedOmniCollator: every sample must have a 'conversation_list' key "
                "(produced by the seedomni data_transform); "
                f"got {first_keys}"
            )

        batch = {key: [f[key] for f in features] for key in first_keys}

        # Worker-side CPU input-prep (tokenize / image normalize) over the grouped
        # batch. Mutates items in place; the modules' thin pre_forward then only
        # moves to device. Order is FIXED and SERIAL: ``config.modules:``
        # declaration order (see :class:`OmniProcessor`).
        self.processor.preprocess_batch(batch, inference=False)
        return batch
