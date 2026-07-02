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

"""Pipeline-internal batch metadata side-channel.

A collator may collect per-sample bookkeeping (``ds_idx``, ``source_name``,
``source_boundaries``, ...) that downstream components (metrics, trainer) need
but the model forward must never see. Rather than scatter these as top-level
batch keys -- where they would be fed to ``default_collate`` / the model -- they
are stashed under a single reserved key, ``BATCH_METADATA_KEY``, as a plain
dict. Helpers below are the only sanctioned way to read / write / strip it.
"""

from typing import Any, Dict


__all__ = [
    "BATCH_METADATA_KEY",
    "attach_batch_metadata",
    "get_batch_metadata",
    "pop_batch_metadata",
]

# Reserved batch key holding the metadata dict. Leading underscore signals it is
# pipeline-internal; collators must keep it out of ``default_collate``.
BATCH_METADATA_KEY = "_batch_metadata"


def attach_batch_metadata(batch: Dict[str, Any], **metadata: Any) -> Dict[str, Any]:
    """Merge ``metadata`` into the batch's metadata dict, creating it if absent.

    Returns the same ``batch`` for convenience. Repeated calls accumulate.
    """
    meta = batch.get(BATCH_METADATA_KEY)
    if not isinstance(meta, dict):
        meta = {}
        batch[BATCH_METADATA_KEY] = meta
    meta.update(metadata)
    return batch


def get_batch_metadata(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Return the batch's metadata dict, or an empty dict if none is attached.

    Read-only accessor: never mutates ``batch``.
    """
    meta = batch.get(BATCH_METADATA_KEY)
    return meta if isinstance(meta, dict) else {}


def pop_batch_metadata(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Remove and return the batch's metadata dict (empty dict if none).

    Call this before the model forward so the side-channel never reaches the
    model.
    """
    meta = batch.pop(BATCH_METADATA_KEY, None)
    return meta if isinstance(meta, dict) else {}
