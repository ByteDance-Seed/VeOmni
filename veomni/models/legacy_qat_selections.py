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

"""Model configs written before quantization-aware training became per-family.

**Delete this file** (and the one call in ``configuration_base._resolve_ops``) to
drop that compatibility. Call site:

* ``configuration_base._resolve_ops`` — every path that validates ops selections
  ends there (initial parse, ``from_pretrained``, ``save_pretrained``
  round-trip, caller overrides), so the migration cannot be bypassed.

Only one retired name is handled here:

* ``qat_implementation`` — a single global recipe. It was removed in favour of
  one row per family (``linear_implementation``, ``swiglu_mlp_implementation``,
  ``moe_implementation``, ``kv_round_implementation``).

Two old values, and they do not migrate the same way:

* ``"none"`` — was the default and meant "train in the model dtype". Today that
  is simply an unset field, so the migration drops the key and nothing else
  changes. This covers essentially every published checkpoint, since ``"none"``
  was what they all recorded unless QAT had been switched on.
* ``"fp8_blockwise"`` — named one recipe that has since become several rows. It
  **raises** rather than translating, because the replacement depends on which
  families the model actually has: a Qwen3 checkpoint has no routed experts and
  no KV cache, while DeepSeek-V4 has both. Guessing would either silently
  under-quantize or select a family the model never had, so the reader is told
  which fields to set instead.

Removal condition: safe once no checkpoint predating the per-family rows is
expected to load. There is no on-disk marker to detect, so this cannot expire on
its own — the date has to be a decision.
"""

from __future__ import annotations

from typing import Any, Mapping


__all__ = [
    "RETIRED_SELECTIONS",
    "migrate_retired_selections",
]

#: Retired ops-selection names, mapped from each old value to the selections that
#: replace it. An old value absent from the inner map cannot be translated here.
RETIRED_SELECTIONS: dict[str, dict[str, dict[str, Any]]] = {
    "qat_implementation": {
        # "train in the model dtype" -- the current spelling is an unset field.
        "none": {},
    },
}


def migrate_retired_selections(selections: Mapping[str, Any]) -> dict[str, Any]:
    """Translate selections written by a release that still had the retired names.

    Args:
        selections: The ops selections read from a config, possibly carrying a
            retired key.

    Returns:
        The selections with retired keys replaced by their current equivalent. A
        mapping that carries none of them is returned unchanged, so the common
        path costs one dict copy at most.

    Raises:
        ValueError: if a retired value has no translation, because choosing one
            needs to know the model. The message names the fields to set instead.
    """
    if not any(field in selections for field in RETIRED_SELECTIONS):
        return dict(selections)

    migrated = dict(selections)
    for field, mapping in RETIRED_SELECTIONS.items():
        if field not in migrated:
            continue
        value = migrated.pop(field)
        if value is None:
            continue
        if value not in mapping:
            raise ValueError(
                f"{field}={value!r} came from a config written before quantization-aware training "
                f"became per-family, and cannot be translated without knowing which families the "
                f"model has. Select them explicitly: 'linear_implementation' / "
                f"'swiglu_mlp_implementation' / 'moe_implementation' = 'qat', plus "
                f"'kv_round_implementation' = 'triton' for stored KV. Only {field}='none' "
                f"(train in the model dtype) migrates automatically. See "
                f"veomni/models/legacy_qat_selections.py."
            )
        migrated.update(mapping[value])
    return migrated
