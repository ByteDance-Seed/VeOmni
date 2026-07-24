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

"""Versioned checkpoint manifest for exact same-topology DCP resume (impl §15.2).

A DCP checkpoint records nothing about the *topology* or *training semantics* it was
produced with. Resuming a same-name checkpoint under a different parallel mesh makes
the DCP load collective reshard incorrectly (or deadlock); resuming under a different
component policy / resolution-bucket / flow config silently changes the training. This
module writes a small ``checkpoint_manifest.json`` beside the DCP files on save and, on
load, has **all ranks validate + collectively agree before entering the DCP collective**
so an incompatible resume fails fast and identically on every rank instead of
deadlocking mid-collective.

Design:
* HARD mismatches (raise before any DCP collective): manifest version, model_type, the
  parallel mesh topology, and every caller-provided ``extra_hashes`` entry (e.g. the
  Hunyuan Image 3 bucket-scheduler policy hash, component policy, flow identity). These
  either corrupt/deadlock the collective or silently change training.
* SOFT mismatches (warn only): informational fields such as ``train_seed``.
* A missing manifest (checkpoint written before this feature) is honored for backward
  compatibility: validation is skipped with a warning.

The manifest is generic — every model gets version + model_type + mesh — and models add
their own identity through ``extra_hashes`` without this module knowing about them.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple


MANIFEST_VERSION = 1
MANIFEST_FILENAME = "checkpoint_manifest.json"


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _mesh_dict(parallel_state) -> Dict[str, Any]:
    return {
        "world_size": int(parallel_state.world_size),
        "dp_size": int(parallel_state.dp_size),
        "dp_replicate_size": int(parallel_state.dp_replicate_size),
        "dp_shard_size": int(parallel_state.dp_shard_size),
        "tp_size": int(parallel_state.tp_size),
        "pp_size": int(parallel_state.pp_size),
        "cp_size": int(parallel_state.cp_size),
        "ulysses_size": int(parallel_state.ulysses_size),
        "extra_parallel_sizes": {k: int(v) for k, v in dict(parallel_state.extra_parallel_sizes).items()},
    }


def build_checkpoint_manifest(
    *,
    model_config,
    parallel_state,
    extra_hashes: Optional[Mapping[str, Any]] = None,
    soft_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the manifest for the current run.

    ``extra_hashes`` are HARD-gated model identity values (already-stable dicts/strings);
    ``soft_fields`` are recorded and only WARN on mismatch (e.g. ``train_seed``).
    """
    mesh = _mesh_dict(parallel_state)
    config_dict = model_config.to_dict() if hasattr(model_config, "to_dict") else dict(model_config or {})
    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "model_type": config_dict.get("model_type"),
        "mesh": mesh,
        "mesh_hash": _stable_hash(mesh),
        # Recorded for debugging only; structural config mismatches are caught by the
        # DCP load itself (key/shape), so config_hash is not a hard gate (it would
        # false-positive on runtime-only fields like attn_implementation).
        "config_hash": _stable_hash(config_dict),
        "extra_hashes": dict(extra_hashes or {}),
        "soft_fields": dict(soft_fields or {}),
    }
    return manifest


def validate_checkpoint_manifest(saved: Mapping[str, Any], current: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """Return ``(hard_reasons, soft_reasons)``; empty ``hard_reasons`` == safe to resume."""
    hard: List[str] = []
    soft: List[str] = []

    if saved.get("manifest_version") != current.get("manifest_version"):
        hard.append(f"manifest_version {saved.get('manifest_version')} != {current.get('manifest_version')}")
    if saved.get("model_type") != current.get("model_type"):
        hard.append(f"model_type {saved.get('model_type')!r} != {current.get('model_type')!r}")
    if saved.get("mesh_hash") != current.get("mesh_hash"):
        hard.append(f"parallel mesh changed: {saved.get('mesh')} != {current.get('mesh')}")

    saved_extra = dict(saved.get("extra_hashes") or {})
    current_extra = dict(current.get("extra_hashes") or {})
    for key in sorted(set(saved_extra) | set(current_extra)):
        if saved_extra.get(key) != current_extra.get(key):
            hard.append(f"{key} identity changed: {saved_extra.get(key)!r} != {current_extra.get(key)!r}")

    saved_soft = dict(saved.get("soft_fields") or {})
    current_soft = dict(current.get("soft_fields") or {})
    for key in sorted(set(saved_soft) | set(current_soft)):
        if saved_soft.get(key) != current_soft.get(key):
            soft.append(f"{key} changed: {saved_soft.get(key)!r} != {current_soft.get(key)!r}")

    return hard, soft


def write_checkpoint_manifest(checkpoint_dir: str, manifest: Mapping[str, Any]) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, MANIFEST_FILENAME)
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
    return path


def read_checkpoint_manifest(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """Return the saved manifest, or ``None`` if the checkpoint predates this feature."""
    path = os.path.join(checkpoint_dir, MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


__all__ = [
    "MANIFEST_VERSION",
    "MANIFEST_FILENAME",
    "build_checkpoint_manifest",
    "validate_checkpoint_manifest",
    "write_checkpoint_manifest",
    "read_checkpoint_manifest",
]
