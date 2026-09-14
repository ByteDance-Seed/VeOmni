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

"""Single source of truth for checkpoint paths.

Every writer and every reader derives its path from this module, so a save and
the load that follows it cannot disagree. That was the failure mode this
replaces: the checkpoint manager built its own ``save_dir`` while the
checkpointer appended ``global_step_{N}`` itself, and the SeedOmni V2 manager
built a third set on top, which is how its HF export ended up sharing a
directory with the DCP shards.

The ``module`` argument is what makes one writer serve both a single-model job
and a multi-module (SeedOmni V2) one. It is the empty string for a single model,
which collapses ``model/<module>/`` to ``model/``; nothing else differs between
the two cases.

On-disk contract: ``docs/usage/checkpoint.md``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from ..utils.checkpoint_utils import _GLOBAL_STEP_PREFIX


# Resume tree. Everything needed to continue training, and nothing else.
MODEL_DIRNAME = "model"
WEIGHTS_DIRNAME = "ckpt"
OPTIMIZER_DIRNAME = "optimizer"
LR_SCHEDULER_FILENAME = "lr_scheduler.pt"
LOADER_DIRNAME = "loader"
EXTRA_STATE_DIRNAME = "extra_state"
RANK_FILE_FORMAT = "rank_{}.pt"

# Export tree. Inference artifacts; resume never reads these.
HF_EXPORT_DIRNAME = "hf_ckpt"
LORA_EXPORT_DIRNAME = "lora_ckpt"

# Step-level completion marker.
MANIFEST_FILENAME = "checkpoint_manifest.json"
MANIFEST_FORMAT_VERSION = 1


def step_dir(save_path: str, global_step: int) -> str:
    """``<save_path>/global_step_{N}`` — the root of one step's checkpoint."""
    return os.path.join(save_path, f"{_GLOBAL_STEP_PREFIX}{global_step}")


def model_dir(step_root: str, module: str = "") -> str:
    """Root of one model's resume state: weights, optimizer and scheduler."""
    root = os.path.join(step_root, MODEL_DIRNAME)
    return os.path.join(root, module) if module else root


def weights_dir(step_root: str, module: str = "") -> str:
    """DCP directory holding the weights alone."""
    return os.path.join(model_dir(step_root, module), WEIGHTS_DIRNAME)


def optimizer_dir(step_root: str, module: str = "") -> str:
    """DCP directory holding the optimizer state alone.

    Separate from the weights so a checkpoint can be shipped, archived or
    converted without dragging along state that is roughly twice the size of the
    model and useless outside this run.
    """
    return os.path.join(model_dir(step_root, module), OPTIMIZER_DIRNAME)


def lr_scheduler_path(step_root: str, module: str = "") -> str:
    """Replicated scheduler pickle, beside the two DCP directories."""
    return os.path.join(model_dir(step_root, module), LR_SCHEDULER_FILENAME)


def loader_path(step_root: str, rank: int) -> str:
    """Rank-local dataloader cursor. Not nested per module: one job, one cursor."""
    return os.path.join(step_root, LOADER_DIRNAME, RANK_FILE_FORMAT.format(rank))


def extra_state_path(step_root: str, rank: int) -> str:
    """Rank-local job state: step counter, RNG, meters."""
    return os.path.join(step_root, EXTRA_STATE_DIRNAME, RANK_FILE_FORMAT.format(rank))


def hf_export_dir(step_root: str, module: str = "") -> str:
    """Full-model safetensors export."""
    root = os.path.join(step_root, HF_EXPORT_DIRNAME)
    return os.path.join(root, module) if module else root


def lora_export_dir(step_root: str, module: str = "") -> str:
    """PEFT-format adapter export.

    Separate from :func:`hf_export_dir` rather than sharing it: the two are
    mutually exclusive today, but a LoRA merge writes both for the same step and
    a reader has to tell them apart by path.
    """
    root = os.path.join(step_root, LORA_EXPORT_DIRNAME)
    return os.path.join(root, module) if module else root


def manifest_path(step_root: str) -> str:
    return os.path.join(step_root, MANIFEST_FILENAME)


def write_manifest(
    step_root: str,
    global_step: int,
    world_size: int,
    modules: Sequence[str] | None = None,
) -> str:
    """Publish the step as complete. Rank 0 only, and last.

    DCP's ``.metadata`` marks one *directory* complete and there are at least two
    per module — weights and optimizer — plus the per-rank cursor files, which
    DCP does not know about at all. No single ``.metadata`` can therefore stand
    for the step, which is what this file is for.
    """
    payload = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "global_step": global_step,
        "world_size": world_size,
        "modules": list(modules or []),
    }
    path = manifest_path(step_root)
    os.makedirs(step_root, exist_ok=True)
    # Write-then-rename: a reader must never see a half-written marker, since the
    # whole point of the file is that its presence means "complete".
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)
    return path


def read_manifest(step_root: str) -> dict[str, Any] | None:
    """Manifest contents, or ``None`` when the step is not published."""
    path = manifest_path(step_root)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


__all__ = [
    "EXTRA_STATE_DIRNAME",
    "HF_EXPORT_DIRNAME",
    "LORA_EXPORT_DIRNAME",
    "LOADER_DIRNAME",
    "LR_SCHEDULER_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_FORMAT_VERSION",
    "MODEL_DIRNAME",
    "OPTIMIZER_DIRNAME",
    "RANK_FILE_FORMAT",
    "WEIGHTS_DIRNAME",
    "extra_state_path",
    "hf_export_dir",
    "loader_path",
    "lora_export_dir",
    "lr_scheduler_path",
    "manifest_path",
    "model_dir",
    "optimizer_dir",
    "read_manifest",
    "step_dir",
    "weights_dir",
    "write_manifest",
]
