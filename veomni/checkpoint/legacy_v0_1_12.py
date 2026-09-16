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

"""Resume checkpoints written before the current on-disk layout.

**Delete this file** (and the imports that load it) to drop that compatibility.
Call sites:

* ``DistributedCheckpointer._load_lr_scheduler`` — the scheduler pickle
* ``GlobalStateCallback.load_global_state`` — the job cursor
* ``DistributedCheckpointer._remove_dcp_markers`` — deletes the old marker when
  a legacy step is overwritten
* ``_validate_dcp_checkpoint_entry`` — accepts a legacy step for resume

On-disk contract and removal notes: ``docs/usage/checkpoint.md``.

Two shapes are read here, both per-rank pickles and neither written any more:

* **0.1.12** — ``{step_dir}/extra_state/extra_state_rank_{R}.pt``, a dict holding
  ``lr_scheduler`` *and* the job cursor (``global_step``, dataloader, RNG,
  meters) together.
* **0.2.x flat** — the same ``extra_state`` file reduced to
  ``{"lr_scheduler": ...}``, with the cursor moved out to
  ``{step_dir}/trainer_state_rank_{R}.pt``.

The current layout splits that cursor again, into ``loader/rank_{R}.pt`` and
``extra_state/rank_{R}.pt``. Note the directory name ``extra_state/`` is reused
with different contents; what tells the layouts apart is the file name inside it
(``rank_{R}.pt`` vs ``extra_state_rank_{R}.pt``), not the directory.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist

from ..utils import logging


logger = logging.get_logger(__name__)

_EXTRA_STATE_DIR = "extra_state"
_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"
_TRAINER_STATE_FORMAT = "trainer_state_rank_{}.pt"
_MARKER_FILENAME = ".metadata"


def extra_state_path(checkpoint_dir: str, rank: int) -> str:
    return os.path.join(checkpoint_dir, _EXTRA_STATE_DIR, _EXTRA_STATE_FORMAT.format(rank))


def marker_path(step_root: str) -> str:
    """Where a pre-split checkpoint carried its completion marker.

    DCP writes ``.metadata`` into the directory it owns, and in these layouts
    that directory *was* the step root. The current layout keeps its copies
    inside ``model/``, so a file at this path can only have come from an older
    VeOmni -- which is what makes it safe to read as "this step is legacy", and
    safe to delete when the step is being rewritten.

    A path, not a probe: discovery walks HDFS as well as local disks and brings
    its own filesystem calls.
    """
    return os.path.join(step_root, _MARKER_FILENAME)


def drop_marker(step_root: str) -> None:
    """Delete a pre-split step's marker, before that step is overwritten.

    Without this a current-layout write over a legacy step leaves the old marker
    in place: the step has no manifest, so it is not complete, yet discovery
    would still accept it through the fallback this module exists for.
    """
    path = marker_path(step_root)
    if os.path.exists(path):
        os.remove(path)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def read_extra_state(checkpoint_dir: str, rank: int) -> dict[str, Any] | None:
    """Load ``extra_state_rank_{rank}.pt``, or ``None`` if the file is absent."""
    path = extra_state_path(checkpoint_dir, rank)
    if not os.path.exists(path):
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(f"legacy extra_state at {path} is {type(blob).__name__}, expected a dict")
    return blob


def apply_legacy_lr_scheduler(checkpoint_dir: str, lr_scheduler: Any) -> bool:
    """Load ``lr_scheduler`` from extra_state next to the DCP shards.

    Returns True if a legacy pickle was found (including ``lr_scheduler=None``).
    Returns False if there is no extra_state file — caller should raise.
    """
    rank = _rank()
    blob = read_extra_state(checkpoint_dir, rank)
    if blob is None and rank != 0:
        blob = read_extra_state(checkpoint_dir, 0)
    if blob is None:
        return False
    if "lr_scheduler" not in blob:
        raise FileNotFoundError(
            f"legacy extra_state at {extra_state_path(checkpoint_dir, rank)} has no lr_scheduler key"
        )

    payload = blob["lr_scheduler"]
    logger.warning_rank0(
        "Loaded lr_scheduler from extra_state/ (VeOmni 0.1.12 layout). "
        "See docs/usage/checkpoint.md; delete veomni/checkpoint/legacy_v0_1_12.py to drop this path."
    )
    if payload is not None and lr_scheduler is not None:
        lr_scheduler.load_state_dict(payload)
    return True


def trainer_state_path(load_path: str, rank: int) -> str:
    return os.path.join(load_path, _TRAINER_STATE_FORMAT.format(rank))


def apply_legacy_global_state(load_path: str, rank: int) -> dict[str, Any] | None:
    """Job cursor from a pre-split checkpoint, or None if there is none.

    Tries the 0.2.x flat cursor first: it is the more recent of the two, and a
    checkpoint that has one also has an ``extra_state`` pickle beside it holding
    nothing but the scheduler, which would otherwise be mistaken for a cursor.
    """
    flat_path = trainer_state_path(load_path, rank)
    if os.path.exists(flat_path):
        logger.warning_rank0(
            f"Loaded job cursor from {_TRAINER_STATE_FORMAT.format(rank)} (VeOmni 0.2.x flat layout). "
            "See docs/usage/checkpoint.md; delete veomni/checkpoint/legacy_v0_1_12.py to drop this path."
        )
        return torch.load(flat_path, map_location="cpu", weights_only=False)

    blob = read_extra_state(load_path, rank)
    if blob is None or "global_step" not in blob:
        return None

    logger.warning_rank0(
        "Loaded job cursor from extra_state/ (VeOmni 0.1.12 layout). "
        "See docs/usage/checkpoint.md; delete veomni/checkpoint/legacy_v0_1_12.py to drop this path."
    )
    return {
        "global_step": blob["global_step"],
        "train_dataloader": blob.get("train_dataloader"),
        "environ_meter": blob.get("environ_meter") or {},
        "channel_loss_callback": blob.get("channel_loss_callback"),
        "torch_rng_state": blob.get("torch_rng_state"),
    }


__all__ = [
    "apply_legacy_global_state",
    "apply_legacy_lr_scheduler",
    "drop_marker",
    "extra_state_path",
    "marker_path",
    "read_extra_state",
    "trainer_state_path",
]
