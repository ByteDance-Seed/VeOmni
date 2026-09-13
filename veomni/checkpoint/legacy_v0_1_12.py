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

"""Resume checkpoints written by VeOmni 0.1.12 through the extra_state layout.

**Delete this file** (and the two imports that load it) to drop that
compatibility. Call sites:

* ``DistributedCheckpointer._load_lr_scheduler``
* ``GlobalStateCallback.load_global_state``

On-disk contract and removal notes: ``docs/usage/checkpoint.md``.

0.1.12 stored a per-rank pickle at
``{step_dir}/extra_state/extra_state_rank_{R}.pt`` whose dict always had
``lr_scheduler`` and the job cursor (``global_step``, dataloader, RNG, meters).
After GlobalStateCallback split, the same path held only
``{"lr_scheduler": ...}`` and the cursor moved to ``trainer_state_rank_{R}.pt``.
This module reads those pickles; it does not write them.
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


def extra_state_path(checkpoint_dir: str, rank: int) -> str:
    return os.path.join(checkpoint_dir, _EXTRA_STATE_DIR, _EXTRA_STATE_FORMAT.format(rank))


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


def apply_legacy_global_state(load_path: str, rank: int) -> dict[str, Any] | None:
    """Job cursor from a 0.1.12 mixed extra_state pickle, or None.

    Post-split extra_state has no ``global_step``; those runs already write
    ``trainer_state_rank_{R}.pt`` and this function stays out of the way.
    """
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
    "extra_state_path",
    "read_extra_state",
]
