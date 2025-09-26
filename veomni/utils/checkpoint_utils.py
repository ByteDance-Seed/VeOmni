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


import os

import torch.distributed as dist


try:
    from hdfs_io import copy, exists, isdir, listdir
except ImportError:
    from .hdfs_io import copy, exists, isdir, listdir


_GLOBAL_STEP_PREFIX = "global_step_"


def _validate_dcp_checkpoint_entry(checkpoints_dir: str, entry: str):
    """Return the checkpoint step if the entry is a valid DCP checkpoint, otherwise None."""
    if not entry.startswith(_GLOBAL_STEP_PREFIX):
        return None

    step_str = entry[len(_GLOBAL_STEP_PREFIX) :]
    try:
        step = int(step_str)
    except ValueError:
        return None

    checkpoint_path = os.path.join(checkpoints_dir, entry)
    if not isdir(checkpoint_path):
        return None

    model_metadata_path = os.path.join(checkpoint_path, "model/.metadata")
    optim_metadata_path = os.path.join(checkpoint_path, "optimizer/.metadata")
    if not exists(model_metadata_path) or not optim_metadata_path:
        return None

    return step


def get_last_iteration(output_dir, is_rank0: bool):
    meta_file = "latest_checkpointed_iteration.txt"
    if is_rank0:
        latest_file = os.path.join(output_dir, "checkpoints", meta_file)
        if exists(latest_file):
            copy(latest_file, meta_file)

    dist.barrier()
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            iteration = int(f.readline())
    else:
        iteration = 0

    dist.barrier()
    if is_rank0:
        if os.path.exists(meta_file):
            os.remove(meta_file)

    return iteration


def dcp_get_last_iteration(output_dir):
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    if not exists(checkpoints_dir):
        return None

    entries = listdir(checkpoints_dir)
    valid_steps = []
    for entry in entries:
        step = _validate_dcp_checkpoint_entry(checkpoints_dir, entry)
        if step is not None:
            valid_steps.append(step)

    if not valid_steps:
        return None

    return max(valid_steps)


def get_checkpoint_path(output_dir, is_rank0: bool, ckpt_manager: str):
    if ckpt_manager == "dcp":
        iteration = dcp_get_last_iteration(output_dir)
    else:  # OmniStore or BCP
        iteration = get_last_iteration(output_dir, is_rank0)

    if not iteration:
        return None

    checkpoint_path = os.path.join(output_dir, "checkpoints", f"global_step_{iteration}")

    return checkpoint_path
