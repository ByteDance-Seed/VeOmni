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
    from hdfs_io import copy, exists
except ImportError:
    from .hdfs_io import copy, exists


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


def get_checkpoint_path(output_dir, is_rank0: bool):
    iteration = get_last_iteration(output_dir, is_rank0)
    if iteration:
        return os.path.join(output_dir, "checkpoints", f"global_step_{iteration}")
