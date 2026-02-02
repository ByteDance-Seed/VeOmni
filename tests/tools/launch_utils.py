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

# tests/utils.py
import os
import socket

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _dist_worker_entry(rank, world_size, port, func, args, kwargs):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if get_torch_device().is_available() and get_torch_device().device_count() >= world_size:
        backend = get_dist_comm_backend()
        get_torch_device().set_device(rank)
    else:
        backend = "gloo"

    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        func(*args, **kwargs)
    except Exception as e:
        raise e
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def torchrun(func, world_size: int = 4, *args, **kwargs):
    if get_torch_device().is_available() and get_torch_device().is_available().device_count() < world_size:
        pytest.skip(f"Requires {world_size} {get_device_type()} devices")

    port = find_free_port()

    mp.spawn(_dist_worker_entry, args=(world_size, port, func, args, kwargs), nprocs=world_size, join=True)
