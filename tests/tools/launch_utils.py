# tests/utils.py
import os

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.testing import find_free_port
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


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
