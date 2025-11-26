# tests/utils.py
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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

    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        backend = "nccl"
        torch.cuda.set_device(rank)
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
    if torch.cuda.is_available() and torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs")

    port = find_free_port()

    mp.spawn(_dist_worker_entry, args=(world_size, port, func, args, kwargs), nprocs=world_size, join=True)
