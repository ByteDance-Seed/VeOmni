import os
import random

import torch.distributed as dist
from torch.multiprocessing import spawn

from veomni.utils.device import get_nccl_backend, get_torch_device


def torchrun(ngpus, test_fn, *args, **kwargs):
    """
    Usage for pytest

    test_xxx = functools.partial(torchrun, 2, example)
    """
    assert len(kwargs) == 0, "kwargs not supported"
    if ngpus == 1:
        return test_fn(*args)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(random.randint(4300, 4800))
        os.environ["OMNISTORE_LOGGING_LEVEL"] = "ERROR"
        spawn(
            entry_fn,
            args=(ngpus, test_fn, *args),
            nprocs=ngpus,
        )


def entry_fn(rank, world_size, fn, *args, **kwargs):
    dist.init_process_group(backend=get_nccl_backend(), init_method="env://", rank=rank, world_size=world_size)
    get_torch_device().set_device(rank)
    try:
        fn(*args, **kwargs)
    finally:
        dist.destroy_process_group()
