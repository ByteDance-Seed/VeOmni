"""Local-sequence execution for the source-derived model.

VeOmni owns data/FSDP parallelism. Source-specific sequence parallelism is not
supported by this baseline and is rejected before any identity operation.
"""

import torch


def require_local_sequence():
    from veomni.distributed.parallel_state import get_parallel_state

    if get_parallel_state().sp_enabled:
        raise NotImplementedError("SeedVR2 currently requires ulysses_size=cp_size=1.")


def slice_inputs(tensor, **kwargs):
    require_local_sequence()
    return tensor


def gather_outputs(tensor, **kwargs):
    require_local_sequence()
    return tensor


def gather_heads_scatter_seq(tensor, **kwargs):
    require_local_sequence()
    return tensor


def gather_seq_scatter_heads_qkv(tensor, **kwargs):
    require_local_sequence()
    return tensor


def get_sequence_parallel_world_size():
    require_local_sequence()
    return 1


def get_sequence_parallel_rank():
    require_local_sequence()
    return 0


def get_sequence_parallel_group():
    require_local_sequence()
    return None


def get_next_sequence_parallel_rank():
    require_local_sequence()
    return 0


def get_prev_sequence_parallel_rank():
    require_local_sequence()
    return 0


def get_device():
    from veomni.utils.device import get_device_type

    return torch.device(get_device_type())


class Gather:
    @staticmethod
    def apply(*args, **kwargs):
        raise NotImplementedError("SeedVR2 sequence-parallel communication is not implemented.")
