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


from typing import Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup


# Test-only injection seam for the Ulysses group.
#
# The sequence-parallel unit tests (``tests/parallel/ulysses/*``) exercise the
# SP collectives against a raw process group WITHOUT building a ParallelState /
# device mesh, so they set this override directly. In every production path it
# stays ``None`` and the group is resolved from the current ParallelState (see
# ``get_ulysses_sequence_parallel_group``).
_ULYSSES_SP_GROUP_OVERRIDE: Optional[ProcessGroup] = None


def _current_state():
    # Lazy import: parallel_state imports this module, so importing at module
    # scope would be circular. The groups intentionally live on the state's
    # device mesh (like dp/fsdp), not in module-level globals — so a per-module
    # forward scoped by ``use_parallel_state`` resolves its own groups, even
    # when sibling Omni modules run at different SP sizes.
    #
    # Reads the module global directly (not ``get_parallel_state()``) so an
    # UNINITIALIZED process resolves to ``None`` — i.e. "no groups" — instead of
    # constructing a default ``ParallelState`` that validates against the
    # distributed world size and raises. Production always initializes via
    # ``init_parallel_state`` before any SP op runs; this ``None`` path only
    # covers pre-init / unit-test code (which drives SP via the override seam).
    from .. import parallel_state

    return parallel_state._PARALLEL_STATE


# ------------------------------ Data Parallel ------------------------------ #
def get_data_parallel_group() -> Optional[dist.ProcessGroup]:
    """Data-parallel group of the current parallel state."""
    ps = _current_state()
    return ps.dp_group if ps is not None else None


def get_data_parallel_rank() -> int:
    """Data-parallel rank of the current parallel state."""
    return dist.get_rank(get_data_parallel_group())


def get_data_parallel_world_size() -> int:
    """Data-parallel world size of the current parallel state."""
    return dist.get_world_size(get_data_parallel_group())


# ----------------------------- Ulysses Parallel ---------------------------- #
def set_ulysses_sequence_parallel_group(group: Optional[dist.ProcessGroup]):
    """Inject the Ulysses group directly (unit-test seam).

    Production code MUST NOT call this — the group follows the current
    ParallelState. It exists only so the SP unit tests can drive the collectives
    without constructing a device mesh (see ``_ULYSSES_SP_GROUP_OVERRIDE``).
    """
    global _ULYSSES_SP_GROUP_OVERRIDE
    _ULYSSES_SP_GROUP_OVERRIDE = group


def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """Ulysses group of the current parallel state.

    Scoped per-module by ``use_parallel_state``, so heterogeneous per-module SP
    sizes each get their own group with no global key bookkeeping.

    The unit-test override takes precedence when set: the SP unit tests drive the
    collectives without an initialized ``ParallelState`` (so resolving the state
    would build a default one that validates against the world size and raises).
    In production the override stays ``None`` and the group resolves from the
    current state's mesh — identical behaviour.
    """
    if _ULYSSES_SP_GROUP_OVERRIDE is not None:
        return _ULYSSES_SP_GROUP_OVERRIDE
    ps = _current_state()
    return ps.ulysses_group if ps is not None else None


def get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int:
    """
    Get ulysses sequence parallel rank.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0


def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:
    """
    Get ulysses sequence parallel world size.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


# ----------------------------- Context Parallel ---------------------------- #
def get_context_parallel_group(check_initialized=True):
    """Get the context parallel group of the current parallel state."""
    ps = _current_state()
    group = ps.cp_group if ps is not None else None
    if check_initialized:
        assert group is not None, "context parallel group is not initialized"
    return group


def get_context_parallel_rank():
    """Return my rank for the context parallel group."""

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=get_context_parallel_group())
    else:
        return 0


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=get_context_parallel_group())
    else:
        return 0


# ----------------------------- Unified Parallel ---------------------------- #
def get_unified_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """Unified (ulysses × cp) sequence-parallel group of the current state."""
    ps = _current_state()
    return ps.sp_group if ps is not None else None


def get_unified_sequence_parallel_rank() -> int:
    """
    Get unified sequence parallel rank.
    """
    group = get_unified_sequence_parallel_group()
    return dist.get_rank(group) if group else 0


def get_unified_sequence_parallel_world_size() -> int:
    """
    Get unified sequence parallel world size.
    """
    group = get_unified_sequence_parallel_group()
    return dist.get_world_size(group) if group else 1


def is_ulysses_sequence_parallel_initialized() -> bool:
    """
    Check if ulysses sequence parallel is initialized.
    """
    return get_ulysses_sequence_parallel_group() is not None


def is_context_parallel_initialized() -> bool:
    """
    Check if context parallel is initialized.
    """
    return get_context_parallel_group(check_initialized=False) is not None
