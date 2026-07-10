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

from contextvars import ContextVar
from typing import Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup


# Tests that exercise the communication primitives without constructing a
# model may inject a Ulysses group. Production always resolves groups from the
# ParallelState active for the model forward.
_UNSET_TEST_GROUP = object()
_ULYSSES_SP_GROUP_OVERRIDE: ContextVar[object] = ContextVar("veomni_ulysses_test_group", default=_UNSET_TEST_GROUP)


def _current_parallel_state():
    # Lazy import avoids the parallel_state -> sequence_parallel import cycle.
    from ..parallel_state import get_current_parallel_state

    return get_current_parallel_state()


def _active_group(name: str) -> Optional[ProcessGroup]:
    state = _current_parallel_state()
    if state is None:
        raise RuntimeError(f"Cannot resolve {name} without an active model-owned ParallelState.")
    return getattr(state, name, None)


def get_data_parallel_group() -> Optional[ProcessGroup]:
    return _active_group("dp_group")


def get_data_parallel_rank() -> int:
    group = get_data_parallel_group()
    return dist.get_rank(group) if group is not None else 0


def get_data_parallel_world_size() -> int:
    group = get_data_parallel_group()
    return dist.get_world_size(group) if group is not None else 1


def set_ulysses_sequence_parallel_group(group: Optional[ProcessGroup]) -> None:
    """Install the group used by standalone communication unit tests."""
    _ULYSSES_SP_GROUP_OVERRIDE.set(group)


def get_ulysses_sequence_parallel_group() -> Optional[ProcessGroup]:
    state = _current_parallel_state()
    if state is not None:
        return state.ulysses_group
    test_group = _ULYSSES_SP_GROUP_OVERRIDE.get()
    if test_group is _UNSET_TEST_GROUP:
        raise RuntimeError("Cannot resolve ulysses_group without an active model-owned ParallelState.")
    return test_group


def get_ulysses_sequence_parallel_cpu_group() -> Optional[ProcessGroup]:
    return _active_group("ulysses_cpu_group")


def get_ulysses_sequence_parallel_rank(group: Optional[ProcessGroup] = None) -> int:
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_rank(group) if group is not None else 0


def get_ulysses_sequence_parallel_world_size(group: Optional[ProcessGroup] = None) -> int:
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group is not None else 1


def get_context_parallel_group(check_initialized: bool = True) -> Optional[ProcessGroup]:
    group = _active_group("cp_group")
    if check_initialized:
        assert group is not None, "context parallel group is not initialized"
    return group


def get_context_parallel_rank() -> int:
    group = get_context_parallel_group(check_initialized=False)
    return dist.get_rank(group) if group is not None else 0


def get_context_parallel_world_size() -> int:
    group = get_context_parallel_group(check_initialized=False)
    return dist.get_world_size(group) if group is not None else 1


def get_unified_sequence_parallel_group() -> Optional[ProcessGroup]:
    return _active_group("sp_group")


def get_unified_sequence_parallel_cpu_group() -> Optional[ProcessGroup]:
    return _active_group("sp_cpu_group")


def get_unified_sequence_parallel_rank() -> int:
    group = get_unified_sequence_parallel_group()
    return dist.get_rank(group) if group is not None else 0


def get_unified_sequence_parallel_world_size() -> int:
    group = get_unified_sequence_parallel_group()
    return dist.get_world_size(group) if group is not None else 1


def is_ulysses_sequence_parallel_initialized() -> bool:
    return get_ulysses_sequence_parallel_group() is not None


def is_context_parallel_initialized() -> bool:
    return get_context_parallel_group(check_initialized=False) is not None
