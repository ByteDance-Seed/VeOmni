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


# Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Tuple, Union

from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from ..utils import logging
from ..utils.device import get_device_type


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.get_logger(__name__)

_PARALLEL_STATE: "ParallelState" = None

# Cache of built parallel states keyed by full topology.
_PARALLEL_STATE_CACHE: Dict[tuple, "ParallelState"] = {}

# Named registry of parallel states (independent of the topology cache).
# This round every trainer registers under ``"base"``; multi-module names later.
_PARALLEL_STATE_REGISTRY: Dict[str, "ParallelState"] = {}


def requires_mesh(fn: Callable) -> Callable:
    @wraps(fn)
    def _inner(self: "ParallelState", *args, **kwargs):
        if self.device_mesh is None:
            raise ValueError("Device mesh is not initialized.")

        return fn(self, *args, **kwargs)

    return _inner


@dataclass(frozen=True)
class ParallelState:
    dp_size: int = 1
    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ulysses_size: int = 1
    dp_mode: Literal["ddp", "fsdp2"] = "fsdp2"
    device_type: str = get_device_type()
    include_sp_in_fsdp: bool = True
    device_mesh: Optional["DeviceMesh"] = None
    extra_parallel_names: Tuple[str] = ("ep",)
    extra_parallel_sizes: Dict[str, int] = field(default_factory=lambda: {"ep": 1})
    extra_parallel_fsdp_device_mesh: Dict[str, Optional["DeviceMesh"]] = field(default_factory=lambda: {"ep": None})
    async_enabled: Optional[bool] = False

    def __post_init__(self):
        if not self.include_sp_in_fsdp:
            raise NotImplementedError("Decoupled sequence parallel has not been implemented.")

        if self.cp_size > 1:
            raise NotImplementedError("Ring attention is not supported yet.")

        if self.pp_size * self.dp_size * self.cp_size * self.ulysses_size * self.tp_size != self.world_size:
            raise ValueError("The product of parallel sizes should be equal to the world size.")

        if self.dp_replicate_size * self.dp_shard_size != self.dp_size:
            raise ValueError(
                f"The product of dp_replicate_size: {self.dp_replicate_size} and dp_shard_size: {self.dp_shard_size} should be equal to dp_size: {self.dp_size}."
            )

        # SP / DP / CP process groups are NOT cached in module-level globals.
        # Every ``sequence_parallel.comm`` getter resolves its group from the
        # *current* ParallelState's device mesh (see ``dp_group`` / ``sp_group``
        # / ``ulysses_group`` / ``cp_group`` below), so a per-module forward
        # scoped by ``use_parallel_state`` automatically gets its own groups —
        # even when sibling Omni modules run at different SP sizes. Meshless SP
        # is therefore unsupported (a bare ``ParallelState(ulysses_size>1)`` with
        # no mesh has no group to resolve).
        if self.sp_enabled and self.device_mesh is None:
            raise ValueError(
                "A sequence-parallel ParallelState must be built with a device mesh "
                "(use init_parallel_state); meshless sequence-parallel init is no longer supported."
            )

    @property
    def is_initialized(self) -> bool:
        return dist.is_initialized()

    @property
    def local_rank(self) -> int:
        return int(os.getenv("LOCAL_RANK", "-1"))

    @property
    def global_rank(self) -> int:
        if self.is_initialized:
            return dist.get_rank()
        return -1

    @property
    def world_size(self) -> int:
        if self.is_initialized:
            return dist.get_world_size()
        return 1

    # ------------------------------ DP ------------------------------ #
    @property
    def dp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp")

        return None

    @property
    def dp_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp")

        return self.fsdp_rank

    @property
    @requires_mesh
    def dp_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp"]

        raise self.fsdp_mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1

    # ------------------------------ DP replicate ------------------------------ #
    @property
    def dp_replicate_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_replicate")

    @property
    def dp_replicate_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp_replicate")

    @property
    @requires_mesh
    def dp_replicate_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp_replicate"]

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate_size > 1

    # ------------------------------ DP shard ------------------------------ #
    @property
    def dp_shard_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_shard")

    @property
    def dp_shard_sp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_shard_sp")

    @property
    def dp_shard_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp_shard")

    @property
    @requires_mesh
    def dp_shard_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp_shard"]

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard_size >= 1

    # ----------------------------- FSDP ----------------------------- #
    @property
    def fsdp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_sp")

    @property
    def fsdp_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp_sp")

        return self.global_rank

    @property
    def dp_shard_sp_enabled(self) -> bool:
        return self.dp_shard_enabled and self.sp_enabled

    @property
    @requires_mesh
    def fsdp_mesh(self) -> "DeviceMesh":
        if self.dp_replicate_enabled:
            # HSDP
            if self.dp_shard_sp_enabled:
                return self.device_mesh["dp_replicate", "dp_shard_sp"]
            elif self.dp_shard_enabled:
                return self.device_mesh["dp_replicate", "dp_shard"]
            else:
                # DDP
                return self.device_mesh["dp_replicate"]
        # FSDP
        elif self.dp_shard_sp_enabled:
            return self.device_mesh["dp_shard_sp"]
        elif self.dp_shard_enabled:
            return self.device_mesh["dp_shard"]
        else:
            return self.device_mesh["dp"]

    @property
    def fsdp_enabled(self) -> bool:
        return self.fsdp_size > 1

    @property
    def fsdp_size(self) -> int:
        return self.world_size // (self.pp_size * self.tp_size)

    # ------------------------------ TP ------------------------------ #
    @property
    @requires_mesh
    def tp_rank(self) -> int:
        return self.device_mesh.get_local_rank("tp")

    @property
    @requires_mesh
    def tp_mesh(self) -> "DeviceMesh":
        return self.device_mesh["tp"]

    @property
    def tp_enabled(self) -> bool:
        return self.tp_size > 1

    # ------------------------------ PP ------------------------------ #
    @property
    @requires_mesh
    def pp_rank(self) -> int:
        return self.device_mesh.get_local_rank("pp")

    @property
    @requires_mesh
    def pp_mesh(self) -> "DeviceMesh":
        return self.device_mesh["pp"]

    @property
    def pp_enabled(self) -> bool:
        return self.pp_size > 1

    @property
    @requires_mesh
    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    @requires_mesh
    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == (self.pp_size - 1)

    # ------------------------------ EP ------------------------------ #
    @property
    @requires_mesh
    def ep_mesh(self) -> "DeviceMesh":
        return self.extra_parallel_mesh("ep")

    @property
    @requires_mesh
    def ep_fsdp_mesh(self) -> "DeviceMesh":
        return self.extra_parallel_fsdp_mesh("ep")

    @cached_property
    def ep_group(self) -> "ProcessGroup":
        return self.extra_parallel_group("ep")

    @property
    def ep_enabled(self) -> bool:
        return self.extra_parallel_enabled("ep")

    @property
    def ep_size(self) -> int:
        return self.extra_parallel_sizes["ep"]

    @property
    def ep_rank(self) -> int:
        return self.extra_parallel_rank("ep")

    @property
    def ep_fsdp_size(self) -> int:
        return self.extra_parallel_fsdp_size("ep")

    @property
    def ep_gradient_divide_factor(self) -> int:
        return self.extra_parallel_gradient_divide_factor("ep")

    # ------------------------------ Parallel list ------------------------------ #
    @requires_mesh
    def extra_parallel_mesh(self, para_name) -> "DeviceMesh":
        return self.extra_parallel_fsdp_device_mesh[para_name][para_name]

    @requires_mesh
    def extra_parallel_fsdp_mesh(self, para_name) -> "DeviceMesh":
        return self.extra_parallel_fsdp_device_mesh[para_name][para_name, f"{para_name}_fsdp"]

    @requires_mesh
    def extra_parallel_group(self, para_name) -> "ProcessGroup":
        if self.extra_parallel_enabled(para_name):
            return self.extra_parallel_mesh(para_name).get_group()
        else:
            return None

    def extra_parallel_enabled(self, para_name) -> bool:
        return self.extra_parallel_sizes[para_name] > 1

    def extra_parallel_rank(self, para_name) -> int:
        return self.extra_parallel_fsdp_device_mesh[para_name].get_local_rank(para_name)

    def extra_parallel_fsdp_size(self, para_name) -> int:
        assert self.extra_parallel_enabled(para_name), (
            f"{para_name}_fsdp_size is only available when {para_name} is enabled ({para_name}_size > 1)"
        )
        return self.fsdp_size // self.extra_parallel_sizes[para_name]

    def extra_parallel_gradient_divide_factor(self, para_name) -> int:
        # We assume the world size is the total dp size by now
        # TP and PP would make this assumption not true
        assert self.tp_size == 1
        assert self.pp_size == 1
        # For ep+fsdp2, the grad divide factor should alwasy be world size (no matter HSDP or not)
        # SP does not affect this since SP groups still replicate params
        # and their grads are all-reduced which would match grads for the same data without SP.
        return self.world_size

    @property
    def any_extra_parallel_enabled(self) -> bool:
        return any(self.extra_parallel_enabled(para_name) for para_name in self.extra_parallel_names)

    # ------------------------------ SP ------------------------------ #
    @property
    def sp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None and self.sp_enabled:
            return self.device_mesh.get_group("sp")

        return None

    @property
    def sp_rank(self) -> int:
        if self.device_mesh is not None and self.sp_enabled:
            return self.device_mesh.get_local_rank("sp")

        return -1

    @property
    def sp_enabled(self) -> bool:
        return self.cp_size > 1 or self.ulysses_size > 1

    @property
    def sp_size(self) -> int:
        return self.ulysses_size * self.cp_size

    @property
    def ulysses_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None and self.ulysses_enabled:
            return self.device_mesh.get_group("ulysses")

        return None

    @property
    def ulysses_rank(self) -> int:
        if self.device_mesh is not None and self.ulysses_enabled:
            return self.device_mesh.get_local_rank("ulysses")

        return -1

    @property
    def ulysses_enabled(self) -> bool:
        return self.ulysses_size > 1

    @property
    def cp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None and self.cp_enabled:
            return self.device_mesh.get_group("cp")

        return None

    @property
    def cp_rank(self) -> int:
        if self.device_mesh is not None and self.cp_enabled:
            return self.device_mesh.get_local_rank("cp")

        return -1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_size > 1


def clear_parallel_state() -> None:
    """
    Drop the ambient state, topology cache, and named registry.

    Call after ``destroy_process_group()`` (or in test teardown) so a later
    ``init_parallel_state`` with the same topology cannot reuse DeviceMesh /
    process groups from a destroyed distributed session.
    """
    global _PARALLEL_STATE
    _PARALLEL_STATE = None
    _PARALLEL_STATE_CACHE.clear()
    _PARALLEL_STATE_REGISTRY.clear()


def get_parallel_state_by_name(name: str) -> "ParallelState":
    """Look up a registered parallel state by module name."""
    if name not in _PARALLEL_STATE_REGISTRY:
        registered = sorted(_PARALLEL_STATE_REGISTRY)
        raise ValueError(f"Parallel state {name!r} is not registered. Registered names: {registered}")
    return _PARALLEL_STATE_REGISTRY[name]


def init_parallel_state(
    dp_size: int = 1,
    dp_replicate_size: int = 1,
    dp_shard_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ulysses_size: int = 1,
    dp_mode: Literal["ddp", "fsdp2"] = "fsdp2",
    device_type: str = None,
    include_sp_in_fsdp: bool = True,
    extra_parallel_sizes: Tuple[int] = (1,),
    extra_parallel_placement_innermost: Tuple[bool] = (False,),
    extra_parallel_names: Tuple[str] = ("ep",),
    async_enabled: Optional[bool] = False,
    name: str = "base",
) -> "ParallelState":
    """
    Initialize a parallel state, register it under ``name``, and set it as the
    global state when none is current yet.

    If ``name`` is already registered, log a warning and return the existing
    state without building, caching, or overwriting anything.
    """
    global _PARALLEL_STATE

    if name in _PARALLEL_STATE_REGISTRY:
        logger.warning(
            f"Parallel state {name!r} is already registered; returning the existing state without rebuilding."
        )
        return _PARALLEL_STATE_REGISTRY[name]

    if _PARALLEL_STATE is not None:
        logger.warning("Parallel state has already been initialized.")

    if device_type is None:
        device_type = get_device_type()

    # Set dp_shard_size to dp_size if dp_shard_size and dp_replicate_size are not set when dp enabled
    if dp_size > 1 and dp_shard_size == 1 and dp_replicate_size == 1:
        dp_shard_size = dp_size

    extra_parallel_sizes = tuple(extra_parallel_sizes)
    extra_parallel_placement_innermost = tuple(extra_parallel_placement_innermost)
    extra_parallel_names = tuple(extra_parallel_names)

    # Note that Expert Parallel is included into Extra Parallel
    assert len(extra_parallel_sizes) == len(extra_parallel_placement_innermost) == len(extra_parallel_names), (
        "each extra parallel should correspond to a size, a placement and a name"
    )

    # Reuse an already-built state for an identical topology (e.g. the omni
    # orchestrator builds one ParallelState per module, many of which share the
    # global topology). Building the device mesh / extra-parallel meshes creates
    # process groups via collectives — all ranks run the same call sequence in
    # the same order, so cache hits/misses are rank-consistent and no duplicate
    # groups are created. The key spans every field that shapes the mesh or the
    # state's behaviour (e.g. ``dp_mode`` is read by grad-clip).
    cache_key = (
        dp_size,
        dp_replicate_size,
        dp_shard_size,
        tp_size,
        pp_size,
        cp_size,
        ulysses_size,
        dp_mode,
        device_type,
        include_sp_in_fsdp,
        extra_parallel_sizes,
        extra_parallel_placement_innermost,
        extra_parallel_names,
        async_enabled,
    )
    cached_state = _PARALLEL_STATE_CACHE.get(cache_key)
    if cached_state is not None:
        logger.info_rank0("Reusing cached parallel state for identical topology.")
        # Mirror the build path below: only (re)establish the default global if it
        # has not been set yet. The cache is a module-level global that outlives the
        # global state (e.g. tests reset ``_PARALLEL_STATE = None`` at teardown but
        # never clear the cache), so a same-topology hit may find the global cleared.
        if _PARALLEL_STATE is None:
            _PARALLEL_STATE = cached_state
        _PARALLEL_STATE_REGISTRY[name] = cached_state
        return cached_state

    logger.info_rank0(
        f"Initializing parallel state: dp_size {dp_size}, dp_replicate_size {dp_replicate_size}, "
        + f"dp_shard_size {dp_shard_size},tp_size {tp_size}, pp_size {pp_size}, cp_size {cp_size}, ulysses_size {ulysses_size}, "
        + ", ".join(
            [
                f"{para_name}_size {para_size}"
                for para_name, para_size in zip(extra_parallel_names, extra_parallel_sizes)
            ]
        )
    )

    device_mesh = None

    extra_parallel_fsdp_device_mesh = {f"{para_name}": None for para_name in extra_parallel_names}

    mesh_shape = []
    mesh_dim_names = []
    for d, dim_name in zip(
        [pp_size, dp_replicate_size, dp_shard_size, ulysses_size, cp_size, tp_size],
        ["pp", "dp_replicate", "dp_shard", "ulysses", "cp", "tp"],
    ):
        if d > 1 or dim_name in ["dp_shard"]:
            mesh_shape.append(d)
            mesh_dim_names.append(dim_name)

    device_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=tuple(mesh_shape),
        mesh_dim_names=tuple(mesh_dim_names),
    )

    # Mesh for data loading (no communication on this mesh)
    dp_mesh_dim_names = []
    # Mesh for param sharding
    dp_shard_sp_mesh_dim_names = []
    # Mesh for loss all-reduce
    dp_sp_mesh_dim_names = []
    # Mesh for sequence parallel
    sp_mesh_dim_names = []

    if dp_replicate_size > 1:
        dp_mesh_dim_names.append("dp_replicate")
        dp_sp_mesh_dim_names.append("dp_replicate")
    if dp_shard_size >= 1:
        dp_mesh_dim_names.append("dp_shard")
        dp_shard_sp_mesh_dim_names.append("dp_shard")
        dp_sp_mesh_dim_names.append("dp_shard")
    if ulysses_size > 1:
        dp_shard_sp_mesh_dim_names.append("ulysses")
        sp_mesh_dim_names.append("ulysses")
        dp_sp_mesh_dim_names.append("ulysses")
    if cp_size > 1:
        dp_shard_sp_mesh_dim_names.append("cp")
        sp_mesh_dim_names.append("cp")
        dp_sp_mesh_dim_names.append("cp")

    if dp_mesh_dim_names != []:
        device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")

    if dp_shard_sp_mesh_dim_names != []:
        device_mesh[tuple(dp_shard_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_sp")

    if dp_sp_mesh_dim_names != []:
        device_mesh[tuple(dp_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_sp")

    if sp_mesh_dim_names != []:
        device_mesh[tuple(sp_mesh_dim_names)]._flatten(mesh_dim_name="sp")

    for para_size, para_outside, para_name in zip(
        extra_parallel_sizes, extra_parallel_placement_innermost, extra_parallel_names
    ):
        if para_size > 1:
            # TODO: drop para_outside?
            assert not para_outside, f"{para_name} is not supported when para_outside is True."

            # NOTE: Support HSDP for extra parallel. For example, world_size=1024
            # - dense param device_mesh: (dp_replicate, dp_shard_sp)=(4, 256)
            # - ep_size=8, expert parallel device_mesh: (ep_replicate, ep_fsdp, ep)=(4, 32, 8)
            # Note that ep_size should be a factor of dp_shard_sp_size.
            param_mesh_shape, para_mesh_dim_names = [], []
            if dp_replicate_size > 1:
                param_mesh_shape.append(dp_replicate_size)
                para_mesh_dim_names.append(f"{para_name}_replicate")
            dp_shard_sp_size = device_mesh["dp_shard_sp"].size()
            assert dp_shard_sp_size % para_size == 0, (
                f"{para_name}_size({para_size}) must be a factor of dp_shard_sp_size({dp_shard_sp_size})"
            )
            para_fsdp_size = dp_shard_sp_size // para_size
            param_mesh_shape.append(para_fsdp_size)
            param_mesh_shape.append(para_size)
            para_mesh_dim_names.append(f"{para_name}_fsdp")
            para_mesh_dim_names.append(para_name)

            extra_parallel_fsdp_device_mesh[f"{para_name}"] = init_device_mesh(
                device_type=device_type,
                mesh_shape=param_mesh_shape,
                mesh_dim_names=para_mesh_dim_names,
            )

    logger.info_rank0(f"Device mesh: {device_mesh}")
    for para_name in extra_parallel_names:
        logger.info_rank0(f"{para_name} FSDP device mesh: {extra_parallel_fsdp_device_mesh[para_name]}")

    parallel_state = ParallelState(
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ulysses_size=ulysses_size,
        dp_mode=dp_mode,
        device_type=device_type,
        include_sp_in_fsdp=include_sp_in_fsdp,
        device_mesh=device_mesh,
        extra_parallel_names=extra_parallel_names,
        extra_parallel_sizes=dict(zip(extra_parallel_names, extra_parallel_sizes)),
        extra_parallel_fsdp_device_mesh=extra_parallel_fsdp_device_mesh,
        async_enabled=async_enabled,
    )

    if _PARALLEL_STATE is None:
        _PARALLEL_STATE = parallel_state

    _PARALLEL_STATE_CACHE[cache_key] = parallel_state
    _PARALLEL_STATE_REGISTRY[name] = parallel_state
    return parallel_state


def set_parallel_state(parallel_state: "ParallelState") -> Optional["ParallelState"]:
    """
    Set the global parallel state to ``parallel_state``; returns the previous one.

    The SP / DP / CP process-group getters in ``sequence_parallel.comm`` resolve
    from whatever state is current here, so an Omni module's forward — scoped by
    :func:`use_parallel_state` — automatically runs its collectives over its own
    groups, even when sibling modules use a different SP size.
    """
    global _PARALLEL_STATE
    old = _PARALLEL_STATE
    _PARALLEL_STATE = parallel_state
    return old


@contextmanager
def use_parallel_state(parallel_state: Union[str, "ParallelState"]):
    """
    Temporarily make ``parallel_state`` the global parallel state, restoring on exit.

    ``parallel_state`` may be a registered module name (``str``) or a
    ``ParallelState`` object. The SP / DP / CP group getters resolve from the
    current state, so a forward run inside this scope uses that state's own
    groups — even when sibling Omni modules use a different SP size.
    """
    if isinstance(parallel_state, str):
        parallel_state = get_parallel_state_by_name(parallel_state)
    old = set_parallel_state(parallel_state)
    try:
        yield
    finally:
        set_parallel_state(old)


def get_parallel_state() -> "ParallelState":
    """
    Returns global parallel state.
    """
    if _PARALLEL_STATE is None:
        logger.warning_once("Parallel state has not been initialized. returning default Single-process state.")
        return ParallelState()

    return _PARALLEL_STATE
