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

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtraParallelMeshSpec:
    """Shape and named dimensions for one extra-parallel parameter mesh."""

    shape: tuple[int, ...]
    dim_names: tuple[str, ...]
    fsdp_dim_names: tuple[str, ...]


def build_extra_parallel_mesh_spec(
    *,
    dp_replicate_size: int,
    dp_shard_sp_size: int,
    parallel_size: int,
    parallel_name: str,
    parallel_outside: bool,
) -> ExtraParallelMeshSpec:
    """Build the rank-ordering contract for an extra-parallel mesh.

    DeviceMesh uses row-major rank order, so the rightmost dimension forms
    contiguous groups. ``parallel_outside`` swaps the parallel and FSDP
    dimensions without moving an outer HSDP replicate dimension.
    """

    if min(dp_replicate_size, dp_shard_sp_size, parallel_size) < 1:
        raise ValueError("mesh dimensions must be positive")
    if not parallel_name:
        raise ValueError("parallel_name must not be empty")
    if dp_shard_sp_size % parallel_size != 0:
        raise ValueError(f"{parallel_name}_size({parallel_size}) must divide dp_shard_sp_size({dp_shard_sp_size})")

    parallel_fsdp_size = dp_shard_sp_size // parallel_size
    shape: list[int] = []
    dim_names: list[str] = []

    if dp_replicate_size > 1:
        shape.append(dp_replicate_size)
        dim_names.append(f"{parallel_name}_replicate")

    if parallel_outside:
        shape.extend((parallel_size, parallel_fsdp_size))
        dim_names.extend((parallel_name, f"{parallel_name}_fsdp"))
    else:
        shape.extend((parallel_fsdp_size, parallel_size))
        dim_names.extend((f"{parallel_name}_fsdp", parallel_name))

    return ExtraParallelMeshSpec(
        shape=tuple(shape),
        dim_names=tuple(dim_names),
        fsdp_dim_names=tuple(dim_name for dim_name in dim_names if dim_name != parallel_name),
    )
