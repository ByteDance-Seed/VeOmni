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

"""Target-centric checkpoint load plan.

A load plan maps every model tensor (parameter or buffer) to the checkpoint
tensor it is loaded from, and to the region of it that this rank holds. It is
built from the safetensors headers alone, so every check runs before any tensor
data is read, and a bad checkpoint fails on every rank alike. Executing the plan
reads each rank's region and nothing else.

In this first stage of https://github.com/ByteDance-Seed/VeOmni/issues/1297 a
checkpoint tensor maps to its model tensor unchanged. A checkpoint that needs a
converter transform still goes through the converter-aware loop of
:func:`veomni.models.module_utils.load_model_weights_ep_sharded`.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import torch
from safetensors import safe_open
from tqdm import tqdm

from ..utils.helper import empty_cache


@dataclass(frozen=True)
class LoadLayout:
    """Where this rank's part of a model tensor sits in the whole tensor.

    ``region`` holds one ``slice(start, stop)`` per dimension, in the coordinates
    of the whole (``global_shape``) tensor.
    """

    global_shape: Tuple[int, ...]
    region: Tuple[slice, ...]

    @classmethod
    def whole(cls, shape: Sequence[int]) -> "LoadLayout":
        shape = tuple(shape)
        return cls(shape, tuple(slice(0, size) for size in shape))

    @classmethod
    def dim0_shard(cls, local_shape: Sequence[int], rank: int, world_size: int) -> "LoadLayout":
        """Rank ``rank``'s even ``Shard(0)`` chunk of a tensor whose local shape is ``local_shape``."""
        local_shape = tuple(local_shape)
        rows = local_shape[0]
        global_shape = (rows * world_size, *local_shape[1:])
        region = (slice(rank * rows, (rank + 1) * rows), *(slice(0, size) for size in local_shape[1:]))
        return cls(global_shape, region)

    @property
    def is_whole(self) -> bool:
        return all(dim.start == 0 and dim.stop == size for dim, size in zip(self.region, self.global_shape))


@dataclass(frozen=True)
class LoadPlanEntry:
    target: str
    is_buffer: bool
    source_key: str
    source_file: str
    layout: LoadLayout


@dataclass
class LoadPlan:
    entries: List[LoadPlanEntry] = field(default_factory=list)
    # Model-side names of checkpoint tensors no model tensor is loaded from.
    unexpected_keys: List[str] = field(default_factory=list)

    def entries_by_file(self) -> Dict[str, List[LoadPlanEntry]]:
        by_file: Dict[str, List[LoadPlanEntry]] = {}
        for entry in self.entries:
            by_file.setdefault(entry.source_file, []).append(entry)
        return by_file


def build_load_plan(
    key_to_file: Mapping[str, str],
    file_to_path: Mapping[str, str],
    *,
    target_name: Callable[[str], str],
    parameter_layouts: Mapping[str, LoadLayout],
    buffer_shapes: Mapping[str, Sequence[int]],
) -> LoadPlan:
    """Plan the load of ``key_to_file``'s checkpoint tensors from their headers.

    Args:
        key_to_file: Checkpoint key -> shard basename.
        file_to_path: Shard basename -> local path.
        target_name: Maps a checkpoint key to the model tensor name it would load.
        parameter_layouts: Layout of every parameter to load, by model name.
        buffer_shapes: Shape of every buffer to load, by model name. Buffers are
            read whole.

    Raises:
        RuntimeError: two checkpoint keys map to one model tensor, or a checkpoint
            tensor's shape differs from its model tensor's whole shape.
    """
    plan = LoadPlan()
    sources: Dict[str, str] = {}
    keys_by_file: Dict[str, List[str]] = {}
    for key, fname in key_to_file.items():
        keys_by_file.setdefault(fname, []).append(key)

    for fname in sorted(keys_by_file):
        with safe_open(file_to_path[fname], framework="pt", device="cpu") as f:
            for key in keys_by_file[fname]:
                name = target_name(key)
                if name in parameter_layouts:
                    is_buffer, layout = False, parameter_layouts[name]
                elif name in buffer_shapes:
                    is_buffer, layout = True, LoadLayout.whole(buffer_shapes[name])
                else:
                    plan.unexpected_keys.append(name)
                    continue
                if name in sources:
                    raise RuntimeError(
                        f"Checkpoint keys '{sources[name]}' and '{key}' both load model tensor '{name}'."
                    )
                sources[name] = key
                shape = tuple(f.get_slice(key).get_shape())
                if shape != layout.global_shape:
                    raise RuntimeError(
                        f"Checkpoint tensor '{key}' has shape {shape}, but model tensor '{name}' "
                        f"is {layout.global_shape} in total."
                    )
                plan.entries.append(LoadPlanEntry(name, is_buffer, key, fname, layout))
    return plan


def execute_load_plan(
    plan: LoadPlan,
    file_to_path: Mapping[str, str],
    dispatch: Callable[[LoadPlanEntry, torch.Tensor], None],
) -> None:
    """Read each entry's region and hand it to ``dispatch``, one shard file at a time."""
    for fname, entries in tqdm(
        sorted(plan.entries_by_file().items()),
        desc="Streaming checkpoint by load plan",
        disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
    ):
        with safe_open(file_to_path[fname], framework="pt", device="cpu") as f:
            for entry in entries:
                if entry.layout.is_whole:
                    tensor = f.get_tensor(entry.source_key)
                else:
                    tensor = f.get_slice(entry.source_key)[entry.layout.region]
                dispatch(entry, tensor)
        empty_cache()
