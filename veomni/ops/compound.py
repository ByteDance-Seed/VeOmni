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

"""Helpers for compound ops that nest other registry rows.

Resolve a nested handle to a ``OpEntry`` and flatten that row's
``SavedState`` into the outer tensor list. Compound math must call the raw
pair, not the generated wrapper.
"""

from __future__ import annotations

from typing import Any, Union

from .registry import OpEntry, SavedState, VeomniOp, resolve_op


# OpEntry, VeomniOp, or None to resolve the default row.
InnerHandle = Union[OpEntry, VeomniOp, None]


def resolve_inner_op(
    handle: InnerHandle,
    *,
    op: str,
    variant: str,
    impl: str = "eager",
) -> OpEntry:
    """Return the nested ``OpEntry`` whose raw pair the compound should call.

    ``None`` resolves ``(op, variant, impl)``. A ``VeomniOp`` unwraps to
    its entry. An ``OpEntry`` is returned as-is.
    """
    if handle is None:
        return resolve_op(op, variant, impl)
    if isinstance(handle, VeomniOp):
        return handle.entry
    return handle


def append_inner(saved: list[Any], state: SavedState) -> tuple[Any, int]:
    """Append ``state.tensors`` onto ``saved`` and return ``(metadata, n)``.

    Pass that spec to ``take_inner`` later to rebuild the inner ``SavedState``.
    """
    saved.extend(state.tensors)
    return state.metadata, len(state.tensors)


def take_inner(rest: tuple[Any, ...], spec: tuple[Any, int]) -> tuple[SavedState, tuple[Any, ...]]:
    """Rebuild an inner ``SavedState`` from the next ``count`` tensors in ``rest``.

    ``spec`` is the ``(metadata, count)`` pair from ``append_inner``. Returns the
    restored state and the unused suffix of ``rest``.
    """
    metadata, count = spec
    return SavedState(rest[:count], metadata), rest[count:]
