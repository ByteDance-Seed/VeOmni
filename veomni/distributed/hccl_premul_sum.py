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

"""HCCL compatibility for PyTorch's ``ReduceOp.PREMUL_SUM`` collectives."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
from torch.distributed.distributed_c10d import ReduceOp


_WRAPPED_MARKER = "_veomni_hccl_premul_sum_wrapped"
_PATCH_LOCK = threading.Lock()
_PATCHED_COLLECTIVES = (
    ("all_reduce", "tensor"),
    ("reduce_scatter", "output"),
    ("reduce_scatter_tensor", "output"),
)
_MISSING = object()


def _parameter_position(function: Callable[..., Any], name: str) -> int | None:
    try:
        return tuple(inspect.signature(function).parameters).index(name)
    except (TypeError, ValueError):
        return None


def _premul_sum_factor(reduce_op: Any) -> Any | None:
    if reduce_op != ReduceOp.PREMUL_SUM:
        return None

    state = reduce_op.__getstate__()
    if not isinstance(state, tuple) or len(state) < 2:
        raise RuntimeError(f"Unexpected PREMUL_SUM state: {state!r}")
    return state[1]


def hccl_premul_sum_wrapper(op: Callable[..., Any], output_name: str) -> Callable[..., Any]:
    """Decompose unsupported HCCL PREMUL_SUM into SUM followed by scaling.

    Other reduce operations pass through without synchronization. A PREMUL_SUM
    call must wait for the SUM before scaling its output; an async call therefore
    returns its original work handle in an already-completed state.
    """
    op_position = _parameter_position(op, "op")

    @wraps(op)
    def wrapper(*args, **kwargs):
        reduce_op = kwargs.get("op", _MISSING)
        op_is_keyword = reduce_op is not _MISSING
        if reduce_op is _MISSING and op_position is not None and len(args) > op_position:
            reduce_op = args[op_position]

        factor = _premul_sum_factor(reduce_op) if reduce_op is not _MISSING else None
        if factor is None:
            return op(*args, **kwargs)

        call_args = args
        call_kwargs = kwargs
        if op_is_keyword:
            call_kwargs = dict(kwargs)
            call_kwargs["op"] = ReduceOp.SUM
        else:
            mutable_args = list(args)
            mutable_args[op_position] = ReduceOp.SUM
            call_args = tuple(mutable_args)

        output = args[0] if args else kwargs[output_name]
        handle = op(*call_args, **call_kwargs)
        if handle is not None:
            handle.wait()
        with torch.no_grad():
            output.mul_(factor)
        return handle

    setattr(wrapper, _WRAPPED_MARKER, True)
    return wrapper


def apply_hccl_premul_sum_patch() -> None:
    """Patch PyTorch collectives once for HCCL PREMUL_SUM compatibility."""
    with _PATCH_LOCK:
        for op_name, output_name in _PATCHED_COLLECTIVES:
            current = getattr(torch.distributed, op_name)
            if getattr(current, _WRAPPED_MARKER, False):
                continue
            setattr(torch.distributed, op_name, hccl_premul_sum_wrapper(current, output_name))


__all__ = ["apply_hccl_premul_sum_patch", "hccl_premul_sum_wrapper"]
