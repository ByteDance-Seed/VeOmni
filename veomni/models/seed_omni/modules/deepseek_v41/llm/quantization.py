# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""V4.1 storage-rounding wiring over registered ``veomni.ops`` ``kv_round`` kernels.

All low-level FP8/FP4 storage-rounding kernels live in
:mod:`veomni.ops.kernels.kv_round` and are invoked through :class:`~veomni.ops.VeomniOp`:

* sliding-window KV -- ``("kv_round", "standard", ...)``, 32-wide groups, UE8M0 scales;
* compressed KV -- ``("kv_round", "fp4", ...)``, 16-wide groups, E4M3 scales
  (or ``("kv_round", "standard", ...)`` when ``compress_kv_dtype == "fp8"``);
* indexer Q/K -- ``("kv_round", "fp4", ...)``, 32-wide groups, UE8M0 scales.

During training, the bound ``VeomniOp`` handle is authoritative (``"none"`` leaves
the tensor untouched; ``"eager"`` / ``"triton"`` rounds it; per-site
``OpsSelection`` rules control each site independently). During eval/inference
(``training=False``), deployment cache rounding is always simulated so an absent
or ``"none"`` handle falls back to the registry default for that variant.
"""

from __future__ import annotations

import torch

from ......ops import VeomniOp


#: Official ``fp8_block_size`` for window KV (and dense GEMM activations).
WINDOW_KV_FP8_BLOCK = 32
#: Official compressed-KV group width. Not the indexer's 32.
COMPRESS_KV_FP4_GROUP = 16
#: Official indexer Q/K group width.
INDEX_FP4_GROUP = 32


def _default_kv_round_op(variant: str) -> VeomniOp:
    """The registry's default ``kv_round`` row, for eval or calls without a bound handle."""
    return VeomniOp("kv_round", variant)


def _resolve_kv_round_handle(op: VeomniOp | None, variant: str, *, training: bool) -> VeomniOp | None:
    """Pick the ``kv_round`` handle for this call, or ``None`` when it is a no-op."""
    if training:
        if op is None or op.impl == "none":
            return None
        return op
    if op is None or op.impl == "none":
        return _default_kv_round_op(variant)
    return op


def round_window_kv(
    kv: torch.Tensor,
    *,
    training: bool,
    inplace: bool = False,
    out: torch.Tensor | None = None,
    op: VeomniOp | None = None,
) -> torch.Tensor:
    """Official window-KV storage: FP8, 32-wide, whole vector including RoPE."""
    del inplace
    handle = None if kv.numel() == 0 else _resolve_kv_round_handle(op, "standard", training=training)
    if handle is None:
        if out is not None:
            out.copy_(kv)
            return out
        return kv
    res = handle(kv, block_size=WINDOW_KV_FP8_BLOCK)
    if out is not None:
        out.copy_(res)
        return out
    return res


def round_compress_kv(
    kv: torch.Tensor,
    compress_kv_dtype: str = "fp4",
    *,
    training: bool,
    inplace: bool = False,
    op_fp4: VeomniOp | None = None,
    op_fp8: VeomniOp | None = None,
) -> torch.Tensor:
    """Compressed-KV storage rounding keyed on ``compress_kv_dtype``.

    * ``"fp4"`` (post-training / released V4.1-Flash): FP4, 16-wide groups, E4M3 scales;
    * ``"fp8"`` (pre-training): FP8, 32-wide groups, UE8M0 scales.
    """
    del inplace
    if kv.numel() == 0:
        return kv
    if compress_kv_dtype == "fp4":
        handle = _resolve_kv_round_handle(op_fp4, "fp4", training=training)
        return kv if handle is None else handle(kv, block_size=COMPRESS_KV_FP4_GROUP, scale_fmt="e4m3")
    if compress_kv_dtype == "fp8":
        handle = _resolve_kv_round_handle(op_fp8, "standard", training=training)
        return kv if handle is None else handle(kv, block_size=WINDOW_KV_FP8_BLOCK)
    raise ValueError(f"Unsupported compress_kv_dtype={compress_kv_dtype!r}; expected 'fp4' or 'fp8'.")


def round_index_act(
    x: torch.Tensor,
    *,
    training: bool,
    inplace: bool = False,
    op: VeomniOp | None = None,
) -> torch.Tensor:
    """Official indexer Q/K storage: FP4, 32-wide groups, E8M0 scales."""
    del inplace
    if x.numel() == 0:
        return x
    handle = _resolve_kv_round_handle(op, "fp4", training=training)
    return x if handle is None else handle(x, block_size=INDEX_FP4_GROUP, scale_fmt="ue8m0")


__all__ = [
    "COMPRESS_KV_FP4_GROUP",
    "INDEX_FP4_GROUP",
    "WINDOW_KV_FP8_BLOCK",
    "round_compress_kv",
    "round_index_act",
    "round_window_kv",
]
