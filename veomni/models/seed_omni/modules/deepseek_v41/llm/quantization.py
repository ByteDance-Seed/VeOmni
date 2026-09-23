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

"""V4.1's storage rounding: which tensors inference/QAT round, and how.

The released ``inference/model.py`` writes three different recipes into the
caches, and they are not interchangeable:

* sliding-window KV -- FP8, 32-wide groups, RoPE tail included
  (``act_quant(..., fp8_block_size=32, dequant=True)``);
* compressed KV -- FP4, 16-wide groups, E4M3 scales
  (``fp4_act_quant(..., 16, True, scale_dtype=e4m3)``);
* indexer Q/K -- FP4, 32-wide groups, E8M0 scales.

V4's QAT helper rounded only the NoPE prefix of a KV entry as FP8/64. That is
the wrong recipe for every one of those three sites. This file is the V4.1
recipes, used both at inference (``module.eval()``) and during QAT.

The round-trip is eager PyTorch so prefill-sized M does not hit the TileLang
``act_quant`` NaNs, and so CPU tests can check the same arithmetic.
"""

import torch
import torch.nn as nn

from ......ops.dispatch import OpsConfigSlot
from ......ops.qat import (
    fp4_fake_quant_weight,
    fp8_fake_quant_stacked_weight,
    qat_linear,
)


veomni_qat_implementation = OpsConfigSlot("qat_implementation")
# ``_bind_veomni_ops`` only scans the OmniModule's own Python module
# (``modeling.py`` / ``accelerated.py``). Those files re-export this object
# so ``ops_implementation.qat_implementation`` actually reaches ``qat_enabled``.

#: Official ``fp8_block_size`` for window KV (and dense GEMM activations).
WINDOW_KV_FP8_BLOCK = 32
#: Official compressed-KV group width. Not the indexer's 32.
COMPRESS_KV_FP4_GROUP = 16
#: Official indexer Q/K group width.
INDEX_FP4_GROUP = 32

_FP8_MAX = 448.0
_FP8_MAX_INV = 1.0 / _FP8_MAX
_FP4_MAX = 6.0
_FP4_ABS_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Compressed-KV E4M3 path keeps an all-zero group's scale nonzero.
_FP4_E4M3_AMAX_FLOOR = _FP4_MAX * (2**-9)
_FP4_E8M0_AMAX_FLOOR = _FP4_MAX * (2**-126)


def qat_enabled() -> bool:
    """Whether the FP8/FP4 simulation is on for this run."""
    return veomni_qat_implementation.value == "fp8_blockwise"


def _storage_roundtrip_enabled(training: bool) -> bool:
    """Inference always rounds; training rounds only under QAT."""
    return (not training) or qat_enabled()


def fake_quant_linear(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run ``linear`` with FP8 fake-quantized operands: 128x128 weight tiles,
    1x128 activation blocks with ue8m0 scales."""
    return qat_linear(linear, x, enabled=qat_enabled())


def _ue8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """Power-of-two scale from ``amax / 448``, matching the eager GEMM path."""
    scaled = (amax * _FP8_MAX_INV).contiguous()
    bits = scaled.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    log2_ceil = exponent + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    return ((log2_ceil + 127) << 23).view(torch.float32)


def _eager_fp8_roundtrip(x: torch.Tensor, block_size: int) -> torch.Tensor:
    last = x.shape[-1]
    if last % block_size:
        raise ValueError(f"last dim {last} is not divisible by FP8 block size {block_size}")
    orig_dtype = x.dtype
    rows = x.reshape(-1, last).float()
    n_blocks = last // block_size
    blocks = rows.reshape(rows.shape[0], n_blocks, block_size)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = _ue8m0_scale(amax)
    quantized = (blocks / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape_as(x).to(orig_dtype)


def _round_to_fp4(values: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 codepoint in ``[-6, 6]``."""
    levels = values.new_tensor(_FP4_ABS_LEVELS)
    magnitude = values.abs()
    nearest = levels[(magnitude.unsqueeze(-1) - levels).abs().argmin(-1)]
    return values.sign() * nearest


def _pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    """Round ``amax / 6`` up to a power of two (E8M0)."""
    ratio = (amax / _FP4_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    return torch.pow(2.0, torch.ceil(torch.log2(ratio)))


def _eager_fp4_roundtrip(x: torch.Tensor, block_size: int, scale_dtype: torch.dtype) -> torch.Tensor:
    last = x.shape[-1]
    if last % block_size:
        raise ValueError(f"last dim {last} is not divisible by FP4 group size {block_size}")
    orig_dtype = x.dtype
    rows = x.reshape(-1, last).float()
    n_blocks = last // block_size
    blocks = rows.reshape(rows.shape[0], n_blocks, block_size)
    amax = blocks.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(_FP4_E4M3_AMAX_FLOOR)
        scale = (amax / _FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(_FP4_E8M0_AMAX_FLOOR)
        scale = _pow2_scale(amax)
    quantized = _round_to_fp4((blocks / scale).clamp(-_FP4_MAX, _FP4_MAX))
    return (quantized * scale).reshape_as(x).to(orig_dtype)


class _StorageRoundTrip(torch.autograd.Function):
    """Straight-through estimator around the official cache round-trip."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "window":
            return _eager_fp8_roundtrip(x, WINDOW_KV_FP8_BLOCK)
        if kind == "compress":
            return _eager_fp4_roundtrip(x, COMPRESS_KV_FP4_GROUP, torch.float8_e4m3fn)
        if kind == "index":
            return _eager_fp4_roundtrip(x, INDEX_FP4_GROUP, torch.float8_e8m0fnu)
        raise ValueError(f"unknown KV storage recipe {kind!r}")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def _round_trip(x: torch.Tensor, kind: str) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return _StorageRoundTrip.apply(x, kind)


def round_window_kv(kv: torch.Tensor, *, training: bool) -> torch.Tensor:
    """Official window-KV storage: FP8, 32-wide, whole vector including RoPE."""
    if kv.numel() == 0 or not _storage_roundtrip_enabled(training):
        return kv
    return _round_trip(kv, "window")


def round_compress_kv(kv: torch.Tensor, *, training: bool) -> torch.Tensor:
    """Official compressed-KV storage: FP4, 16-wide groups, E4M3 scales."""
    if kv.numel() == 0 or not _storage_roundtrip_enabled(training):
        return kv
    return _round_trip(kv, "compress")


def round_index_act(x: torch.Tensor, *, training: bool) -> torch.Tensor:
    """Official indexer Q/K storage: FP4, 32-wide groups, E8M0 scales."""
    if x.numel() == 0 or not _storage_roundtrip_enabled(training):
        return x
    return _round_trip(x, "index")


def fake_quant_expert_weight(weight: torch.Tensor, expert_dtype: str) -> torch.Tensor:
    """Fake-quantize a stacked routed-expert weight ``[E, out, in]``.

    Follows the checkpoint's declared ``expert_dtype`` rather than the global QAT
    flag's default, matching what the converter writes on export. Each expert is
    quantized as its own matrix, which is also what expert parallelism needs: EP
    shards only the expert dimension, so no block spans a shard boundary.
    """
    if not qat_enabled():
        return weight
    if expert_dtype == "fp4":
        return fp4_fake_quant_weight(weight)
    return fp8_fake_quant_stacked_weight(weight)


__all__ = [
    "COMPRESS_KV_FP4_GROUP",
    "INDEX_FP4_GROUP",
    "WINDOW_KV_FP8_BLOCK",
    "fake_quant_expert_weight",
    "fake_quant_linear",
    "qat_enabled",
    "round_compress_kv",
    "round_index_act",
    "round_window_kv",
]
