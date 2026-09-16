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

"""V4.1's QAT recipe: which tensors training rounds, and how.

The released checkpoint serves dense projections in FP8 and routed experts in
FP4, so training has to reproduce that rounding or the deployed model meets a
numeric regime it never saw. The recipe lives in one file for the same reason
V4's does: *which* call sites round is the recipe, and spreading the decision
across modules makes it un-reviewable.

These mirror ``deepseek_v4``'s ``veomni_qat_*`` helpers, which are patchgen
helpers injected into a generated module and therefore not importable as
library functions. The block sizes come from the checkpoint's
``quantization_config``, not from a tuning knob: a per-site override would train
against rounding no inference kernel performs.

Two sites differ from V4 and are called out where they occur:

* The compressed KV latent is FP4 with 16-wide groups and E4M3 scales, against
  the indexer's 32-wide groups with E8M0. Both are the released
  ``inference/model.py``'s choices, and they are not interchangeable.
* V4's indexer Q/K recipe is the FP8 variant of DeepSeek's Hadamard-rotated FP4,
  chosen because the SM90 target has no Hadamard. V4.1 inherits that reasoning.
"""

import torch
import torch.nn as nn

from ......ops.dispatch import OpsConfigSlot
from ......ops.qat import (
    fp4_fake_quant_weight,
    fp8_fake_quant_act,
    fp8_fake_quant_act_prefix,
    fp8_fake_quant_stacked_weight,
    qat_linear,
)


veomni_qat_implementation = OpsConfigSlot("qat_implementation")


def qat_enabled() -> bool:
    """Whether the FP8/FP4 simulation is on for this run."""
    return veomni_qat_implementation.value == "fp8_blockwise"


def fake_quant_linear(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run ``linear`` with FP8 fake-quantized operands: 128x128 weight tiles,
    1x128 activation blocks with ue8m0 scales."""
    return qat_linear(linear, x, enabled=qat_enabled())


def fake_quant_kv(kv: torch.Tensor, rope_features: int) -> torch.Tensor:
    """FP8-simulate the NoPE channels of a cached KV entry.

    A storage recipe rather than a GEMM operand: inference caches the entry in
    FP8 but attends in BF16, so training only reproduces the cache round-trip's
    rounding. The trailing ``rope_features`` channels stay in the model dtype --
    RoPE encodes position as an angle, where mantissa noise costs more than the
    bytes save. Blocks are 64 wide because ``head_dim - rope_features`` is 448 on
    V4.1-Flash, which no 128-wide block divides.
    """
    if not qat_enabled() or kv.numel() == 0:
        return kv
    return fp8_fake_quant_act_prefix(kv, kv.shape[-1] - rope_features, block_size=64)


def fake_quant_act(x: torch.Tensor) -> torch.Tensor:
    """FP8-simulate an activation across its whole last dimension, 1x128 blocks.

    For operands where every channel enters the FP8 product -- the indexer's Q
    and index keys, and the routed experts' input tokens. Unlike
    :func:`fake_quant_kv` the RoPE channels are included, because the indexer's
    logits are served as a real FP8 x FP8 product instead of being dequantized
    for a BF16 attention.
    """
    if not qat_enabled() or x.numel() == 0:
        return x
    return fp8_fake_quant_act(x, block_size=128)


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
    "fake_quant_act",
    "fake_quant_expert_weight",
    "fake_quant_kv",
    "fake_quant_linear",
    "qat_enabled",
]
