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

"""Linear layers that consume the released checkpoint's own quantized weights.

Ported from ``inference/model.py``'s ``linear()`` and ``Linear``. The released
DeepSeek-V4.1 weights are fp8 for the dense projections and fp4 for the routed
experts, and running them in that form is the difference between 475 GiB and
1.5 TB -- the latter does not fit on eight H20s.

The activation is quantized to fp8 on the fly and the weight is never
dequantized, which is what ``fp8_gemm`` and ``fp4_gemm`` expect. fp4 buys memory
rather than arithmetic: the kernel widens each fp4 weight to fp8 in shared
memory and runs an ordinary fp8 tensor-core GEMM.

Not every projection is quantized, and which ones are is a property of the
checkpoint rather than a choice. Reading it off the released tensors:

====================================  ==========
``self_attn.wq_a`` / ``wq_b``         fp8
``self_attn.wkv`` / ``wo_a`` / ``wo_b``  fp8
``self_attn.indexer.wq_b``            fp8
``mlp.shared_experts.*``              fp8
``engram.N.wkv``                      fp8
``mlp.experts.*`` (routed)            fp4
``compressor.wkv`` / ``wgate``        bf16
``indexer.wk`` / ``weights_proj``     bf16
``mlp.gate`` (router)                 bf16
====================================  ==========

The bf16 entries are not an oversight upstream. They are the small projections
whose inputs carry the positional and routing decisions -- a compressor gate, an
indexer key, the router -- where a quantization error does not shrink an output
slightly but selects a different slot or a different expert.

There is a shortcut this file deliberately does not take. All the dense fp8
together is 6.4 GiB, so dequantizing it to bf16 at load would cost 0.8 GiB per
rank across eight and need none of this code, losing nothing numerically: fp8
has fewer mantissa bits than bf16 and every scale is a power of two, so the
round trip is exact. It is rejected on decode throughput, not on accuracy. A
decode step is memory-bound and reads every dense weight but only
``num_experts_per_tok`` of each layer's experts, so the dense side is roughly
45% of the bytes moved per token, and doubling it costs about a quarter of the
achievable rate.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ......ops.kernels.deepseek_v4 import act_quant, fp4_gemm, fp8_gemm
from .reuse import DeepseekV4GroupedLinear, DeepseekV4MLP


#: ``quantization_config.weight_block_size`` in the released config, and the
#: width of one activation quantization block. Both sides must agree: the
#: kernels index a weight's scale by the same block counter as the activation's.
QUANT_BLOCK_SIZE = 32

#: ``quantization_config.scale_fmt``. Rounds every scale up to a power of two,
#: which is what lets it be stored as a bare exponent.
SCALE_FMT = "ue8m0"

#: The bare exponent that ``SCALE_FMT`` makes possible: no mantissa, one byte.
SCALE_DTYPE = torch.float8_e8m0fnu


def is_quantized(weight: torch.Tensor) -> bool:
    """Whether ``weight`` needs a scale and a kernel rather than ``F.linear``.

    ``int8`` means packed fp4 rather than an integer weight -- the release
    stores ``float4_e2m1fn_x2`` under safetensors' ``I8``, since the format has
    no code for a 4-bit float.
    """
    return weight.dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2, torch.int8)


def linear_quant_dtype(config) -> Optional[torch.dtype]:
    """The storage dtype for a dense projection, or ``None`` for an ordinary layer.

    ``config.weight_quantization`` is the released ``quantization_config`` and is
    present only on the native artifact; the bf16 one has been dequantized and
    gets plain ``nn.Linear``. Every dense projection the release quantizes is
    fp8 -- fp4 is only ever the stacked routed experts, which are not a Linear.

    The scheme is checked rather than assumed. The kernels index a weight's
    scale by the same block counter as the activation's, so a checkpoint
    quantized on a different block width would not fail to run, it would read
    the wrong scale for every block and return plausible nonsense.
    """
    declared = getattr(config, "weight_quantization", None)
    if not declared:
        return None

    block_size = declared.get("weight_block_size", [QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE])
    if tuple(block_size) != (QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE):
        raise ValueError(
            f"This checkpoint is quantized in {tuple(block_size)} blocks, but the kernels here are "
            f"built for {(QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE)}. Running it anyway would pair every "
            "block with the wrong scale."
        )
    scale_fmt = declared.get("scale_fmt", SCALE_FMT)
    if scale_fmt != SCALE_FMT:
        raise ValueError(
            f"This checkpoint's scales are {scale_fmt!r}, not {SCALE_FMT!r}. Only the latter rounds "
            f"every scale to a power of two, which is what lets it be stored as the bare exponent "
            f"{SCALE_DTYPE} the kernels read."
        )
    return torch.float8_e4m3fn


def quantized_matmul(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` with ``weight`` left in the precision it was released in.

    Both paths quantize the activation to fp8 first; fp4 weights take an fp8
    activation too, and the kernel handles the mixed precision.
    """
    quantized_x, x_scale = act_quant(x, QUANT_BLOCK_SIZE, SCALE_FMT, SCALE_DTYPE)
    if weight.dtype == torch.float8_e4m3fn:
        return fp8_gemm(
            quantized_x,
            x_scale,
            weight,
            scale,
            scale_dtype=SCALE_DTYPE,
            block_size=QUANT_BLOCK_SIZE,
            out_dtype=x.dtype,
        )
    return fp4_gemm(
        quantized_x,
        x_scale,
        weight.view(torch.float4_e2m1fn_x2),
        scale,
        scale_dtype=SCALE_DTYPE,
        act_block_size=QUANT_BLOCK_SIZE,
        out_dtype=x.dtype,
    )


class QuantizedWeightMixin:
    """Gives an ``nn.Linear`` subclass the option of a quantized weight.

    A mixin rather than a class because two different ``nn.Linear`` subclasses
    need it -- the plain projections and ``DeepseekV4GroupedLinear``, whose
    forward reshapes into blocks first and so cannot simply inherit one.

    When ``quantized`` is false this leaves ``nn.Linear`` exactly as it was,
    down to the parameter name, so the bf16 checkpoint and the training path see
    no change at all.
    """

    def _replace_weight_with_quantized(self, quant_dtype: torch.dtype) -> None:
        out_features, in_features = self.weight.shape
        blocks = lambda size: (size + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE  # noqa: E731
        device = self.weight.device

        if quant_dtype == torch.float8_e4m3fn:
            weight = torch.empty(out_features, in_features, dtype=quant_dtype, device=device)
            # One scale per square tile of the weight, not per row: a dense
            # projection's rows share an output space, so they can share range.
            scale_shape = (blocks(out_features), blocks(in_features))
        else:
            raise ValueError(
                f"{quant_dtype} is not a quantized storage dtype for a linear weight. The released "
                "checkpoint quantizes dense projections to fp8 and only the stacked routed experts to "
                "fp4, and a stack is not an nn.Linear."
            )

        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(
            torch.empty(*scale_shape, dtype=SCALE_DTYPE, device=device),
            requires_grad=False,
        )

    @property
    def is_quantized(self) -> bool:
        return getattr(self, "weight_scale", None) is not None


class DeepseekV41Linear(nn.Linear, QuantizedWeightMixin):
    """``nn.Linear``, optionally holding the released fp8 weight instead of bf16.

    ``quant_dtype=None`` is the ordinary layer. The distinction is made at
    construction because a parameter's storage has to exist before the
    checkpoint can be read into it, and the checkpoint is the only thing that
    knows which projections were quantized.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        quant_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        if quant_dtype is not None:
            self._replace_weight_with_quantized(quant_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_quantized:
            return F.linear(x, self.weight, self.bias)
        out = quantized_matmul(x, self.weight, self.weight_scale)
        return out if self.bias is None else out + self.bias


class DeepseekV41GroupedLinear(DeepseekV4GroupedLinear, QuantizedWeightMixin):
    """The block-diagonal output projection, optionally over the released fp8 weight.

    ``wo_a`` is one ``[out_features, in_per_group]`` weight that the generated
    forward *views* as ``n_groups`` blocks and runs through a ``bmm``. The
    checkpoint quantizes it as that one flat weight, on the same 32x32 tiles as
    every other projection, so nothing about the scale layout is special here --
    a group's rows are simply a contiguous slice of it.

    There is no batched fp8 GEMM to call, so the groups are looped. At decode
    the loop is over eight tiny GEMMs and the cost is launch overhead on a step
    that is memory-bound anyway.
    """

    def __init__(
        self,
        in_features_per_group: int,
        out_features: int,
        n_groups: int,
        bias: bool = False,
        quant_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(in_features_per_group, out_features, n_groups=n_groups, bias=bias)
        if quant_dtype is None:
            return
        out_per_group = out_features // n_groups
        if out_per_group % QUANT_BLOCK_SIZE:
            # Slicing the scale by group would otherwise cut a tile in half and
            # hand the neighbouring group's exponent to the last rows.
            raise ValueError(
                f"A group's {out_per_group} output rows do not divide into {QUANT_BLOCK_SIZE}-row "
                "scale tiles, so this weight cannot be sliced per group."
            )
        self._replace_weight_with_quantized(quant_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_quantized:
            return super().forward(x)

        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        out_per_group = self.weight.shape[0] // self.n_groups
        scale_rows = out_per_group // QUANT_BLOCK_SIZE

        grouped = x.reshape(-1, self.n_groups, hidden_dim)
        outputs = [
            quantized_matmul(
                grouped[:, group].contiguous(),
                self.weight[group * out_per_group : (group + 1) * out_per_group],
                self.weight_scale[group * scale_rows : (group + 1) * scale_rows],
            )
            for group in range(self.n_groups)
        ]
        return torch.stack(outputs, dim=1).reshape(*input_shape, self.n_groups, out_per_group)


class DeepseekV41SharedExpert(DeepseekV4MLP):
    """The shared expert, optionally over the released fp8 weights.

    Only ``__init__`` differs. The generated forward reaches its projections
    through ``veomni_qat_linear``, which with QAT off is the identity wrapper
    over ``linear(x)`` and dispatches to whatever layer is installed -- so the
    clamp and the SwiGLU fusion above it keep working untouched.
    """

    def __init__(self, config, quant_dtype: Optional[torch.dtype] = None) -> None:
        super().__init__(config)
        if quant_dtype is None:
            return
        for name in ("gate_proj", "up_proj", "down_proj"):
            plain = getattr(self, name)
            setattr(
                self,
                name,
                DeepseekV41Linear(
                    plain.in_features,
                    plain.out_features,
                    bias=plain.bias is not None,
                    quant_dtype=quant_dtype,
                ),
            )


__all__ = [
    "QUANT_BLOCK_SIZE",
    "SCALE_DTYPE",
    "SCALE_FMT",
    "DeepseekV41GroupedLinear",
    "DeepseekV41Linear",
    "DeepseekV41SharedExpert",
    "QuantizedWeightMixin",
    "is_quantized",
    "linear_quant_dtype",
    "quantized_matmul",
]
