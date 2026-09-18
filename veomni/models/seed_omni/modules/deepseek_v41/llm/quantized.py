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
from ..quant_scheme import QUANT_BLOCK_SIZE, SCALE_DTYPE, SCALE_FMT, quantized_storage_dtype
from .reuse import DeepseekV4Experts, DeepseekV4GroupedLinear, DeepseekV4MLP


def is_quantized(weight: torch.Tensor) -> bool:
    """Whether ``weight`` needs a scale and a kernel rather than ``F.linear``.

    ``int8`` means packed fp4 rather than an integer weight -- the release
    stores ``float4_e2m1fn_x2`` under safetensors' ``I8``, since the format has
    no code for a 4-bit float.
    """
    return weight.dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2, torch.int8)


def linear_quant_dtype(config) -> Optional[torch.dtype]:
    """The storage dtype for a dense projection, or ``None`` for an ordinary layer.

    Every dense projection the release quantizes is fp8 -- fp4 is only ever the
    stacked routed experts, which are not a Linear, so this is the shared scheme
    resolver under the name the Linear sites read it by.
    """
    return quantized_storage_dtype(config)


def expert_quant_dtype(config) -> Optional[torch.dtype]:
    """The storage dtype for the stacked routed experts, or ``None``.

    Separate from :func:`linear_quant_dtype` because the experts are the one
    place the release goes to 4 bits, and which format that is comes from
    ``expert_dtype`` rather than the checkpoint-wide ``quant_method``.
    """
    if linear_quant_dtype(config) is None:
        return None
    expert_dtype = getattr(config, "expert_dtype", "fp8")
    if expert_dtype != "fp4":
        raise ValueError(
            f"This checkpoint stores its routed experts as {expert_dtype!r}. Only fp4 is handled "
            "here: V4.1-Flash is an fp4-expert release, and an fp8 stack would carry a different "
            "scale layout that the fp4 path would misread rather than reject."
        )
    return torch.float4_e2m1fn_x2


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


class DeepseekV41QuantizedExperts(DeepseekV4Experts):
    """The routed experts, holding the released fp4 slabs instead of bf16 ones.

    This is where fp4 earns its place: the routed experts are the overwhelming
    majority of the parameters, and at 4 bits they are what brings the
    checkpoint under what eight H20s hold.

    Two things differ from the fp8 sites. The weights are not an ``nn.Linear``
    but one stacked ``[E, out, in]`` parameter per projection, so the scale is
    named ``<param>_scale`` and is per row and per 32-wide block of the reduce
    dimension rather than per square tile -- experts do not share a range with
    each other. And safetensors has no code for a 4-bit float, so the release
    stores the packed pairs as ``int8``, two values to a byte along the reduce
    dimension; the kernel reinterprets them.

    The fused path is not available here. It fake-quantizes in bf16 and then
    multiplies in bf16, which needs a bf16 weight to start from; there is no
    bf16 weight any more. The eager loop is what the quantized forward extends,
    and at decode it is the right shape anyway -- a token touches only
    ``num_experts_per_tok`` of the experts, so the loop runs a handful of
    narrow GEMMs rather than the whole slab.
    """

    def __init__(self, config, quant_dtype: Optional[torch.dtype] = None) -> None:
        super().__init__(config)
        self.experts_are_quantized = quant_dtype is not None
        if not self.experts_are_quantized:
            return
        if self.hidden_dim % QUANT_BLOCK_SIZE or self.intermediate_dim % QUANT_BLOCK_SIZE:
            raise ValueError(
                f"Expert dimensions {self.hidden_dim} and {self.intermediate_dim} must be whole "
                f"numbers of {QUANT_BLOCK_SIZE}-wide scale blocks."
            )

        device = self.gate_up_proj.device
        self.gate_up_proj = self._packed(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim, device=device)
        self.gate_up_proj_scale = self._scale(
            self.num_experts, 2 * self.intermediate_dim, self.hidden_dim, device=device
        )
        self.down_proj = self._packed(self.num_experts, self.hidden_dim, self.intermediate_dim, device=device)
        self.down_proj_scale = self._scale(self.num_experts, self.hidden_dim, self.intermediate_dim, device=device)

    @staticmethod
    def _packed(experts: int, out_features: int, in_features: int, device) -> nn.Parameter:
        """``int8`` and half the columns: two fp4 values share a byte along ``in_features``."""
        return nn.Parameter(
            torch.empty(experts, out_features, in_features // 2, dtype=torch.int8, device=device),
            requires_grad=False,
        )

    @staticmethod
    def _scale(experts: int, out_features: int, in_features: int, device) -> nn.Parameter:
        return nn.Parameter(
            torch.empty(
                experts,
                out_features,
                in_features // QUANT_BLOCK_SIZE,
                dtype=SCALE_DTYPE,
                device=device,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self.experts_are_quantized:
            return super().forward(hidden_states, top_k_index, top_k_weights)

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = quantized_matmul(
                current_state,
                self.gate_up_proj[expert_idx],
                self.gate_up_proj_scale[expert_idx],
            )
            # The clamp and SwiGLU are the parent's, unchanged: they run on the
            # GEMM's output and are not an operand of the quantized product.
            current_hidden_states = self._apply_gate(gate_up)
            current_hidden_states = quantized_matmul(
                current_hidden_states,
                self.down_proj[expert_idx],
                self.down_proj_scale[expert_idx],
            )
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


__all__ = [
    "QUANT_BLOCK_SIZE",
    "SCALE_DTYPE",
    "SCALE_FMT",
    "DeepseekV41GroupedLinear",
    "DeepseekV41Linear",
    "DeepseekV41QuantizedExperts",
    "DeepseekV41SharedExpert",
    "QuantizedWeightMixin",
    "expert_quant_dtype",
    "is_quantized",
    "linear_quant_dtype",
    "quantized_matmul",
]
