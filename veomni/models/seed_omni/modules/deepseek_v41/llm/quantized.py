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

"""DeepSeek-V4.1 projection and expert modules with optional pre-quantized storage.

The released DeepSeek-V4.1 weights are FP8 for dense projections and packed FP4
for routed experts. Which parameters hold quantized storage is a property of
``config.weight_quantization``; which kernel executes them is controlled
explicitly by ``config.ops_implementation`` via instance-local :class:`VeomniOp`
handles:

* Training (BF16 weights): ``eager`` / ``qat`` / ``torchao`` / ``deepgemm`` /
  ``liger_kernel`` / ``fused_triton`` / etc.

Modeling classes only allocate parameters and invoke their bound ``VeomniOp``
handle -- they never branch on ``is_quantized`` in ``forward`` or auto-switch
implementations in ``__init__``. Mismatches between parameter storage and the
configured op implementation fail fast inside the op row at runtime.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from ......ops import VeomniOp
from ......ops.qat import act_quant
from ..quant_scheme import QUANT_BLOCK_SIZE, SCALE_DTYPE, SCALE_FMT, quantized_storage_dtype
from .reuse import (
    DeepseekV4Experts,
    DeepseekV4GroupedLinear,
    DeepseekV4MLP,
    resolve_module_op_impl,
)


def is_quantized(weight: torch.Tensor) -> bool:
    """Whether ``weight`` holds pre-quantized FP8 or packed FP4 storage.

    ``int8`` means packed FP4 rather than an integer weight -- the release
    stores ``float4_e2m1fn_x2`` under safetensors' ``I8``, since the format has
    no code for a 4-bit float.
    """
    return weight.dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2, torch.int8)


def linear_quant_dtype(config) -> torch.dtype | None:
    """The storage dtype for a dense projection, or ``None`` for an ordinary layer."""
    return quantized_storage_dtype(config)


def expert_quant_dtype(config) -> torch.dtype | None:
    """The storage dtype for the stacked routed experts, or ``None``."""
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


def quantized_linear_pair(
    x: torch.Tensor,
    linear_a: nn.Module,
    linear_b: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run two projections sharing input ``x`` via the ``linear`` family's paired dispatcher."""
    from ......ops.kernels.linear import paired_linear

    return paired_linear(x, linear_a, linear_b)


class QuantizedWeightMixin:
    """Gives an ``nn.Linear`` subclass the option of a pre-quantized FP8 weight and scale."""

    def _replace_weight_with_quantized(self, quant_dtype: torch.dtype) -> None:
        out_features, in_features = self.weight.shape
        blocks = lambda size: (size + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE  # noqa: E731
        device = self.weight.device

        if quant_dtype == torch.float8_e4m3fn:
            weight = torch.empty(out_features, in_features, dtype=quant_dtype, device=device)
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
    """``nn.Linear`` dispatched through ``VeomniOp("linear", "standard", ...)``."""

    #: Unquantized ``DeepseekV41Linear`` owns a plain weight and delegates its
    #: forward to ``self.veomni_linear``; declare that directly on this class so
    #: every ``linear/standard`` quantizing row (``qat``, ``deepgemm``, ``torchao``)
    #: accepts it while dynamic FSDP subclasses stay refused.
    _veomni_plain_linear = True
    #: Signal ``linear/standard/eager`` to invoke ``_eager_forward`` instead of
    #: recursing back into ``self.forward``.
    _veomni_self_dispatched = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        quant_dtype: torch.dtype | None = None,
        config: Any = None,
        *,
        site: str | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.veomni_linear = VeomniOp(
            "linear",
            "standard",
            resolve_module_op_impl(config, "linear_implementation"),
            site=site,
        )
        if quant_dtype is not None:
            self._replace_weight_with_quantized(quant_dtype)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.Linear.forward(self, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.veomni_linear(self, x)


class DeepseekV41GroupedLinear(DeepseekV4GroupedLinear, QuantizedWeightMixin):
    """Block-diagonal grouped projection dispatched through ``VeomniOp("linear", "grouped", ...)``."""

    _veomni_plain_linear = True
    _veomni_self_dispatched = True

    def __init__(
        self,
        in_features_per_group: int,
        out_features: int,
        n_groups: int,
        bias: bool = False,
        quant_dtype: torch.dtype | None = None,
        config: Any = None,
        *,
        site: str | None = None,
    ) -> None:
        super().__init__(in_features_per_group, out_features, n_groups=n_groups, bias=bias)
        self.veomni_linear = VeomniOp(
            "linear",
            "grouped",
            resolve_module_op_impl(config, "linear_implementation"),
            site=site,
        )
        if quant_dtype is None:
            return
        out_per_group = out_features // n_groups
        if out_per_group % QUANT_BLOCK_SIZE:
            raise ValueError(
                f"A group's {out_per_group} output rows do not divide into {QUANT_BLOCK_SIZE}-row "
                "scale tiles, so this weight cannot be sliced per group."
            )
        self._replace_weight_with_quantized(quant_dtype)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        return DeepseekV4GroupedLinear.forward(self, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.veomni_linear(self, x)


class DeepseekV41SharedExpert(DeepseekV4MLP):
    """Shared expert MLP dispatched through ``VeomniOp("gated_mlp", "standard", ...)``."""

    def __init__(self, config, quant_dtype: torch.dtype | None = None, *, layer_idx: int = 0) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit
        self.veomni_gated_mlp = VeomniOp(
            "gated_mlp",
            "standard",
            resolve_module_op_impl(config, "gated_mlp_implementation"),
            site=f"layers.{layer_idx}.mlp",
        )
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
                    config=config,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.veomni_gated_mlp(
            self,
            x,
            swiglu_limit=self.limit,
            activation=self.config.hidden_act,
        )


class DeepseekV41QuantizedExperts(DeepseekV4Experts):
    """Routed experts dispatched through ``VeomniOp("moe_experts", "standard", ...)``."""

    def __init__(self, config, quant_dtype: torch.dtype | None = None, *, layer_idx: int = 0) -> None:
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.activation = config.hidden_act
        self.limit = config.swiglu_limit
        self.assume_distinct_experts = False
        self.expert_dtype = getattr(config, "expert_dtype", "fp8")
        self.veomni_moe = VeomniOp(
            "moe_experts",
            "standard",
            resolve_module_op_impl(config, "moe_implementation"),
            site=f"layers.{layer_idx}.experts",
        )
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
        unused = self.gate_up_proj.new_empty(0)
        routing_weights = top_k_weights if self.veomni_moe.impl == "eager" else top_k_weights.to(hidden_states.dtype)
        scale_kwargs: dict[str, Any] = {}
        gate_up_scale = getattr(self, "gate_up_proj_scale", None)
        down_scale = getattr(self, "down_proj_scale", None)
        if gate_up_scale is not None or down_scale is not None:
            scale_kwargs["fc1_1_2_scale"] = gate_up_scale
            scale_kwargs["fc2_scale"] = down_scale
        return self.veomni_moe(
            hidden_states,
            routing_weights,
            top_k_index,
            unused,
            unused,
            self.down_proj,
            self.gate_up_proj,
            num_experts=self.num_experts,
            activation=self.activation,
            swiglu_limit=self.limit,
            assume_distinct_experts=self.assume_distinct_experts,
            dtype=self.expert_dtype,
            **scale_kwargs,
        )


__all__ = [
    "QUANT_BLOCK_SIZE",
    "SCALE_DTYPE",
    "SCALE_FMT",
    "DeepseekV41GroupedLinear",
    "DeepseekV41Linear",
    "DeepseekV41QuantizedExperts",
    "DeepseekV41SharedExpert",
    "QuantizedWeightMixin",
    "act_quant",
    "expert_quant_dtype",
    "is_quantized",
    "linear_quant_dtype",
    "quantized_linear_pair",
]
