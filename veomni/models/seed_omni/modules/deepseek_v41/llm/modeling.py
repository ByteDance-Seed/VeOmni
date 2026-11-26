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

"""DeepSeek-V4.1 CED backbone (``deepseek_v41_llm``).

Single-file modeling for the 40-layer decoder stack, covering:
- shared ``deepseek_v4`` reuse surface & module-scoped ``VeomniOp`` resolution,
- storage-rounding hooks (``kv_round``),
- optional pre-quantized checkpoint storage modules (training runs on plain
  ``nn.Linear`` / ``DeepseekV4GroupedLinear`` / ``DeepseekV4MLP`` /
  ``DeepseekV4Experts`` dispatched through ``VeomniOp``),
- ``SequenceShard`` context parallelism,
- pipelined manifold-constrained Hyper-Connections (mHC),
- Engram residual-stream connection,
- hybrid sliding-window + compressed-KV attention & two-level DSA indexer,
- aux-loss-free VL-aware MoE router & sparse MoE block,
- ``DeepseekV41DecoderLayer``, ``DeepseekV41Backbone``, and ``DeepseekV41Llm``.

**No embedding and no LM head**: those live in ``deepseek_v41_text_encoder``.
**No DSpark / MTP**: those live in ``deepseek_v41_mtp``.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils.generic import maybe_autocast

from ......distributed.context_parallel.dsa_cp import all_gather_kv
from ......distributed.sequence_parallel import gather_outputs
from ......models.transformers.deepseek_v4 import modeling_deepseek_v4 as _generated
from ......models.transformers.deepseek_v4.dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    dense_sparse_attn,
    main_window_indices,
)
from ......ops import OpsSelection, VeomniOp
from ......ops.kernels.dsa.sparse_mqa_target import sparse_attn_tilelang, sparse_mqa_target_fwd
from ......utils.device import IS_CUDA_AVAILABLE, get_torch_device
from ......utils.import_utils import is_package_available
from ......utils.moe_router_replay import get_active_replay, maybe_replay_indices
from ...module_modeling_base import PretrainedOmniModule
from ..quant_scheme import QUANT_BLOCK_SIZE, SCALE_DTYPE, SCALE_FMT, quantized_storage_dtype
from .configuration import DeepseekV41LlmConfig


_HAS_TRITON = is_package_available("triton")
if _HAS_TRITON:
    import triton
    import triton.language as tl

if is_package_available("tilelang"):
    from ......ops.kernels.dsa.vendor import (
        tilelang_indexer_fwd as _dsa_tl_indexer_fwd,  # noqa: F401
    )


# ---------------------------------------------------------------------------
# DeepSeek-V4 reuse surface & module-level op resolution
# ---------------------------------------------------------------------------


def resolve_module_op_impl(config: Any, field: str, *, variant: str | None = None) -> str | OpsSelection | None:
    """Read one op field off this module's own ``config.ops_implementation``.

    ``None`` falls through to the host default for the ``(op, variant)``.
    An explicit ``"none"`` is returned as written (for families like ``kv_round``).
    Mapping-form per-site selections are converted to :class:`~veomni.ops.OpsSelection`.
    """
    del variant
    ops_impl = getattr(config, "ops_implementation", None)
    if isinstance(ops_impl, dict):
        impl = ops_impl.get(field)
        if isinstance(impl, dict):
            return OpsSelection.from_config_value(impl)
        if impl is not None:
            return impl
    return None


class DeepseekV4RMSNorm(_generated.DeepseekV4RMSNorm):
    """RMS norm with VeOmni's fused-kernel dispatch, respecting module-level ``config.ops_implementation``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, config: Any = None) -> None:
        super().__init__(hidden_size, eps=eps)
        if config is not None:
            self.veomni_rms_norm = VeomniOp(
                "rms_norm",
                "deepseek_v4",
                resolve_module_op_impl(config, "rms_norm_implementation"),
            )


class DeepseekV4UnweightedRMSNorm(_generated.DeepseekV4UnweightedRMSNorm):
    """The mHC input norm -- RMS without a learned weight, respecting module-level ``config.ops_implementation``."""

    def __init__(self, eps: float = 1e-6, config: Any = None) -> None:
        super().__init__(eps=eps)
        if config is not None:
            self.veomni_unweighted_rms_norm = VeomniOp(
                "rms_norm",
                "unweighted",
                resolve_module_op_impl(config, "rms_norm_implementation", variant="unweighted"),
            )


DeepseekV4GroupedLinear = _generated.DeepseekV4GroupedLinear
DeepseekV4Experts = _generated.DeepseekV4Experts
DeepseekV4MLP = _generated.DeepseekV4MLP


# ---------------------------------------------------------------------------
# Storage rounding hooks (``kv_round``)
# ---------------------------------------------------------------------------

WINDOW_KV_FP8_BLOCK = 32
COMPRESS_KV_FP4_GROUP = 16
INDEX_FP4_GROUP = 32


def _default_kv_round_op(variant: str) -> VeomniOp:
    return VeomniOp("kv_round", variant)


def _resolve_kv_round_handle(op: VeomniOp | None, variant: str, *, training: bool) -> VeomniOp | None:
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
    """Compressed-KV storage rounding keyed on ``compress_kv_dtype`` (``'fp4'`` or ``'fp8'``)."""
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


# ---------------------------------------------------------------------------
# Pre-quantized storage helpers & training linear factories
# ---------------------------------------------------------------------------


def is_quantized(weight: torch.Tensor) -> bool:
    """Whether ``weight`` holds pre-quantized FP8 or packed FP4 storage."""
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


def _raise_prequantized_forward_error(module_name: str, op_name: str, op_impl: str) -> None:
    raise ValueError(
        f"{module_name} holds pre-quantized weights (`weight_scale` / packed FP4) and cannot execute under "
        f"{op_name}={op_impl!r}. Switch to a quantized-weight operator or clear `weight_quantization` so the "
        "checkpoint converter dequantizes weights to bf16 on load for training."
    )


def dispatch_linear(op: VeomniOp, linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Dispatch a projection through ``op``, failing fast on pre-quantized storage."""
    if getattr(linear, "is_quantized", False):
        _raise_prequantized_forward_error(type(linear).__name__, "linear_implementation", op.impl)
    return op(linear, x)


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
    """``nn.Linear`` holding optional pre-quantized FP8 weight + ``weight_scale`` storage."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        quant_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        if quant_dtype is not None:
            self._replace_weight_with_quantized(quant_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_quantized:
            _raise_prequantized_forward_error("DeepseekV41Linear", "linear_implementation", "eager")
        return super().forward(x)


class DeepseekV41GroupedLinear(DeepseekV4GroupedLinear, QuantizedWeightMixin):
    """Grouped linear holding optional pre-quantized FP8 weight + ``weight_scale`` storage."""

    def __init__(
        self,
        in_features_per_group: int,
        out_features: int,
        n_groups: int,
        bias: bool = False,
        quant_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features_per_group, out_features, n_groups=n_groups, bias=bias)
        if quant_dtype is None:
            return
        out_per_group = out_features // n_groups
        if out_per_group % QUANT_BLOCK_SIZE:
            raise ValueError(
                f"A group's {out_per_group} output rows do not divide into {QUANT_BLOCK_SIZE}-row "
                "scale tiles, so this weight cannot be sliced per group."
            )
        self._replace_weight_with_quantized(quant_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_quantized:
            _raise_prequantized_forward_error("DeepseekV41GroupedLinear", "linear_implementation", "eager")
        return super().forward(x)


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = False,
    quant_dtype: torch.dtype | None = None,
) -> nn.Linear:
    """Return a plain ``nn.Linear`` for training, or ``DeepseekV41Linear`` when holding pre-quantized FP8."""
    if quant_dtype is None:
        return nn.Linear(in_features, out_features, bias=bias)
    return DeepseekV41Linear(in_features, out_features, bias=bias, quant_dtype=quant_dtype)


def make_grouped_linear(
    in_features_per_group: int,
    out_features: int,
    *,
    n_groups: int,
    bias: bool = False,
    quant_dtype: torch.dtype | None = None,
) -> DeepseekV4GroupedLinear:
    """Return a plain ``DeepseekV4GroupedLinear`` for training, or ``DeepseekV41GroupedLinear`` for FP8 storage."""
    if quant_dtype is None:
        return DeepseekV4GroupedLinear(in_features_per_group, out_features, n_groups=n_groups, bias=bias)
    return DeepseekV41GroupedLinear(
        in_features_per_group, out_features, n_groups=n_groups, bias=bias, quant_dtype=quant_dtype
    )


class DeepseekV41SharedExpert(DeepseekV4MLP):
    """Shared expert MLP dispatched through ``VeomniOp("gated_mlp", "standard", ...)``."""

    def __init__(self, config, quant_dtype: torch.dtype | None = None, *, layer_idx: int = 0) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = make_linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias, quant_dtype=quant_dtype
        )
        self.up_proj = make_linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias, quant_dtype=quant_dtype
        )
        self.down_proj = make_linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias, quant_dtype=quant_dtype
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit
        self.veomni_gated_mlp = VeomniOp(
            "gated_mlp",
            "standard",
            resolve_module_op_impl(config, "gated_mlp_implementation"),
            site=f"layers.{layer_idx}.mlp",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self.gate_proj, "is_quantized", False):
            _raise_prequantized_forward_error(
                "DeepseekV41SharedExpert", "gated_mlp_implementation", self.veomni_gated_mlp.impl
            )
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
        if self.experts_are_quantized:
            _raise_prequantized_forward_error(
                "DeepseekV41QuantizedExperts", "moe_implementation", self.veomni_moe.impl
            )
        unused = self.gate_up_proj.new_empty(0)
        routing_weights = top_k_weights if self.veomni_moe.impl == "eager" else top_k_weights.to(hidden_states.dtype)
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
        )


# ---------------------------------------------------------------------------
# Context-parallel SequenceShard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceShard:
    """Rows ``[start, end)`` of a row of ``full_len`` tokens, ``valid_len`` of them real."""

    group: dist.ProcessGroup
    rank: int
    size: int
    position_ids: torch.Tensor
    valid_len: int

    def __post_init__(self) -> None:
        full_len = self.position_ids.shape[1]
        if full_len % self.size:
            raise ValueError(f"a {full_len}-token row does not split into {self.size} equal shards; pad it first")
        if not 0 < self.valid_len <= full_len:
            raise ValueError(f"valid_len={self.valid_len} is outside the {full_len}-token row")

    @property
    def full_len(self) -> int:
        return int(self.position_ids.shape[1])

    @property
    def local_len(self) -> int:
        return self.full_len // self.size

    @property
    def start(self) -> int:
        return self.rank * self.local_len

    @property
    def end(self) -> int:
        return self.start + self.local_len

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        return gather_outputs(x, gather_dim=1, group=self.group)

    def gather_kv(self, kv: torch.Tensor) -> torch.Tensor:
        return all_gather_kv(kv.unsqueeze(1), self.group).squeeze(1)

    def local_rows(self, x: torch.Tensor) -> torch.Tensor:
        return x.narrow(1, self.start, self.local_len)

    def valid_rows(self, device: torch.device) -> torch.Tensor:
        return torch.arange(self.start, self.end, device=device) < self.valid_len


# ---------------------------------------------------------------------------
# Pipelined Manifold-Constrained Hyper-Connections (mHC)
# ---------------------------------------------------------------------------

POST_MULTIPLIER = 2.0


if _HAS_TRITON:

    @triton.jit
    def _triton_mhc_collapse4_kernel(
        Streams_ptr,
        Pre_ptr,
        Out_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        p0 = tl.load(Pre_ptr + row * 4 + 0).to(tl.float32)
        p1 = tl.load(Pre_ptr + row * 4 + 1).to(tl.float32)
        p2 = tl.load(Pre_ptr + row * 4 + 2).to(tl.float32)
        p3 = tl.load(Pre_ptr + row * 4 + 3).to(tl.float32)

        base = Streams_ptr + row * (4 * D) + offs_d
        s0 = tl.load(base + 0 * D, mask=mask_d, other=0.0).to(tl.float32)
        s1 = tl.load(base + 1 * D, mask=mask_d, other=0.0).to(tl.float32)
        s2 = tl.load(base + 2 * D, mask=mask_d, other=0.0).to(tl.float32)
        s3 = tl.load(base + 3 * D, mask=mask_d, other=0.0).to(tl.float32)

        acc = ((p0 * s0 + p1 * s1) + p2 * s2) + p3 * s3
        tl.store(Out_ptr + row * D + offs_d, acc.to(Out_ptr.dtype.element_ty), mask=mask_d)

    @triton.jit
    def _triton_mhc_collapse4_bwd_kernel(
        GradOut_ptr,
        Streams_ptr,
        Pre_ptr,
        GradStreams_ptr,
        GradPre_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        g = tl.load(GradOut_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        p0 = tl.load(Pre_ptr + row * 4 + 0).to(tl.float32)
        p1 = tl.load(Pre_ptr + row * 4 + 1).to(tl.float32)
        p2 = tl.load(Pre_ptr + row * 4 + 2).to(tl.float32)
        p3 = tl.load(Pre_ptr + row * 4 + 3).to(tl.float32)

        base = Streams_ptr + row * (4 * D) + offs_d
        s0 = tl.load(base + 0 * D, mask=mask_d, other=0.0).to(tl.float32)
        s1 = tl.load(base + 1 * D, mask=mask_d, other=0.0).to(tl.float32)
        s2 = tl.load(base + 2 * D, mask=mask_d, other=0.0).to(tl.float32)
        s3 = tl.load(base + 3 * D, mask=mask_d, other=0.0).to(tl.float32)

        g_base = GradStreams_ptr + row * (4 * D) + offs_d
        tl.store(g_base + 0 * D, (g * p0).to(GradStreams_ptr.dtype.element_ty), mask=mask_d)
        tl.store(g_base + 1 * D, (g * p1).to(GradStreams_ptr.dtype.element_ty), mask=mask_d)
        tl.store(g_base + 2 * D, (g * p2).to(GradStreams_ptr.dtype.element_ty), mask=mask_d)
        tl.store(g_base + 3 * D, (g * p3).to(GradStreams_ptr.dtype.element_ty), mask=mask_d)

        dp0 = tl.sum(g * s0, axis=0)
        dp1 = tl.sum(g * s1, axis=0)
        dp2 = tl.sum(g * s2, axis=0)
        dp3 = tl.sum(g * s3, axis=0)
        tl.atomic_add(GradPre_ptr + row * 4 + 0, dp0)
        tl.atomic_add(GradPre_ptr + row * 4 + 1, dp1)
        tl.atomic_add(GradPre_ptr + row * 4 + 2, dp2)
        tl.atomic_add(GradPre_ptr + row * 4 + 3, dp3)


class DeepseekV41HyperConnection(nn.Module):
    """One site's mHC coefficients: ``(pre, post, comb)`` from the stream itself."""

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps, config=config)
        self.veomni_mhc_post = VeomniOp("mhc", "post", resolve_module_op_impl(config, "mhc_implementation"))
        self._use_tile_kernels = resolve_module_op_impl(
            config, "mhc_implementation"
        ) == "tilelang" and is_package_available("tile_kernels")
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))
        self._initializer_range = config.initializer_range
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.fn, mean=0.0, std=self._initializer_range)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, S, hc, D]`` -> ``pre[B, S, hc]``, ``post[B, S, hc]``, ``comb[B, S, hc, hc]``."""
        hc = self.hc_mult
        if (
            torch.is_grad_enabled()
            and IS_CUDA_AVAILABLE
            and getattr(self, "_use_tile_kernels", False)
            and hidden_streams.is_cuda
            and hidden_streams.dtype == torch.bfloat16
            and hidden_streams.ndim == 4
            and hc == 4
            and self.hc_sinkhorn_iters >= 1
        ):
            from tile_kernels.modeling.mhc.ops import (
                mhc_pre_norm_fn,
                mhc_pre_split_mixes,
                sinkhorn_normalize,
            )

            seq_len = hidden_streams.shape[1]
            pad_s = -seq_len % 128
            hs_in = F.pad(hidden_streams, (0, 0, 0, 0, 0, pad_s)) if pad_s else hidden_streams.contiguous()
            mixes = mhc_pre_norm_fn(
                hs_in,
                self.fn.float().contiguous(),
                None,
                float(self.input_norm.eps),
                fuse_grad_acc=False,
            )
            pre, post, comb = mhc_pre_split_mixes(
                mixes,
                self.scale.float().contiguous(),
                self.base.float().contiguous(),
                hc,
                POST_MULTIPLIER,
                self.hc_eps,
            )
            comb = sinkhorn_normalize(comb, repeat=self.hc_sinkhorn_iters, eps=self.hc_eps)
            if pad_s:
                pre = pre[:, :seq_len]
                post = post[:, :seq_len]
                comb = comb[:, :seq_len]
            return pre.squeeze(-1), post.squeeze(-1), comb

        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = POST_MULTIPLIER * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb


class _MhcCollapse4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        lead_shape = hidden_streams.shape[:-2]
        d = hidden_streams.shape[-1]
        m = hidden_streams.numel() // (4 * d)
        streams_c = hidden_streams.contiguous()
        pre_c = pre.contiguous()
        out = torch.empty((*lead_shape, d), dtype=hidden_streams.dtype, device=hidden_streams.device)
        if m > 0:
            with get_torch_device().device(hidden_streams.device):
                _triton_mhc_collapse4_kernel[(m, triton.cdiv(d, 1024))](
                    streams_c,
                    pre_c,
                    out,
                    D=d,
                    BLOCK_D=1024,
                    enable_fp_fusion=False,
                )
        ctx.save_for_backward(streams_c, pre_c)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        streams_c, pre_c = ctx.saved_tensors
        d = streams_c.shape[-1]
        m = streams_c.numel() // (4 * d)
        grad_out_c = grad_output.contiguous()
        grad_streams = torch.empty_like(streams_c)
        grad_pre = torch.zeros_like(pre_c, dtype=torch.float32)
        if m > 0:
            with get_torch_device().device(streams_c.device):
                _triton_mhc_collapse4_bwd_kernel[(m, triton.cdiv(d, 1024))](
                    grad_out_c,
                    streams_c,
                    pre_c,
                    grad_streams,
                    grad_pre,
                    D=d,
                    BLOCK_D=1024,
                    enable_fp_fusion=False,
                )
        return grad_streams, grad_pre.to(pre_c.dtype) if ctx.needs_input_grad[1] else None


def mhc_collapse(hidden_streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Weight the ``hc_mult`` copies down into one sublayer input."""
    if (
        IS_CUDA_AVAILABLE
        and _HAS_TRITON
        and hidden_streams.is_cuda
        and hidden_streams.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and pre.dtype == torch.float32
        and hidden_streams.shape[-2] == 4
    ):
        if torch.is_grad_enabled() and (hidden_streams.requires_grad or pre.requires_grad):
            return _MhcCollapse4Function.apply(hidden_streams, pre)
        lead_shape = hidden_streams.shape[:-2]
        d = hidden_streams.shape[-1]
        m = hidden_streams.numel() // (4 * d)
        out = torch.empty((*lead_shape, d), dtype=hidden_streams.dtype, device=hidden_streams.device)
        if m > 0:
            with get_torch_device().device(hidden_streams.device):
                _triton_mhc_collapse4_kernel[(m, triton.cdiv(d, 1024))](
                    hidden_streams.contiguous(),
                    pre.contiguous(),
                    out,
                    D=d,
                    BLOCK_D=1024,
                    enable_fp_fusion=False,
                )
        return out
    return (pre.unsqueeze(-1) * hidden_streams.float()).sum(dim=2).to(hidden_streams.dtype)


def mhc_expand(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    op: VeomniOp | None = None,
) -> torch.Tensor:
    """Place a sublayer output back across the copies and mix the residual in."""
    if op is None:
        op = VeomniOp("mhc", "post")
    return op(output, residual, post, comb)


def require_fp32_pre_mix(pre_mix: torch.Tensor) -> None:
    """Refuse a ``pre`` that something outside the model narrowed."""
    if torch.finfo(pre_mix.dtype).bits < 32:
        raise RuntimeError(
            f"the hyper-connection pre mix reached a block as {pre_mix.dtype}, narrower than float32. Under FSDP2 mixed "
            "precision set `accelerator.fsdp_config.mixed_precision.cast_forward_inputs: false` for the "
            "DeepSeek-V4.1 llm."
        )


def identity_pre_mix(hidden_streams: torch.Tensor, hc_mult: int) -> torch.Tensor:
    pre_mix = hidden_streams.new_zeros(hidden_streams.shape[0], hidden_streams.shape[1], hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


def expand_to_streams(hidden_states: torch.Tensor, hc_mult: int) -> torch.Tensor:
    return hidden_states.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()


# ---------------------------------------------------------------------------
# Engram residual-stream connection
# ---------------------------------------------------------------------------


class DeepseekV41EngramConnection(nn.Module):
    """Write one engram layer's gathered rows into the mHC residual stream."""

    clamp_value: float = 1e-6

    def __init__(self, config: DeepseekV41LlmConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_hash_index = config.engram_layer_ids.index(layer_idx)
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps

        n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.wkv = make_linear(
            n_hash_cols * config.engram_head_dim,
            config.hidden_size * (config.hc_mult + 1),
            bias=False,
            quant_dtype=linear_quant_dtype(config),
        )
        self.veomni_wkv = VeomniOp(
            "linear",
            "standard",
            resolve_module_op_impl(config, "linear_implementation"),
            site=f"layers.{layer_idx}.engram_wkv",
        )
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def forward(
        self,
        hidden_streams: torch.Tensor,
        ngram_embeds: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv = dispatch_linear(self.veomni_wkv, self.wkv, ngram_embeds.flatten(-2))
        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))

        weight = self.q_weight.float() * self.k_weight.float()
        stream = hidden_streams.float()
        rstd = torch.rsqrt(stream.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (stream * weight * key).sum(-1) * rstd * self.hidden_size**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (stream + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden_streams.dtype)


def engram_layer_order(config: DeepseekV41LlmConfig) -> list[int]:
    return list(config.engram_layer_ids)


# ---------------------------------------------------------------------------
# Hybrid sliding-window + compressed-KV attention & two-level DSA indexer
# ---------------------------------------------------------------------------

INDEXER_QUERY_CHUNK = 512


@dataclass
class SharedAttentionState:
    """What a source layer publishes for the layers that follow it."""

    compress_kv: torch.Tensor | None = None
    compress_ratio: int | None = None
    index_k: torch.Tensor | None = None
    topk_idxs: torch.Tensor | None = None
    candidates: torch.Tensor | None = None
    window_indices: torch.Tensor | None = None
    main_rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None
    compress_rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class IndexerLossTerms:
    """One layer's contribution to the indexer objective."""

    kl: torch.Tensor
    uniform: torch.Tensor


def indexer_kl_terms(index_score: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query ``KL(target || softmax(index_score))``, and its uniform reference."""
    scoreable = torch.isfinite(index_score)
    nothing_to_learn = ~scoreable.any(-1, keepdim=True) | (target.sum(-1, keepdim=True) <= 0)
    scores = torch.where(nothing_to_learn, 0.0, index_score.float())
    log_q = torch.log_softmax(scores, dim=-1)
    log_target = torch.log(target.clamp_min(torch.finfo(torch.float32).tiny))
    contributions = torch.where(target > 0, target * (log_target - log_q), 0.0)
    neg_entropy = (target * log_target).sum(-1)
    uniform_kl = torch.where(
        nothing_to_learn.squeeze(-1),
        0.0,
        torch.log(scoreable.sum(-1).clamp_min(1).to(torch.float32)) + neg_entropy,
    )
    return contributions.sum(-1), uniform_kl.detach()


@dataclass
class CompressionPlan:
    """Which tokens form each compression group, and which groups a query sees."""

    ratio: int
    window_indices: torch.Tensor
    window_positions: torch.Tensor
    range_starts: torch.Tensor
    range_ends: torch.Tensor

    @property
    def compressed_len(self) -> int:
        return int(self.window_indices.shape[0])

    def visibility_mask(self, device: torch.device) -> torch.Tensor:
        groups = torch.arange(self.compressed_len, device=device)
        return (groups >= self.range_starts.unsqueeze(-1)) & (groups < self.range_ends.unsqueeze(-1))

    def narrow_queries(self, start: int, end: int) -> CompressionPlan:
        return CompressionPlan(
            ratio=self.ratio,
            window_indices=self.window_indices,
            window_positions=self.window_positions,
            range_starts=self.range_starts[start:end],
            range_ends=self.range_ends[start:end],
        )


def sequence_slices_from_positions(position_ids: torch.Tensor) -> tuple[tuple[int, int], ...]:
    """Recover the packed samples' ``[start, end)`` spans from their positions."""
    if position_ids.ndim != 2:
        raise ValueError(f"expected [batch, seq_len] position ids, got {position_ids.ndim} dims")
    seq_len = position_ids.shape[1]
    cached_slices = getattr(position_ids, "_sequence_slices", None)
    if cached_slices is not None and cached_slices and cached_slices[-1][1] == seq_len:
        return cached_slices
    starts = (position_ids[0] == 0).nonzero(as_tuple=True)[0].tolist()
    if not starts or starts[0] != 0:
        raise ValueError(
            "position ids must start a sample at index 0; got a row whose first position is "
            f"{int(position_ids[0, 0])}. Decode-time offsets are not a training input."
        )
    ends = starts[1:] + [seq_len]
    return tuple(zip(starts, ends))


def trailing_partial_group(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    ratio: int,
) -> torch.Tensor | None:
    if ratio <= 1:
        return None
    pending = None
    for start, end in sequence_slices_from_positions(position_ids):
        remainder = (end - start) % ratio
        pending = hidden_states[:, end - remainder : end] if remainder else None
    return pending


def build_compression_plans(
    config: DeepseekV41LlmConfig,
    position_ids: torch.Tensor,
    sequence_slices: Sequence[tuple[int, int]] | None = None,
) -> dict[int, CompressionPlan]:
    """One plan per distinct non-zero ``compress_ratio``, for this forward."""
    if position_ids.ndim != 2:
        raise ValueError(
            f"DeepSeek-V4.1 compression needs [batch, seq_len] position ids, got {position_ids.ndim} dims"
        )
    batch_size, seq_len = position_ids.shape
    if batch_size > 1 and not bool((position_ids == position_ids[:1]).all()):
        raise ValueError(
            "DeepSeek-V4.1 compression requires every batch row to share one position layout; "
            "pack the samples into a single row instead of batching rows with different positions"
        )
    if sequence_slices is None:
        sequence_slices = ((0, seq_len),)

    device = position_ids.device
    positions = position_ids[0]
    slice_begins = torch.tensor([begin for begin, _ in sequence_slices], device=device, dtype=torch.long)
    slice_lengths = torch.tensor([end - begin for begin, end in sequence_slices], device=device, dtype=torch.long)
    plans: dict[int, CompressionPlan] = {}
    for ratio in sorted({r for r in config.compress_ratios if r > 0}):
        group_counts = slice_lengths // ratio
        compressed_offsets = torch.cumsum(group_counts, dim=0) - group_counts
        total_groups = int(sum((end - begin) // ratio for begin, end in sequence_slices))
        if total_groups > 0:
            group_slice_begins = torch.repeat_interleave(slice_begins, group_counts, output_size=total_groups)
            group_compressed_offsets = torch.repeat_interleave(
                compressed_offsets, group_counts, output_size=total_groups
            )
            group_idx_in_slice = torch.arange(total_groups, device=device, dtype=torch.long) - group_compressed_offsets
            window_starts = group_slice_begins + group_idx_in_slice * ratio
        else:
            window_starts = torch.empty(0, device=device, dtype=torch.long)
        offsets = torch.arange(ratio, device=device, dtype=torch.long)
        window_indices = window_starts[:, None] + offsets[None, :]

        if seq_len > 0:
            range_starts = torch.repeat_interleave(
                compressed_offsets.to(torch.int32), slice_lengths, output_size=seq_len
            )
            max_groups = torch.repeat_interleave(group_counts.to(torch.int32), slice_lengths, output_size=seq_len)
            visible = torch.minimum(((positions + 1) // ratio).to(torch.int32), max_groups)
            range_ends = range_starts + visible
        else:
            range_starts = torch.empty(0, device=device, dtype=torch.int32)
            range_ends = torch.empty(0, device=device, dtype=torch.int32)

        plans[ratio] = CompressionPlan(
            ratio=ratio,
            window_indices=window_indices,
            window_positions=positions[window_starts] if window_starts.numel() else window_starts,
            range_starts=range_starts,
            range_ends=range_ends,
        )
    return plans


def rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


if _HAS_TRITON:

    @triton.jit
    def _triton_interleaved_rope_inplace_kernel(
        X_ptr,
        Cos_ptr,
        Sin_ptr,
        STRIDE_M,
        H,
        D: tl.constexpr,
        ROPE_HALF: tl.constexpr,
        CONJUGATE: tl.constexpr,
    ):
        m = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1).to(tl.int64)
        p = tl.arange(0, ROPE_HALF)
        c = tl.load(Cos_ptr + m * ROPE_HALF + p).to(tl.float32)
        s = tl.load(Sin_ptr + m * ROPE_HALF + p).to(tl.float32)
        if CONJUGATE:
            s = -s
        base = X_ptr + m * STRIDE_M + h * D + (D - 2 * ROPE_HALF)
        even = tl.load(base + 2 * p).to(tl.float32)
        odd = tl.load(base + 2 * p + 1).to(tl.float32)
        rot_even = (even * c + (-odd) * s).to(X_ptr.dtype.element_ty)
        rot_odd = (odd * c + even * s).to(X_ptr.dtype.element_ty)
        tl.store(base + 2 * p, rot_even)
        tl.store(base + 2 * p + 1, rot_odd)


class _InterleavedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, conjugate: bool) -> torch.Tensor:
        r_half = cos.shape[-1]
        d = x.shape[-1]
        h = x.shape[2] if x.ndim == 4 else 1
        m = x.numel() // (h * d)
        out = x.contiguous().clone()
        cos_c = cos if cos.is_contiguous() else cos.contiguous()
        sin_c = sin if sin.is_contiguous() else sin.contiguous()
        stride_m = h * d
        with get_torch_device().device(x.device):
            _triton_interleaved_rope_inplace_kernel[(m, h)](
                out,
                cos_c,
                sin_c,
                stride_m,
                h,
                D=d,
                ROPE_HALF=r_half,
                CONJUGATE=conjugate,
                num_warps=1,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(cos_c, sin_c)
        ctx.conjugate = conjugate
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        cos_c, sin_c = ctx.saved_tensors
        r_half = cos_c.shape[-1]
        d = grad_output.shape[-1]
        h = grad_output.shape[2] if grad_output.ndim == 4 else 1
        m = grad_output.numel() // (h * d)
        dx = grad_output.contiguous().clone()
        stride_m = h * d
        with get_torch_device().device(grad_output.device):
            _triton_interleaved_rope_inplace_kernel[(m, h)](
                dx,
                cos_c,
                sin_c,
                stride_m,
                h,
                D=d,
                ROPE_HALF=r_half,
                CONJUGATE=not ctx.conjugate,
                num_warps=1,
                enable_fp_fusion=False,
            )
        return dx, None, None, None


def apply_interleaved_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Rotate the trailing ``2 * cos.shape[-1]`` channels of ``x``, leaving NoPE alone."""
    if (
        IS_CUDA_AVAILABLE
        and _HAS_TRITON
        and x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.ndim in (2, 3, 4)
        and not cos.requires_grad
        and not sin.requires_grad
    ):
        r_half = cos.shape[-1]
        d = x.shape[-1]
        if r_half > 0 and (r_half & (r_half - 1)) == 0 and 2 * r_half <= d:
            h = x.shape[2] if x.ndim == 4 else 1
            m = x.numel() // (h * d)
            if m > 0 and cos.numel() == m * r_half and sin.numel() == m * r_half:
                if torch.is_grad_enabled() and x.requires_grad:
                    return _InterleavedRoPEFunction.apply(x, cos, sin, conjugate)
                can_strided_inplace = (
                    inplace
                    and not torch.is_grad_enabled()
                    and x.stride(-1) == 1
                    and (h == 1 or x.stride(-2) == d)
                    and (x.is_contiguous() or x.ndim == 2 or (x.ndim in (3, 4) and x.shape[0] == 1))
                )
                out = x if can_strided_inplace else x.contiguous().clone()
                if out.is_contiguous() or m == 1:
                    stride_m = h * d
                elif out.ndim == 2:
                    stride_m = out.stride(0)
                else:
                    stride_m = out.stride(1)
                with get_torch_device().device(x.device):
                    _triton_interleaved_rope_inplace_kernel[(m, h)](
                        out,
                        cos if cos.is_contiguous() else cos.contiguous(),
                        sin if sin.is_contiguous() else sin.contiguous(),
                        stride_m,
                        h,
                        D=d,
                        ROPE_HALF=r_half,
                        CONJUGATE=conjugate,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
                return out

    cos = cos.repeat_interleave(2, dim=-1)
    sin = (-sin if conjugate else sin).repeat_interleave(2, dim=-1)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = (rope.float() * cos + rotate_half_interleaved(rope).float() * sin).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


class DeepseekV41RotaryEmbedding(nn.Module):
    """The two rotary schedules V4.1 uses, selected per layer by compress ratio."""

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__()
        scaling = config.rope_scaling or {}
        self.register_buffer(
            "main_inv_freq",
            self._inv_freq(config.qk_rope_head_dim, config.rope_theta),
            persistent=False,
        )
        self.register_buffer(
            "compress_inv_freq",
            self._inv_freq(
                config.qk_rope_head_dim,
                config.compress_rope_theta,
                original_seq_len=int(scaling.get("original_max_position_embeddings", 0)),
                factor=float(scaling.get("factor", 1.0)),
                beta_fast=float(scaling.get("beta_fast", 32.0)),
                beta_slow=float(scaling.get("beta_slow", 1.0)),
            ),
            persistent=False,
        )

    @staticmethod
    def _inv_freq(
        dim: int,
        base: float,
        original_seq_len: int = 0,
        factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> torch.Tensor:
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if original_seq_len <= 0:
            return freqs

        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        return freqs / factor * (1 - smooth) + freqs * smooth

    def forward(self, positions: torch.Tensor, compressed: bool) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.compress_inv_freq if compressed else self.main_inv_freq
        freqs = positions.float().unsqueeze(-1) * inv_freq
        return freqs.cos(), freqs.sin()


class DeepseekV41Compressor(nn.Module):
    """Pool ``compress_ratio`` consecutive tokens into one KV latent."""

    def __init__(self, config: DeepseekV41LlmConfig, layer_idx: int) -> None:
        super().__init__()
        self.compress_ratio = config.compress_ratios[layer_idx]
        if self.compress_ratio < 1:
            raise ValueError(f"Layer {layer_idx} has compress_ratio {self.compress_ratio} and cannot compress")
        self.head_dim = config.head_dim
        self.norm = DeepseekV4RMSNorm(config.head_dim, eps=config.rms_norm_eps, config=config)
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False) if self.compress_ratio > 1 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        plan: CompressionPlan,
        shard: SequenceShard | None = None,
    ) -> torch.Tensor:
        if plan.ratio != self.compress_ratio:
            raise ValueError(
                f"Compression plan is for ratio {plan.ratio}, this layer compresses by {self.compress_ratio}"
            )
        if plan.compressed_len == 0:
            return hidden_states.new_zeros((hidden_states.shape[0], 0, self.head_dim))

        if self.wgate is None:
            if shard is None:
                return self.norm(self.wkv(hidden_states[:, plan.window_indices[:, 0]]))
            return self.norm(shard.gather(self.wkv(hidden_states))[:, plan.window_indices[:, 0]])

        x = hidden_states.float()
        kv = torch.nn.functional.linear(x, self.wkv.weight.float())
        gate = torch.nn.functional.linear(x, self.wgate.weight.float())
        if shard is not None:
            kv, gate = shard.gather(kv), shard.gather(gate)
        kv, gate = kv[:, plan.window_indices], gate[:, plan.window_indices]
        pooled = (kv * gate.softmax(dim=2)).sum(dim=2)
        return self.norm(pooled.to(hidden_states.dtype))


def select_candidate_blocks(
    scores: torch.Tensor,
    range_ends: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    *,
    return_block_mask: bool = False,
) -> torch.Tensor:
    """Level one of the two-level selection: the best ``topk_blocks`` blocks per query."""
    width = scores.shape[-1]
    pad_w = -width % block_size
    padded = torch.nn.functional.pad(scores, (0, pad_w), value=-torch.inf) if pad_w else scores
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = block_scores.shape[-1]

    last = (range_ends.to(torch.long) - 1) // block_size
    block_ids = torch.arange(num_blocks, device=scores.device)
    block_scores = block_scores.masked_fill(block_ids == last.unsqueeze(-1), torch.inf)

    top = block_scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    if return_block_mask:
        return keep
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _mask_scores_by_candidates(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    block_size: int,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    width = scores.shape[-1]
    if candidates.shape[-1] == width:
        return scores.masked_fill_(~candidates, -torch.inf) if inplace else scores.masked_fill(~candidates, -torch.inf)
    num_blocks = candidates.shape[-1]
    if width == num_blocks * block_size:
        target = scores if inplace else scores.clone()
        target.view(*scores.shape[:-1], num_blocks, block_size).masked_fill_(~candidates.unsqueeze(-1), -torch.inf)
        return target
    expanded = candidates.repeat_interleave(block_size, dim=-1)[..., :width]
    return scores.masked_fill_(~expanded, -torch.inf) if inplace else scores.masked_fill(~expanded, -torch.inf)


def _resolve_kv_round_impl(config: DeepseekV41LlmConfig):
    impl = resolve_module_op_impl(config, "kv_round_implementation")
    return "none" if impl is None else impl


class _V41TileLangIndexerStudentFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        index_k: torch.Tensor,
        weights_fp32: torch.Tensor,
        topk_idxs: torch.Tensor,
        range_starts: torch.Tensor,
        range_ends: torch.Tensor,
        student_scores: torch.Tensor,
    ) -> torch.Tensor:
        q_sbhd = query.transpose(0, 1).contiguous()
        k_sbd = index_k.transpose(0, 1).contiguous()
        w_sbh = weights_fp32.transpose(0, 1).contiguous()
        ctx.save_for_backward(q_sbhd, k_sbd, w_sbh, topk_idxs, range_starts, range_ends)
        return student_scores

    @staticmethod
    def backward(ctx, grad_student: torch.Tensor):
        from ......ops.kernels.dsa.vendor.tilelang_indexer_bwd import batched_indexer_bwd

        q_sbhd, k_sbd, w_sbh, topk_idxs, range_starts, range_ends = ctx.saved_tensors
        grad_q_sbhd, grad_w_sbh, grad_k_sbd = batched_indexer_bwd(
            q_sbhd,
            w_sbh,
            k_sbd,
            topk_idxs,
            grad_student.contiguous(),
            cu_seqlen_ks=range_starts,
            cu_seqlen_ke=range_ends,
        )
        grad_q = grad_q_sbhd.transpose(0, 1).contiguous()
        grad_k = grad_k_sbd.transpose(0, 1).to(q_sbhd.dtype).contiguous()
        grad_w = grad_w_sbh.transpose(0, 1).contiguous()
        return grad_q, grad_k, grad_w, None, None, None, None


class DeepseekV41Indexer(nn.Module):
    """Pick the ``index_topk`` most promising compressed entries for each query."""

    def __init__(self, config: DeepseekV41LlmConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.owns_keys = layer_idx in config.kv_source_layer_ids
        self.is_candidate_source = layer_idx == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_idx
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.index_topk = config.index_topk
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = make_linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
            quant_dtype=linear_quant_dtype(config),
        )
        self.veomni_wq_b = VeomniOp(
            "linear",
            "standard",
            resolve_module_op_impl(config, "linear_implementation"),
            site=f"layers.{layer_idx}.indexer_q_b_proj",
        )
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        if self.owns_keys:
            self.wk = nn.Linear(config.head_dim, config.index_head_dim, bias=False)
            self.k_norm = DeepseekV4RMSNorm(config.index_head_dim, eps=config.rms_norm_eps, config=config)
        self.veomni_kv_round_fp4 = VeomniOp(
            "kv_round",
            "fp4",
            _resolve_kv_round_impl(config),
            site=f"layers.{layer_idx}.indexer",
        )
        indexer_impl = resolve_module_op_impl(config, "dsa_indexer_implementation")
        self.dsa_indexer_implementation = indexer_impl.name if isinstance(indexer_impl, OpsSelection) else indexer_impl

    def build_keys(
        self,
        latent: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        latent = latent.detach()
        keys = self.k_norm(self.wk(latent))
        keys = apply_interleaved_rope(keys, cos, sin)
        return round_index_act(
            keys,
            training=self.training,
            op=self.veomni_kv_round_fp4,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        index_k: torch.Tensor,
        plan: CompressionPlan,
        shared: SharedAttentionState,
        cos: torch.Tensor,
        sin: torch.Tensor,
        return_scores: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        compressed_len = plan.compressed_len

        hidden_states = hidden_states.detach()
        q_residual = q_residual.detach()

        with torch.set_grad_enabled(return_scores and torch.is_grad_enabled()):
            query = dispatch_linear(self.veomni_wq_b, self.wq_b, q_residual).unflatten(
                -1, (self.n_heads, self.head_dim)
            )
            query = apply_interleaved_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2))
            query = round_index_act(
                query,
                training=self.training,
                op=self.veomni_kv_round_fp4,
            )
            weights = self.weights_proj(hidden_states) * (self.softmax_scale * self.n_heads**-0.5)

        topk = min(self.index_topk, compressed_len)
        use_tl_indexer = (
            self.dsa_indexer_implementation == "tilelang"
            and IS_CUDA_AVAILABLE
            and query.is_cuda
            and query.dtype == torch.bfloat16
            and index_k.dtype == torch.bfloat16
            and compressed_len > 0
            and self.n_heads <= 64
            and self.n_heads % 8 == 0
            and self.head_dim >= 32
        )
        if use_tl_indexer:
            weights_fp32 = weights.float()
            with torch.no_grad():
                q_sbhd = query.detach().transpose(0, 1).contiguous()
                k_sbd = index_k.detach().transpose(0, 1).contiguous()
                w_sbh = weights_fp32.detach().transpose(0, 1).contiguous()
                if self.uses_candidates and shared.candidates is None:
                    raise RuntimeError(
                        f"Layer {self.layer_idx} indexes inside the candidate blocks but none were published; "
                        "candidate_source_layer_id must name a layer in index_source_layer_ids"
                    )
                tl_chunk = 1024 if compressed_len > 32768 else seq_len
                if seq_len <= tl_chunk:
                    scores = torch.ops.DSA.lighting_indexer_fwd(
                        q_sbhd, k_sbd, w_sbh, plan.range_starts, plan.range_ends
                    )
                    ranked = scores
                    if self.is_candidate_source:
                        shared.candidates = select_candidate_blocks(
                            scores,
                            plan.range_ends,
                            self.candidate_topk_blocks,
                            self.candidate_block_size,
                            return_block_mask=True,
                        )
                    elif self.uses_candidates:
                        assert shared.candidates is not None
                        ranked = _mask_scores_by_candidates(
                            scores,
                            shared.candidates,
                            self.candidate_block_size,
                            inplace=not return_scores,
                        )

                    topk_res = ranked.topk(topk, dim=-1, sorted=False)
                    selected, order = topk_res.indices.sort(dim=-1)
                    reachable = (selected >= plan.range_starts.view(1, -1, 1)) & (
                        selected < plan.range_ends.view(1, -1, 1)
                    )
                    if self.uses_candidates:
                        reachable = reachable & torch.isfinite(topk_res.values.gather(-1, order))
                    topk_idxs = selected.to(torch.int32).masked_fill(~reachable, -1)
                    student_scores = (
                        scores.gather(-1, selected).masked_fill(~reachable, -torch.inf) if return_scores else None
                    )
                    del scores, ranked, topk_res, order
                else:
                    topk_idxs = torch.empty((batch, seq_len, topk), dtype=torch.int32, device=device)
                    student_scores = (
                        torch.empty((batch, seq_len, topk), dtype=torch.float32, device=device)
                        if return_scores
                        else None
                    )
                    num_blocks = (compressed_len + self.candidate_block_size - 1) // self.candidate_block_size
                    candidates_buf = (
                        torch.empty((batch, seq_len, num_blocks), dtype=torch.bool, device=device)
                        if self.is_candidate_source
                        else None
                    )
                    for begin in range(0, seq_len, tl_chunk):
                        end = min(begin + tl_chunk, seq_len)
                        rs_chunk = plan.range_starts[begin:end]
                        re_chunk = plan.range_ends[begin:end]
                        scores_chunk = torch.ops.DSA.lighting_indexer_fwd(
                            q_sbhd[begin:end], k_sbd, w_sbh[begin:end], rs_chunk, re_chunk
                        )
                        ranked_chunk = scores_chunk
                        if self.is_candidate_source:
                            assert candidates_buf is not None
                            candidates_buf[:, begin:end] = select_candidate_blocks(
                                scores_chunk,
                                re_chunk,
                                self.candidate_topk_blocks,
                                self.candidate_block_size,
                                return_block_mask=True,
                            )
                        elif self.uses_candidates:
                            assert shared.candidates is not None
                            ranked_chunk = _mask_scores_by_candidates(
                                scores_chunk,
                                shared.candidates[:, begin:end],
                                self.candidate_block_size,
                                inplace=not return_scores,
                            )
                        topk_chunk = ranked_chunk.topk(topk, dim=-1, sorted=False)
                        sel_chunk, order_chunk = topk_chunk.indices.sort(dim=-1)
                        reach_chunk = (sel_chunk >= rs_chunk.view(1, -1, 1)) & (sel_chunk < re_chunk.view(1, -1, 1))
                        if self.uses_candidates:
                            reach_chunk = reach_chunk & torch.isfinite(topk_chunk.values.gather(-1, order_chunk))
                        topk_idxs[:, begin:end] = sel_chunk.to(torch.int32).masked_fill(~reach_chunk, -1)
                        if student_scores is not None:
                            student_scores[:, begin:end] = scores_chunk.gather(-1, sel_chunk).masked_fill(
                                ~reach_chunk, -torch.inf
                            )
                        del scores_chunk, ranked_chunk, topk_chunk, sel_chunk, order_chunk, reach_chunk
                    if self.is_candidate_source:
                        shared.candidates = candidates_buf
            if not return_scores:
                return topk_idxs, None
            student = _V41TileLangIndexerStudentFunction.apply(
                query,
                index_k,
                weights_fp32,
                topk_idxs,
                plan.range_starts,
                plan.range_ends,
                student_scores,
            )
            return topk_idxs, student

        visible = plan.visibility_mask(device)
        with torch.no_grad():
            scores = torch.empty(batch, seq_len, compressed_len, device=device, dtype=torch.float32)
            for begin in range(0, seq_len, INDEXER_QUERY_CHUNK):
                end = min(begin + INDEXER_QUERY_CHUNK, seq_len)
                chunk = torch.einsum("bshd,btd->bsht", query[:, begin:end], index_k)
                chunk = (chunk.float().relu() * weights[:, begin:end].float().unsqueeze(-1)).sum(dim=2)
                scores[:, begin:end] = chunk.masked_fill(~visible[begin:end], -torch.inf)

            ranked = scores
            if self.is_candidate_source:
                shared.candidates = select_candidate_blocks(
                    scores, plan.range_ends, self.candidate_topk_blocks, self.candidate_block_size
                )
            elif self.uses_candidates:
                if shared.candidates is None:
                    raise RuntimeError(
                        f"Layer {self.layer_idx} indexes inside the candidate blocks but none were published; "
                        "candidate_source_layer_id must name a layer in index_source_layer_ids"
                    )
                ranked = scores.masked_fill(~shared.candidates, -torch.inf)

            topk_res = ranked.topk(topk, dim=-1, sorted=False)
            selected, order = topk_res.indices.sort(dim=-1)
            reachable = visible.unsqueeze(0).expand(batch, -1, -1).gather(-1, selected)
            if self.uses_candidates:
                reachable = reachable & torch.isfinite(topk_res.values.gather(-1, order))
            topk_idxs = selected.to(torch.int32).masked_fill(~reachable, -1)
            del scores, ranked, topk_res, order
        if not return_scores:
            return topk_idxs, None
        student_chunks = []
        for begin in range(0, seq_len, INDEXER_QUERY_CHUNK):
            end = min(begin + INDEXER_QUERY_CHUNK, seq_len)
            sel_chunk = selected[:, begin:end]
            k_sel = index_k.gather(1, sel_chunk.reshape(batch, -1, 1).expand(-1, -1, index_k.size(-1))).unflatten(
                1, (end - begin, topk)
            )
            q_chunk = query[:, begin:end]
            w_chunk = weights[:, begin:end].float().unsqueeze(-1)
            s_chunk = (torch.einsum("bshd,bskd->bshk", q_chunk, k_sel).float().relu() * w_chunk).sum(dim=2)
            student_chunks.append(s_chunk)
        student = torch.cat(student_chunks, dim=1).masked_fill(~reachable, -torch.inf)
        return topk_idxs, student


class DeepseekV41Attention(nn.Module):
    """One layer's attention over the sliding window plus the compressed history."""

    def __init__(self, config: DeepseekV41LlmConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.sliding_window = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.softmax_scale = config.head_dim**-0.5

        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        quant_dtype = linear_quant_dtype(config)
        linear_impl = resolve_module_op_impl(config, "linear_implementation")

        self.wq_a = make_linear(config.hidden_size, config.q_lora_rank, bias=False, quant_dtype=quant_dtype)
        self.veomni_wq_a = VeomniOp("linear", "standard", linear_impl, site=f"layers.{layer_idx}.q_a_proj")
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps, config=config)

        self.wq_b = make_linear(
            config.q_lora_rank, self.n_heads * config.head_dim, bias=False, quant_dtype=quant_dtype
        )
        self.veomni_wq_b = VeomniOp("linear", "standard", linear_impl, site=f"layers.{layer_idx}.q_b_proj")

        self.wkv = make_linear(config.hidden_size, config.head_dim, bias=False, quant_dtype=quant_dtype)
        self.veomni_wkv = VeomniOp("linear", "standard", linear_impl, site=f"layers.{layer_idx}.kv_proj")
        self.kv_norm = DeepseekV4RMSNorm(config.head_dim, eps=config.rms_norm_eps, config=config)

        self.wo_a = make_grouped_linear(
            self.n_heads * config.head_dim // self.n_groups,
            self.n_groups * config.o_lora_rank,
            n_groups=self.n_groups,
            quant_dtype=quant_dtype,
        )
        self.veomni_wo_a = VeomniOp("linear", "grouped", linear_impl, site=f"layers.{layer_idx}.o_a_proj")

        self.wo_b = make_linear(
            self.n_groups * config.o_lora_rank, config.hidden_size, bias=False, quant_dtype=quant_dtype
        )
        self.veomni_wo_b = VeomniOp("linear", "standard", linear_impl, site=f"layers.{layer_idx}.o_b_proj")

        self.compressor = DeepseekV41Compressor(config, layer_idx) if self.is_kv_source else None
        self.indexer = DeepseekV41Indexer(config, layer_idx) if self.is_index_source else None
        self.compress_kv_dtype = getattr(config, "compress_kv_dtype", "fp4")
        kv_round_impl = _resolve_kv_round_impl(config)
        self.veomni_kv_round_fp8 = VeomniOp(
            "kv_round", "standard", kv_round_impl, site=f"layers.{layer_idx}.window_kv"
        )
        self.veomni_kv_round_compress_fp8 = VeomniOp(
            "kv_round", "standard", kv_round_impl, site=f"layers.{layer_idx}.compress_kv"
        )
        self.veomni_kv_round_fp4 = VeomniOp("kv_round", "fp4", kv_round_impl, site=f"layers.{layer_idx}.compress_kv")

    @property
    def rope_is_compressed(self) -> bool:
        return self.compress_ratio > 0

    @property
    def builds_indexer_loss(self) -> bool:
        if self.indexer is None or not getattr(self.config, "dsa_indexer_loss", False):
            return False
        return getattr(self.config, "dsa_indexer_loss_coef", 1.0) > 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        shared: SharedAttentionState,
        plans: dict[int, CompressionPlan],
        shard: SequenceShard | None = None,
    ) -> tuple[torch.Tensor, IndexerLossTerms | None]:
        batch, seq_len, _ = hidden_states.shape
        cached_rope = shared.compress_rope_cos_sin if self.rope_is_compressed else shared.main_rope_cos_sin
        if cached_rope is not None and cached_rope[0].shape[1] == position_ids.shape[1]:
            cos, sin = cached_rope
        else:
            cos, sin = rotary_emb(position_ids, compressed=self.rope_is_compressed)
        can_inplace = not torch.is_grad_enabled()

        wq_a_out = dispatch_linear(self.veomni_wq_a, self.wq_a, hidden_states)
        wkv_out = dispatch_linear(self.veomni_wkv, self.wkv, hidden_states)
        q_residual = self.q_norm(wq_a_out)
        query = dispatch_linear(self.veomni_wq_b, self.wq_b, q_residual).unflatten(-1, (self.n_heads, self.head_dim))
        query = apply_interleaved_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2), inplace=can_inplace)

        kv = self.kv_norm(wkv_out)
        kv = apply_interleaved_rope(kv, cos, sin, inplace=can_inplace)
        kv = round_window_kv(
            kv,
            training=self.training,
            op=self.veomni_kv_round_fp8,
        )

        student = None
        if shard is None:
            if shared.window_indices is not None and shared.window_indices.shape[1] == seq_len:
                topk_idxs = shared.window_indices
            else:
                topk_idxs = self._window_indices(position_ids, seq_len)
        else:
            win_start = max(0, shard.start - self.sliding_window + 1)
            kv = shard.gather_kv(kv)[:, win_start : shard.end]
            if shared.window_indices is not None and shared.window_indices.shape[1] == seq_len:
                topk_idxs = shared.window_indices
            else:
                topk_idxs = self._local_window_indices(shard)
        if self.compress_ratio:
            compress_kv, compress_idxs, student = self._compressed_kv(
                hidden_states,
                q_residual,
                position_ids,
                rotary_emb,
                shared,
                plans,
                offset=kv.shape[1],
                shard=shard,
            )
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        query = query.contiguous()
        kv = kv.contiguous()
        topk_idxs = topk_idxs.contiguous()
        sink = self.attn_sink.float().contiguous()

        indexer_loss = None
        if student is None:
            output = sparse_attn_tilelang(query, kv, sink, topk_idxs, self.softmax_scale)
        else:
            output, lse = sparse_attn_tilelang(query, kv, sink, topk_idxs, self.softmax_scale, return_lse=True)
            valid_rows = None if shard is None else shard.valid_rows(query.device)
            indexer_loss = self._indexer_loss(query, kv, topk_idxs, lse, student, valid_rows)

        output = apply_interleaved_rope(
            output, cos.unsqueeze(-2), sin.unsqueeze(-2), inplace=can_inplace, conjugate=True
        )
        grouped = dispatch_linear(self.veomni_wo_a, self.wo_a, output.reshape(batch, seq_len, self.n_groups, -1))
        return dispatch_linear(self.veomni_wo_b, self.wo_b, grouped.flatten(2)), indexer_loss

    def _indexer_loss(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
        lse: torch.Tensor,
        student: torch.Tensor,
        valid_rows: torch.Tensor | None = None,
    ) -> IndexerLossTerms:
        width = student.shape[-1]
        if width == 0:
            zero = query.new_zeros((), dtype=torch.float32)
            return IndexerLossTerms(kl=zero, uniform=zero.clone())
        if width > topk_idxs.shape[-1]:
            raise RuntimeError(
                f"the indexer scored {width} slots but the attention selected {topk_idxs.shape[-1]} in total; "
                "the compressed slice cannot be wider than the concatenation holding it"
            )
        target = sparse_mqa_target_fwd(query, kv, topk_idxs[:, :, -width:].contiguous(), lse, self.softmax_scale)
        mass = target.sum(-1, keepdim=True)
        tiny = torch.finfo(torch.float32).tiny
        target = torch.where(mass > tiny, target / mass.clamp_min(tiny), 0.0)
        kl, uniform = indexer_kl_terms(student, target)
        if valid_rows is not None:
            kl = torch.where(valid_rows, kl, 0.0)
            uniform = torch.where(valid_rows, uniform, 0.0)
        return IndexerLossTerms(kl=kl.sum(), uniform=uniform.sum())

    def _window_indices(self, position_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        device = position_ids.device
        width = min(self.sliding_window, seq_len)
        queries = torch.arange(seq_len, device=device, dtype=torch.int32)
        offsets = torch.arange(width - 1, -1, -1, device=device, dtype=torch.int32)
        indices = (queries[None, :, None] - offsets[None, None, :]).expand(position_ids.shape[0], -1, -1)
        return indices.masked_fill(offsets[None, None, :] > position_ids.to(torch.int32)[:, :, None], -1)

    def _local_window_indices(self, shard: SequenceShard) -> torch.Tensor:
        device = shard.position_ids.device
        width = min(self.sliding_window, shard.full_len)
        win_start = max(0, shard.start - self.sliding_window + 1)
        local_pos = shard.local_rows(shard.position_ids)
        queries = torch.arange(shard.start - win_start, shard.end - win_start, device=device, dtype=torch.int32)
        offsets = torch.arange(width - 1, -1, -1, device=device, dtype=torch.int32)
        indices = (queries[None, :, None] - offsets[None, None, :]).expand(local_pos.shape[0], -1, -1)
        return indices.masked_fill(offsets[None, None, :] > local_pos.to(torch.int32)[:, :, None], -1)

    def _compressed_kv(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        shared: SharedAttentionState,
        plans: dict[int, CompressionPlan],
        offset: int,
        shard: SequenceShard | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        plan = plans[self.compress_ratio]

        if self.is_kv_source:
            latent = self.compressor(hidden_states, plan, shard)
            group_cos, group_sin = rotary_emb(plan.window_positions, compressed=True)
            if self.indexer is not None:
                shared.index_k = self.indexer.build_keys(latent, group_cos, group_sin)
            rotated = apply_interleaved_rope(latent, group_cos, group_sin)
            shared.compress_kv = round_compress_kv(
                rotated,
                self.compress_kv_dtype,
                training=self.training,
                op_fp4=self.veomni_kv_round_fp4,
                op_fp8=self.veomni_kv_round_compress_fp8,
            )
            shared.compress_ratio = self.compress_ratio

        if shared.compress_kv is None:
            raise RuntimeError(
                f"Layer {self.layer_idx} attends over compressed KV but no source layer has published any; "
                "kv_source_layer_ids must contain a layer at or before it"
            )
        if shared.compress_ratio is not None and shared.compress_ratio != self.compress_ratio:
            raise RuntimeError(
                f"Layer {self.layer_idx} expects compress_ratio={self.compress_ratio} but the active KV source "
                f"published compress_ratio={shared.compress_ratio}."
            )

        student: torch.Tensor | None = None
        if self.indexer is not None:
            if shared.index_k is None:
                raise RuntimeError(
                    f"Layer {self.layer_idx} runs an indexer but no index keys have been published; "
                    "every index source must be at or after a kv source"
                )
            if (
                shared.compress_rope_cos_sin is not None
                and shared.compress_rope_cos_sin[0].shape[1] == position_ids.shape[1]
            ):
                cos, sin = shared.compress_rope_cos_sin
            else:
                cos, sin = rotary_emb(position_ids, compressed=True)
            shared.topk_idxs, student = self.indexer(
                hidden_states,
                q_residual,
                shared.index_k,
                plan if shard is None else plan.narrow_queries(shard.start, shard.end),
                shared,
                cos,
                sin,
                return_scores=self.builds_indexer_loss,
            )
        elif shared.topk_idxs is None:
            raise RuntimeError(
                f"Layer {self.layer_idx} reuses a top-k selection but none has been published; "
                "index_source_layer_ids must contain a layer at or before it"
            )

        selection = torch.where(shared.topk_idxs >= 0, shared.topk_idxs + offset, -1)
        return shared.compress_kv, selection, student


# ---------------------------------------------------------------------------
# Aux-loss-free VL-aware MoE router & sparse MoE block
# ---------------------------------------------------------------------------

TOPK_NORM_EPS = 1e-20


class DeepseekV41TopKRouter(nn.Module):
    """Pick ``num_experts_per_tok`` experts per token; weight them by unbiased scores."""

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.gate_temp = config.gate_temp
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func_name = config.scoring_func
        self.score_fn = ACT2FN[config.scoring_func]
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self._initializer_range = config.initializer_range
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))
        self.e_score_correction_bias_vl = (
            nn.Buffer(torch.zeros(self.num_experts)) if config.enable_vision_routing_bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self._initializer_range)

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        device_type = flat.device.type if isinstance(flat.device.type, str) and flat.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            logits = F.linear(flat.float(), self.weight.float())
            if self.gate_temp != 1.0:
                logits = logits / self.gate_temp
        scores = self.score_fn(logits)

        bias = self.e_score_correction_bias.float()
        if image_mask is not None and self.e_score_correction_bias_vl is not None:
            bias = torch.where(
                image_mask.reshape(-1, 1),
                self.e_score_correction_bias_vl.float(),
                bias,
            )

        indices = torch.topk(scores + bias, self.top_k, dim=-1, sorted=False).indices
        if get_active_replay() is not None:
            indices = maybe_replay_indices(self, scores, indices)
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + TOPK_NORM_EPS)
        weights = weights * self.routed_scaling_factor
        if image_mask is not None and getattr(self, "_uniform_route_image_mask", False):
            img_flat = image_mask.reshape(-1, 1)
            ar = torch.arange(indices.shape[0], device=indices.device, dtype=indices.dtype).unsqueeze(1)
            ks = torch.arange(self.top_k, device=indices.device, dtype=indices.dtype).unsqueeze(0)
            uniform_idx = (ar * self.top_k + ks) % self.num_experts
            indices = torch.where(img_flat, uniform_idx, indices)
            weights = torch.where(img_flat, torch.zeros_like(weights), weights)
        return logits, weights, indices


class DeepseekV41SparseMoeBlock(nn.Module):
    """Routed experts plus the one shared expert every token passes through."""

    def __init__(
        self,
        config: DeepseekV41LlmConfig,
        *,
        layer_idx: int = 0,
        n_routed_experts: int | None = None,
        num_experts_per_tok: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        if n_routed_experts is not None or num_experts_per_tok is not None:
            config = copy.copy(config)
            if n_routed_experts is not None:
                config.n_routed_experts = n_routed_experts
            if num_experts_per_tok is not None:
                config.num_experts_per_tok = num_experts_per_tok
        self.gate = DeepseekV41TopKRouter(config)
        self.experts = DeepseekV41QuantizedExperts(config, quant_dtype=expert_quant_dtype(config), layer_idx=layer_idx)
        self.experts.assume_distinct_experts = True
        self.shared_experts = DeepseekV41SharedExpert(
            config, quant_dtype=linear_quant_dtype(config), layer_idx=layer_idx
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_dim = hidden_states.shape
        logits, weights, indices = self.gate(hidden_states, image_mask)
        routed = self.experts(hidden_states.reshape(-1, hidden_dim), indices, weights)
        shared_out = self.shared_experts(hidden_states)
        return routed.view(batch, seq_len, hidden_dim) + shared_out, logits.detach()


# ---------------------------------------------------------------------------
# Decoder layer, backbone & OmniModule
# ---------------------------------------------------------------------------


class DeepseekV41DecoderLayer(GradientCheckpointingLayer):
    """Attention then MoE, both wrapped in manifold-constrained Hyper-Connections."""

    def __init__(self, config: DeepseekV41LlmConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.engram = DeepseekV41EngramConnection(config, layer_idx) if layer_idx in config.engram_layer_ids else None
        self.self_attn = DeepseekV41Attention(config, layer_idx)
        n_routed, n_activated = config.moe_config_for_layer(layer_idx)
        self.mlp = DeepseekV41SparseMoeBlock(
            config, layer_idx=layer_idx, n_routed_experts=n_routed, num_experts_per_tok=n_activated
        )
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.attn_hc = DeepseekV41HyperConnection(config)
        self.ffn_hc = DeepseekV41HyperConnection(config)
        self.veomni_mhc_post = self.attn_hc.veomni_mhc_post

    def forward(
        self,
        hidden_streams: torch.Tensor,
        pre_mix: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: DeepseekV41RotaryEmbedding,
        shared: SharedAttentionState,
        plans: dict[int, CompressionPlan],
        image_mask: torch.Tensor | None = None,
        shard: SequenceShard | None = None,
        ngram_embed: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        capture_draft_input: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, IndexerLossTerms | None, torch.Tensor | None]:
        require_fp32_pre_mix(pre_mix)
        if self.engram is not None and ngram_embed is not None:
            hidden_streams = self.engram(hidden_streams, ngram_embed, token_mask)
        draft_input = hidden_streams.mean(dim=2) if capture_draft_input else None
        residual = hidden_streams
        attn_pre, attn_post, attn_comb = self.attn_hc(hidden_streams)
        collapsed = mhc_collapse(hidden_streams, pre_mix)
        attn_out, indexer_loss = self.self_attn(
            self.input_layernorm(collapsed),
            position_ids,
            rotary_emb,
            shared,
            plans,
            shard=shard,
        )
        hidden_streams = mhc_expand(attn_out, residual, attn_post, attn_comb, op=self.veomni_mhc_post)

        residual = hidden_streams
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(hidden_streams)
        collapsed = mhc_collapse(hidden_streams, attn_pre)
        mlp_out, router_logits = self.mlp(self.post_attention_layernorm(collapsed), image_mask)
        hidden_streams = mhc_expand(mlp_out, residual, ffn_post, ffn_comb, op=self.veomni_mhc_post)

        return hidden_streams, ffn_pre, router_logits, indexer_loss, draft_input


class DeepseekV41Backbone(nn.Module):
    """The 40-layer decoder stack, separated from the OmniModule wrapper around it."""

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [DeepseekV41DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)

    @property
    def has_engram(self) -> bool:
        return any(layer.engram is not None for layer in self.layers)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        ngram_embeds: list[torch.Tensor] | None = None,
        image_mask: torch.Tensor | None = None,
        return_draft_hiddens: bool = False,
        shard: SequenceShard | None = None,
    ) -> dict[str, Any]:
        if ngram_embeds is None and self.has_engram:
            raise ValueError(
                f"this backbone has Engram connections at layers {list(self.config.engram_layer_ids)} but no "
                "ngram_embeds were passed. The deepseek_v41_ngram node must run before this one in the graph; "
                "its absence would train the connections' wkv on nothing while the loss still fell."
            )
        if ngram_embeds is not None and len(ngram_embeds) != len(self.config.engram_layer_ids):
            raise ValueError(
                f"ngram_embeds has {len(ngram_embeds)} entries but engram_layer_ids names "
                f"{len(self.config.engram_layer_ids)} layers. They are positional -- entry i belongs to "
                f"layer {list(self.config.engram_layer_ids)[:1]}... -- so a length mismatch silently "
                "pairs a layer with another layer's table."
            )

        token_mask = None if image_mask is None else ~image_mask

        row_position_ids = position_ids if shard is None else shard.position_ids
        plans = build_compression_plans(
            self.config, row_position_ids, sequence_slices_from_positions(row_position_ids)
        )
        shared = SharedAttentionState()
        if len(self.layers) > 0:
            first_attn = self.layers[0].self_attn
            if shard is None:
                shared.window_indices = first_attn._window_indices(position_ids, position_ids.shape[1])
            else:
                shared.window_indices = first_attn._local_window_indices(shard)
            shared.main_rope_cos_sin = self.rotary_emb(position_ids, compressed=False)
            shared.compress_rope_cos_sin = self.rotary_emb(position_ids, compressed=True)

        hidden_streams = expand_to_streams(inputs_embeds, self.config.hc_mult)
        pre_mix = identity_pre_mix(hidden_streams, self.config.hc_mult)

        router_logits: list[torch.Tensor] = []
        indexer_terms: list[IndexerLossTerms] = []
        draft_targets = set(self.config.dspark_target_layer_ids)
        draft_inputs: dict[int, torch.Tensor] = {}

        for layer_idx, layer in enumerate(self.layers):
            layer_ngram = None if layer.engram is None else ngram_embeds[layer.engram.layer_hash_index]
            capture_draft = return_draft_hiddens and layer_idx in draft_targets

            shared = dataclasses.replace(shared)
            hidden_streams, pre_mix, layer_logits, indexer_loss, draft_input = layer(
                hidden_streams,
                pre_mix,
                position_ids,
                self.rotary_emb,
                shared,
                plans,
                image_mask,
                shard=shard,
                ngram_embed=layer_ngram,
                token_mask=token_mask if layer_ngram is not None else None,
                capture_draft_input=capture_draft,
            )
            if capture_draft:
                draft_inputs[layer_idx] = draft_input
            router_logits.append(layer_logits)
            if indexer_loss is not None:
                indexer_terms.append(indexer_loss)

        if self.config.dsa_indexer_loss and not indexer_terms:
            raise RuntimeError(
                "dsa_indexer_loss is enabled but no layer built an indexer KL: index_source_layer_ids="
                f"{list(self.config.index_source_layer_ids)} selects no layer that owns an indexer, or "
                "dsa_indexer_loss_coef is non-positive. The flag would be accepted and train nothing."
            )

        hidden_states = self.norm(mhc_collapse(hidden_streams, pre_mix))

        outputs: dict[str, Any] = {
            "hidden_states": hidden_states,
            "router_logits": router_logits,
        }
        if indexer_terms:
            kl_total = torch.stack([term.kl for term in indexer_terms]).sum()
            uniform_total = torch.stack([term.uniform for term in indexer_terms]).sum()
            query_tokens = hidden_states.shape[0] * hidden_states.shape[1]
            if shard is not None:
                kl_total = kl_total * shard.size
                uniform_total = uniform_total * shard.size
                query_tokens = hidden_states.shape[0] * shard.valid_len
            outputs["indexer_kl_total"] = kl_total
            outputs["indexer_uniform_total"] = uniform_total
            outputs["indexer_kl_layers"] = len(indexer_terms)
            outputs["indexer_query_tokens"] = query_tokens
        if return_draft_hiddens:
            outputs["draft_hiddens"] = [draft_inputs[idx] for idx in self.config.dspark_target_layer_ids]
        return outputs


class DeepseekV41Llm(PretrainedOmniModule):
    """``deepseek_v41_llm`` -- the CED backbone as an OmniModule."""

    config_class = DeepseekV41LlmConfig
    supports_sp_sharded_handoff = True
    base_model_prefix = "deepseek_v41_llm"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["DeepseekV41DecoderLayer"]
    supports_gradient_checkpointing = True

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        return attn_implementation

    @staticmethod
    def _create_checkpoint_tensor_converter(model: DeepseekV41Llm):
        from ..checkpoint_tensor_converter import create_deepseek_v41_llm_checkpoint_tensor_converter

        return create_deepseek_v41_llm_checkpoint_tensor_converter(model)

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = DeepseekV41Backbone(config)
        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs) -> None:
        checkpoint_kwargs = dict(gradient_checkpointing_kwargs or {})
        if checkpoint_kwargs.get("use_reentrant"):
            raise ValueError(
                "deepseek_v41_llm does not support reentrant gradient checkpointing because layers return "
                "IndexerLossTerms and publish shared compressed KV state across layers. "
                "Set model.accelerator.gradient_checkpointing.enable_reentrant=False."
            )
        checkpoint_kwargs["use_reentrant"] = False
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs, **kwargs)

    def _init_weights(self, module: nn.Module) -> None:
        if getattr(module, "is_quantized", False):
            return
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, DeepseekV41HyperConnection):
            nn.init.normal_(module.fn, mean=0.0, std=std)
            nn.init.zeros_(module.base)
            nn.init.ones_(module.scale)
        elif isinstance(module, DeepseekV41TopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Experts) and not getattr(module, "experts_are_quantized", False):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(module.down_proj, mean=0.0, std=std)

    def get_input_embeddings(self) -> nn.Module:
        raise NotImplementedError(
            "deepseek_v41_llm holds no token embedding: it lives in deepseek_v41_text_encoder, which also "
            "serves the DSpark draft stages. This module takes inputs_embeds."
        )

    def set_input_embeddings(self, value: nn.Module) -> None:
        raise NotImplementedError(
            "deepseek_v41_llm holds no token embedding; set it on deepseek_v41_text_encoder instead."
        )

    def freeze_model(self) -> None:
        trainable = self._trainable_backbone_layer_ids()
        if trainable is None:
            return
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx not in trainable:
                layer.requires_grad_(False)
        last = self.config.num_hidden_layers - 1
        if last not in trainable:
            self.model.norm.requires_grad_(False)

    def _trainable_backbone_layer_ids(self) -> set | None:
        if self.config.trainable_layer_ids is not None:
            return set(self.config.trainable_layer_ids)
        n = self.config.n_trainable_layers
        if n is None:
            return None
        return set(range(n))

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        ngram_embeds: list[torch.Tensor] | None = None,
        image_mask: torch.Tensor | None = None,
        return_draft_hiddens: bool | None = None,
        sequence_shard: SequenceShard | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if inputs_embeds is None:
            raise ValueError("deepseek_v41_llm requires inputs_embeds; it holds no token embedding.")
        if position_ids is None:
            raise ValueError(
                "deepseek_v41_llm requires position_ids: they carry the packed sample boundaries, and "
                "assuming a single arange here would compress across them."
            )
        if return_draft_hiddens is None:
            return_draft_hiddens = bool(
                self.config.emit_draft_hiddens
                and self.config.dspark_target_layer_ids
                and self.config.dspark_block_size > 0
            )
        return self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            ngram_embeds=ngram_embeds,
            image_mask=image_mask,
            return_draft_hiddens=return_draft_hiddens,
            shard=sequence_shard,
        )


__all__ = [
    "COMPRESS_KV_FP4_GROUP",
    "INDEX_FP4_GROUP",
    "POST_MULTIPLIER",
    "QUANT_BLOCK_SIZE",
    "SCALE_DTYPE",
    "SCALE_FMT",
    "TOPK_NORM_EPS",
    "WINDOW_KV_FP8_BLOCK",
    "CompressionPlan",
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "DeepseekV41Attention",
    "DeepseekV41Backbone",
    "DeepseekV41Compressor",
    "DeepseekV41DecoderLayer",
    "DeepseekV41EngramConnection",
    "DeepseekV41GroupedLinear",
    "DeepseekV41HyperConnection",
    "DeepseekV41Indexer",
    "DeepseekV41Linear",
    "DeepseekV41Llm",
    "DeepseekV41QuantizedExperts",
    "DeepseekV41RotaryEmbedding",
    "DeepseekV41SharedExpert",
    "DeepseekV41SparseMoeBlock",
    "DeepseekV41TopKRouter",
    "DeepseekV4Experts",
    "DeepseekV4GroupedLinear",
    "DeepseekV4MLP",
    "DeepseekV4RMSNorm",
    "DeepseekV4UnweightedRMSNorm",
    "IndexerLossTerms",
    "QuantizedWeightMixin",
    "SequenceShard",
    "SharedAttentionState",
    "apply_interleaved_rope",
    "build_compression_plans",
    "dense_sparse_attn",
    "dispatch_linear",
    "engram_layer_order",
    "expand_to_streams",
    "expert_quant_dtype",
    "identity_pre_mix",
    "indexer_kl_terms",
    "is_quantized",
    "linear_quant_dtype",
    "main_window_indices",
    "make_grouped_linear",
    "make_linear",
    "mhc_collapse",
    "mhc_expand",
    "require_fp32_pre_mix",
    "resolve_module_op_impl",
    "rotate_half_interleaved",
    "round_compress_kv",
    "round_index_act",
    "round_window_kv",
    "select_candidate_blocks",
    "sequence_slices_from_positions",
    "trailing_partial_group",
]
