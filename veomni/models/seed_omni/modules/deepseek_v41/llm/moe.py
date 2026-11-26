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

"""V4.1's MoE: V4's experts behind a router that routes image tokens separately.

The expert side is imported wholesale. ``DeepseekV4Experts`` already carries the
stacked ``[E, 2I, H]`` / ``[E, H, I]`` layout, the VeomniOp-guarded fused-MoE path,
the fp4 weight QAT keyed on ``expert_dtype`` and the ``swiglu_limit`` clamp --
all of which V4.1 wants unchanged -- and ``DeepseekV4MLP`` is the shared expert
down to the same clamp. Both read the config by attribute name, which is what
``intermediate_size`` / ``num_local_experts`` / ``mlp_bias`` on
:class:`DeepseekV41LlmConfig` are for.

The shared expert is taken through :class:`DeepseekV41SharedExpert`, which is
that same class with its three projections able to hold the released fp8
weights. It subclasses rather than replaces because the arithmetic around them
-- the fp32 clamp, the fused SwiGLU -- is what V4.1 wants and is generated code.

The router is rewritten. Subclassing ``DeepseekV4TopKRouter`` would mean
overriding ``forward`` in full anyway -- three of its five lines differ -- while
inheriting a ``__init__`` that declares the wrong buffers. What it would still
buy is the one line allocating ``weight``, against a base class whose every
future edit lands in a method V4.1 does not run.

Three differences, in the order the forward meets them:

``gate_temp``
    V4.1 divides the logits before scoring. V4 has no such field, which is the
    same as a hardcoded 1.0 -- and 1.0 is what every released V4.1 config also
    resolves to, so this only matters for a checkpoint that ships another value.
    Declared so such a checkpoint cannot silently lose it.
``bias_vl``
    Tokens inside an image span select a second correction bias. This is the
    ``noaux_tc_for_vl`` rule from training: the two modalities have very
    different expert affinities, and one balancing controller for both would
    keep pushing text off the experts vision saturates.
``norm_topk_prob``
    V4 renormalises unconditionally. V4.1 does it only when
    ``norm_topk_prob and num_experts_per_tok > 1``, and divides by
    ``sum + 1e-20`` -- the literal from training rather than ``rms_norm_eps``,
    which on V4.1 happens to also be 1e-20 and would make the two look
    interchangeable.

Both biases are buffers, not parameters, matching V3 / V4 / GLM-MoE-DSA
throughout this repo. The released checkpoint stores them as parameters, but
under the aux-loss-free scheme they are load-balancing controllers updated by a
sign rule on expert load, not by the objective's gradient. Leaving them as
parameters would hand them to the optimizer, which would apply momentum and
weight decay to a quantity whose whole job is to track load.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils.generic import maybe_autocast

from ......utils.device import IS_CUDA_AVAILABLE, get_torch_device
from ......utils.import_utils import is_package_available
from ......utils.moe_router_replay import get_active_replay, maybe_replay_indices
from .configuration import DeepseekV41LlmConfig
from .quantized import (
    DeepseekV41QuantizedExperts,
    DeepseekV41SharedExpert,
    expert_quant_dtype,
    linear_quant_dtype,
)


_HAS_TRITON = is_package_available("triton")
if _HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_moe_topk_gate_kernel(
        Logits_ptr,
        Bias_ptr,
        BiasVL_ptr,
        ImageMask_ptr,
        Weights_ptr,
        Indices_ptr,
        E: tl.constexpr,
        BLOCK_E: tl.constexpr,
        TOP_K: tl.constexpr,
        NEXT_POW2_K: tl.constexpr,
        GATE_TEMP: tl.constexpr,
        HAS_TEMP: tl.constexpr,
        HAS_IMAGE_MASK: tl.constexpr,
        NORM_TOPK: tl.constexpr,
        ROUTED_SCALE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs_e = tl.arange(0, BLOCK_E)
        mask_e = offs_e < E
        l = tl.load(Logits_ptr + row * E + offs_e, mask=mask_e, other=0.0).to(tl.float32)
        if HAS_TEMP:
            l = l / GATE_TEMP
            tl.store(Logits_ptr + row * E + offs_e, l, mask=mask_e)
        softplus = tl.where(l > 20.0, l, tl.log(1.0 + tl.exp(l)))
        scores = tl.where(mask_e, tl.sqrt(softplus), 0.0)

        if HAS_IMAGE_MASK:
            is_img = tl.load(ImageMask_ptr + row).to(tl.int1)
            b = tl.where(
                is_img,
                tl.load(BiasVL_ptr + offs_e, mask=mask_e, other=0.0).to(tl.float32),
                tl.load(Bias_ptr + offs_e, mask=mask_e, other=0.0).to(tl.float32),
            )
        else:
            b = tl.load(Bias_ptr + offs_e, mask=mask_e, other=0.0).to(tl.float32)
        biased = tl.where(mask_e, scores + b, -float("inf"))

        w_sum = tl.zeros((), dtype=tl.float32)
        for k in tl.static_range(TOP_K):
            best_idx = tl.argmax(biased, axis=0)
            best_unbiased = tl.sum(tl.where(offs_e == best_idx, scores, 0.0), axis=0)
            tl.store(Indices_ptr + row * TOP_K + k, best_idx.to(tl.int64))
            tl.store(Weights_ptr + row * TOP_K + k, best_unbiased)
            w_sum += best_unbiased
            biased = tl.where(offs_e == best_idx, -float("inf"), biased)

        offs_k = tl.arange(0, NEXT_POW2_K)
        mask_k = offs_k < TOP_K
        w_k = tl.load(Weights_ptr + row * TOP_K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        if NORM_TOPK and TOP_K > 1:
            w_k = w_k * (ROUTED_SCALE / (w_sum + 1e-20))
        elif ROUTED_SCALE != 1.0:
            w_k = w_k * ROUTED_SCALE
        tl.store(Weights_ptr + row * TOP_K + offs_k, w_k, mask=mask_k)


_SHARED_EXPERT_STREAMS: dict[int, Any] = {}


def _get_shared_expert_stream(device: torch.device) -> Any:
    torch_device = get_torch_device()
    idx = device.index if device.index is not None else torch_device.current_device()
    stream = _SHARED_EXPERT_STREAMS.get(idx)
    if stream is None:
        with torch_device.device(idx):
            stream = torch_device.Stream(device=device)
        _SHARED_EXPERT_STREAMS[idx] = stream
    return stream


#: The renormalisation floor, from training. Not ``rms_norm_eps``: that it holds
#: the same value on V4.1-Flash is a coincidence of that checkpoint.
TOPK_NORM_EPS = 1e-20


class DeepseekV41TopKRouter(nn.Module):
    """Pick ``num_experts_per_tok`` experts per token; weight them by unbiased scores.

    The correction bias steers *selection* only -- it is added before the top-k
    and dropped before the gather. Scaling by the biased score instead would let
    the balancing controller change how much an expert contributes, and the
    controller has no information about that.
    """

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

    def _get_fp32_weight(self) -> torch.Tensor:
        if self.weight.dtype == torch.float32:
            return self.weight
        key = (self.weight.data_ptr(), self.weight._version, self.weight.dtype)
        cached = self.__dict__.get("_fp32_weight_cache")
        if cached is None or cached[0] != key:
            cached = (key, self.weight.float())
            self.__dict__["_fp32_weight_cache"] = cached
        return cached[1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(logits, weights, indices)``, all flattened over batch and sequence.

        ``image_mask`` is ``[B, S]`` bool, true inside an image span. ``None``
        means a text-only batch, which is not the same as an all-false mask only
        in that it skips the select.
        """
        flat = hidden_states.reshape(-1, self.hidden_dim)
        if (
            IS_CUDA_AVAILABLE
            and _HAS_TRITON
            and flat.is_cuda
            and flat.shape[0] > 0
            and self.scoring_func_name == "sqrtsoftplus"
            and self.gate_temp == 1.0
            and 0 < self.num_experts <= 512
            and 0 < self.top_k <= 16
            and get_active_replay() is None
        ):
            device_type = (
                flat.device.type if isinstance(flat.device.type, str) and flat.device.type != "mps" else "cpu"
            )
            with maybe_autocast(device_type=device_type, enabled=False):
                weight_fp32 = self.weight.float() if torch.is_grad_enabled() else self._get_fp32_weight()
                logits = F.linear(flat.float(), weight_fp32).contiguous()
            _, b_fp32, b_vl_fp32 = self._get_bf16_weight_and_biases()
            has_img = image_mask is not None and b_vl_fp32 is not None
            img_mask_1d = image_mask.reshape(-1).contiguous() if has_img else b_fp32
            m = flat.shape[0]
            weights = torch.empty((m, self.top_k), dtype=torch.float32, device=flat.device)
            indices = torch.empty((m, self.top_k), dtype=torch.int64, device=flat.device)
            block_e = max(16, triton.next_power_of_2(self.num_experts))
            next_pow2_k = max(1, triton.next_power_of_2(self.top_k))
            with torch.no_grad(), get_torch_device().device(flat.device):
                _triton_moe_topk_gate_kernel[(m,)](
                    logits.detach(),
                    b_fp32,
                    b_vl_fp32 if has_img else b_fp32,
                    img_mask_1d,
                    weights,
                    indices,
                    E=self.num_experts,
                    BLOCK_E=block_e,
                    TOP_K=self.top_k,
                    NEXT_POW2_K=next_pow2_k,
                    GATE_TEMP=float(self.gate_temp),
                    HAS_TEMP=self.gate_temp != 1.0,
                    HAS_IMAGE_MASK=has_img,
                    NORM_TOPK=bool(self.norm_topk_prob),
                    ROUTED_SCALE=float(self.routed_scaling_factor),
                    num_warps=4,
                )
            if torch.is_grad_enabled() and logits.requires_grad:
                selected_scores = self.score_fn(logits.gather(1, indices))
                if self.norm_topk_prob and self.top_k > 1:
                    selected_scores = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + TOPK_NORM_EPS)
                weights = selected_scores * self.routed_scaling_factor
            return logits.detach(), weights, indices

        device_type = flat.device.type if isinstance(flat.device.type, str) and flat.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            weight_fp32 = self.weight.float() if torch.is_grad_enabled() else self._get_fp32_weight()
            logits = F.linear(flat.float(), weight_fp32)
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
        """``[B, S, D]`` -> the block output and the router logits.

        The logits leave as a return value so a balance metric can be computed
        without the block holding onto them; V4.1 trains aux-loss-free, so
        nothing in the objective consumes them.
        """
        batch, seq_len, hidden_dim = hidden_states.shape
        logits, weights, indices = self.gate(hidden_states, image_mask)
        if getattr(self, "_enable_recompute_cache", False):
            cached_routing = getattr(self, "_recompute_routing_cache", None)
            if cached_routing is not None:
                indices._ep_routing_cache = cached_routing
                self._recompute_routing_cache = None
            else:
                indices._ep_routing_cache_holder = self
        if IS_CUDA_AVAILABLE and hidden_states.is_cuda:
            side_stream = _get_shared_expert_stream(hidden_states.device)
            torch_device = get_torch_device()
            main_stream = torch_device.current_stream(hidden_states.device)
            fork_evt = torch_device.Event()
            fork_evt.record(main_stream)

            def _run_shared_on_side_stream() -> torch.Tensor:
                side_stream.wait_event(fork_evt)
                with get_torch_device().stream(side_stream):
                    hidden_states.record_stream(side_stream)
                    h_in = hidden_states.view_as(hidden_states)
                    if torch.is_grad_enabled() and h_in.requires_grad:
                        h_in.register_hook(
                            lambda _g, _ms=main_stream, _ss=side_stream: (_ms.wait_stream(_ss), None)[1]
                        )
                    return self.shared_experts(h_in)

            indices._shared_experts_fn = _run_shared_on_side_stream
            routed = self.experts(hidden_states.reshape(-1, hidden_dim), indices, weights)
            shared_out = getattr(indices, "_shared_experts_out", None)
            if shared_out is not None:
                main_stream.wait_stream(side_stream)
                shared_out.record_stream(main_stream)
            else:
                shared_out = self.shared_experts(hidden_states)
        else:
            indices._shared_experts_fn = lambda: self.shared_experts(hidden_states)
            routed = self.experts(hidden_states.reshape(-1, hidden_dim), indices, weights)
            shared_out = getattr(indices, "_shared_experts_out", None)
            if shared_out is None:
                shared_out = self.shared_experts(hidden_states)
        return routed.view(batch, seq_len, hidden_dim) + shared_out, logits.detach()


__all__ = [
    "TOPK_NORM_EPS",
    "DeepseekV41SparseMoeBlock",
    "DeepseekV41TopKRouter",
]
