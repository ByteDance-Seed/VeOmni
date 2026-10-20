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

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils.generic import maybe_autocast

from ......utils.moe_router_replay import get_active_replay, maybe_replay_indices
from .configuration import DeepseekV41LlmConfig
from .quantized import (
    DeepseekV41QuantizedExperts,
    DeepseekV41SharedExpert,
    expert_quant_dtype,
    linear_quant_dtype,
)


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
        """``(logits, weights, indices)``, all flattened over batch and sequence.

        ``image_mask`` is ``[B, S]`` bool, true inside an image span. ``None``
        means a text-only batch, which is not the same as an all-false mask only
        in that it skips the select.
        """
        flat = hidden_states.reshape(-1, self.hidden_dim)
        device_type = flat.device.type if isinstance(flat.device.type, str) and flat.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            logits = F.linear(flat.float(), self.weight.float()) / self.gate_temp
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
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV41SparseMoeBlock(nn.Module):
    """Routed experts plus the one shared expert every token passes through."""

    def __init__(
        self,
        config: DeepseekV41LlmConfig,
        *,
        n_routed_experts: int | None = None,
        num_experts_per_tok: int | None = None,
    ) -> None:
        super().__init__()
        if n_routed_experts is not None or num_experts_per_tok is not None:
            config = copy.copy(config)
            if n_routed_experts is not None:
                config.n_routed_experts = n_routed_experts
            if num_experts_per_tok is not None:
                config.num_experts_per_tok = num_experts_per_tok
        self.gate = DeepseekV41TopKRouter(config)
        self.experts = DeepseekV41QuantizedExperts(config, quant_dtype=expert_quant_dtype(config))
        self.shared_experts = DeepseekV41SharedExpert(config, quant_dtype=linear_quant_dtype(config))

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
        routed = self.experts(hidden_states.reshape(-1, hidden_dim), indices, weights)
        return routed.view(batch, seq_len, hidden_dim) + self.shared_experts(hidden_states), logits


__all__ = [
    "TOPK_NORM_EPS",
    "DeepseekV41SparseMoeBlock",
    "DeepseekV41TopKRouter",
]
