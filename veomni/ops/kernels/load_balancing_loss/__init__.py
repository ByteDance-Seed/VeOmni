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

from typing import Optional, Union

import torch

from ....utils import logging
from ...config.registry import BackendSpec, OpScope, OpSpec, register_op
from .eager import load_balancing_loss_pytorch


logger = logging.get_logger(__name__)

# Default to the pure-PyTorch implementation so imports of this module are
# safe on hosts without Triton; ``apply_ops_config`` rebinds this via the
# registry based on ``load_balancing_loss_implementation``.
_load_balancing_loss = load_balancing_loss_pytorch


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    """Compute the load balancing auxiliary loss for Mixture-of-Experts models.

    Drop-in replacement for ``transformers.models.qwen3_moe.modeling_qwen3_moe.load_balancing_loss_func``
    that dispatches to a fused Triton kernel on CUDA or a pure-PyTorch fallback otherwise.

    Computes the auxiliary load balancing loss from the Switch Transformer paper
    (Fedus et al., 2021; https://arxiv.org/abs/2101.03961), equations (4)-(6)::

        loss = num_experts * sum_e(f_e * P_e)

    where ``f_e`` is the fraction of tokens routed to expert *e* and ``P_e`` is
    the average router probability assigned to expert *e* across all tokens.

    Args:
        gate_logits: Tuple of per-layer gate logits, each ``[tokens, num_experts]``.
        num_experts: Total number of experts.
        top_k: Number of experts selected per token.
        attention_mask: Optional ``[batch_size, seq_len]`` padding mask.
            Named ``attention_mask`` (rather than ``loss_mask``) for
            compatibility with the HuggingFace API.

    Returns:
        Scalar auxiliary loss tensor, or ``0`` when *gate_logits* is ``None`` / not a tuple.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    return _load_balancing_loss(gate_logits, num_experts, top_k, attention_mask)


register_op(
    OpSpec(
        name="load_balancing_loss",
        config_field="load_balancing_loss_implementation",
        label="LoadBalancingLoss",
        scope=OpScope.GLOBAL,
        default="triton",
        global_slot="veomni.ops.kernels.load_balancing_loss:_load_balancing_loss",
        backends={
            "eager": BackendSpec(entry="veomni.ops.kernels.load_balancing_loss.eager:load_balancing_loss_pytorch"),
            # ``apply_global_ops`` resolves GLOBAL ops eagerly for every model
            # — including dense models that never call this loss. Declare the
            # triton dependency on the BackendSpec so the availability check
            # fires at ``__post_init__`` time with an actionable error rather
            # than crashing later when ``apply_global_ops`` imports the
            # ``triton`` kernel module on a host that doesn't have triton /
            # triton-ascend installed.
            "triton": BackendSpec(
                entry="veomni.ops.kernels.load_balancing_loss.triton:load_balancing_loss_triton",
                requires=("triton",),
            ),
        },
    )
)


# ── OpSlot kernel registration ───────────────────────────────────────────────

from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


def _triton_load_balancing_loss_factory():
    from .triton import load_balancing_loss_triton

    return load_balancing_loss_triton


KERNEL_REGISTRY.register(
    KernelSpec(
        name="triton",
        op_name="load_balancing_loss",
        variant="standard",
        factory=_triton_load_balancing_loss_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="Fused Triton load-balancing loss for MoE",
    )
)
