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

from ...utils import logging
from ...utils.import_utils import is_torch_npu_available


logger = logging.get_logger(__name__)

_load_balancing_loss = None


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

    if _load_balancing_loss is None:
        raise RuntimeError(
            "Load balancing loss backend has not been initialized. "
            "Call apply_veomni_load_balancing_loss_patch() first."
        )

    return _load_balancing_loss(gate_logits, num_experts, top_k, attention_mask)


def apply_veomni_load_balancing_loss_patch(load_balancing_loss_implementation: str = "triton"):
    """Select and bind the load balancing loss implementation.

    Args:
        load_balancing_loss_implementation: ``"triton"`` or ``"eager"``.
            Should already be resolved from ``"auto"`` by
            ``OpsImplementationConfig._resolve_auto_implementations``.
    """
    global _load_balancing_loss

    if load_balancing_loss_implementation == "eager" or is_torch_npu_available():
        from .torch_native import load_balancing_loss_pytorch

        _load_balancing_loss = load_balancing_loss_pytorch
        logger.info_rank0("Load balancing loss: using PyTorch eager backend.")
    else:
        from .triton_kernel import load_balancing_loss_triton

        _load_balancing_loss = load_balancing_loss_triton
        logger.info_rank0("Load balancing loss: using fused Triton backend.")
