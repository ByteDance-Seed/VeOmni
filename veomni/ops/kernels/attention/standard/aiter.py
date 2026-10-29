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

"""AITER (AMD AI Tensor Engine for ROCm) FlashAttention kernel adapter."""

from types import SimpleNamespace
from typing import Optional

import torch


def _needs_backward(q, k, v) -> bool:
    """Return whether AITER must retain forward state for Q/K/V gradients."""
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v))


def aiter_window_size(window_size) -> tuple:
    """Translate a FlashAttention window into AITER's three-element tuple."""
    if window_size is None:
        return (-1, -1, 0)
    ws = tuple(window_size)
    if len(ws) == 2:
        return (ws[0], ws[1], 0)
    if len(ws) != 3:
        raise ValueError(
            f"attn_implementation='aiter' expects a 2- or 3-element window_size, got {window_size!r}. "
            "FlashAttention and AITER disagree on the tuple width, so an unexpected width is a wiring bug."
        )
    return ws


def _reject_attention_sinks(s_aux, learnable_sink) -> None:
    """Reject sink logits until their semantics are validated against eager."""
    if s_aux is not None or learnable_sink is not None:
        raise ValueError(
            "attn_implementation='aiter' does not support attention sinks yet (the model passed "
            "s_aux/learnable_sink). Sink logits would be silently ignored, changing the attention "
            "maths. Use attn_implementation='eager' for sink models such as DeepSeek-V4 and gpt_oss; "
            "'flash_attention_2' is not a substitute because flash-attn 2.8.3 has no sink argument "
            "either and drops them without raising."
        )


def _reject_softcap_under_autograd(softcap, q, k, v) -> None:
    """Reject a varlen logits softcap when AITER would compute uncapped gradients."""
    if softcap and _needs_backward(q, k, v):
        raise ValueError(
            "attn_implementation='aiter' does not support a logits softcap under autograd: "
            "AITER applies the cap in the forward but its backward ignores it, so gradients "
            "would be silently wrong. Use attn_implementation='eager' to train a model with "
            "attention logit softcapping. Inference (torch.no_grad) is unaffected."
        )


def build_aiter_flash_kernels() -> SimpleNamespace:
    """Build a kernel-like object matching Transformers' FlashAttention API.

    AITER differs from flash-attn in four details absorbed here:

    * it requires ``return_lse`` under autograd;
    * ``window_size`` has a third sink-size element;
    * the varlen softcap is named ``logits_soft_cap`` and is forward-only;
    * sink logits are not supported by this integration.

    Every Transformers-forwarded keyword is explicit, so a future API addition
    fails at this boundary instead of changing attention behavior silently.
    """
    try:
        import aiter
    except ImportError as e:
        raise ImportError(
            "VeOmni attention implementation `veomni_flash_attention_aiter` requires "
            "`aiter` (AMD ROCm AITER) to be importable."
        ) from e

    missing = [name for name in ("flash_attn_func", "flash_attn_varlen_func") if not hasattr(aiter, name)]
    if missing:
        raise RuntimeError(
            f"`aiter` is installed but does not expose {missing}. AITER disables its CK/HIP kernels "
            "when it cannot query the GPU architecture, which happens on a non-ROCm host or when the "
            "container is missing `--device /dev/kfd --device /dev/dri`. Fix the ROCm runtime, or pick "
            "another attn_implementation."
        )

    def flash_attn_func(
        q,
        k,
        v,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size=(-1, -1),
        softcap: float = 0.0,
        deterministic: bool = False,
        return_attn_probs: bool = False,
        s_aux=None,
        learnable_sink=None,
    ):
        if softcap:
            raise ValueError(
                "attn_implementation='aiter' does not support a logits softcap on the dense "
                "attention path (aiter.flash_attn_func has no softcap argument). Use a model "
                "without softcap, or the packed/varlen path which supports logits_soft_cap."
            )
        _reject_attention_sinks(s_aux, learnable_sink)
        return aiter.flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=aiter_window_size(window_size),
            deterministic=deterministic,
            return_lse=_needs_backward(q, k, v),
            return_attn_probs=return_attn_probs,
        )

    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size=(-1, -1),
        softcap: float = 0.0,
        deterministic: bool = False,
        return_attn_probs: bool = False,
        s_aux=None,
        learnable_sink=None,
    ):
        _reject_attention_sinks(s_aux, learnable_sink)
        _reject_softcap_under_autograd(softcap, q, k, v)
        return aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            logits_soft_cap=float(softcap) if softcap else 0.0,
            causal=causal,
            window_size=aiter_window_size(window_size),
            deterministic=deterministic,
            return_lse=_needs_backward(q, k, v),
            return_attn_probs=return_attn_probs,
        )

    return SimpleNamespace(
        flash_attn_func=flash_attn_func,
        flash_attn_varlen_func=flash_attn_varlen_func,
    )
