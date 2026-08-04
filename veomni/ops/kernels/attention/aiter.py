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

"""aiter (AMD AI Tensor Engine for ROCm) attention kernels.

Unlike the flex backend, aiter does not need its own SP-aware forward: it reuses
the FlashAttention adapter in ``flash.py`` and only swaps the underlying kernels.
This module therefore exposes a kernel-like object rather than an attention
forward, and ``flash._load_veomni_local_flash_kernel`` picks it up for the
``veomni_flash_attention_aiter_with_sp`` implementation name.
"""

from types import SimpleNamespace
from typing import Optional

import torch


def aiter_window_size(window_size) -> tuple:
    """Translate a flash-attn 2-tuple ``(left, right)`` window into aiter's 3-tuple
    ``(left, right, sink)`` (sink defaults to 0). ``None`` maps to the full-attention
    window ``(-1, -1, 0)``."""
    if window_size is None:
        return (-1, -1, 0)
    ws = tuple(window_size)
    if len(ws) == 2:
        return (ws[0], ws[1], 0)
    return ws


def build_aiter_flash_kernels() -> SimpleNamespace:
    """
    Build a kernel-like object backed by aiter's flash-attention kernels, adapted to
    the calling convention Transformers' ``_flash_attention_forward`` expects (i.e. it
    exposes ``flash_attn_func`` and ``flash_attn_varlen_func``).

    aiter's public API mirrors flash-attn but differs in three ways, absorbed here so
    the surrounding VeOmni/Transformers plumbing is unchanged:

    * the forward asserts ``return_lse`` whenever autograd is enabled (it needs the
      log-sum-exp for the backward pass), and the flag picks a distinct CK
      instantiation, so we request it exactly then rather than unconditionally.
      ``out`` is bit-identical either way. Without the log-sum-exp aiter returns a
      bare tensor rather than an ``(out, lse, ...)`` tuple; Transformers guards
      every call site with ``isinstance(out, tuple)``, so both shapes are handled.
    * ``window_size`` is a 3-tuple ``(left, right, sink)`` rather than a 2-tuple.
    * the varlen entry point names the softcap ``logits_soft_cap``; the dense entry
      point exposes no softcap argument at all, which this shim rejects explicitly
      rather than silently ignoring.

    The parameter names declared on the varlen shim are what Transformers'
    ``_lazy_define_process_function`` introspects to decide which optional kwargs
    (dropout, window, deterministic, softcap, max_seqlen) it forwards.
    """
    import aiter

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
        **ignored,
    ):
        if softcap:
            raise ValueError(
                "attn_implementation='aiter' does not support a logits softcap on the dense "
                "attention path (aiter.flash_attn_func has no softcap argument). Use a model "
                "without softcap, or the packed/varlen path which supports logits_soft_cap."
            )
        return aiter.flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=aiter_window_size(window_size),
            deterministic=deterministic,
            return_lse=torch.is_grad_enabled(),
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
        **ignored,
    ):
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
            return_lse=torch.is_grad_enabled(),
        )

    return SimpleNamespace(
        flash_attn_func=flash_attn_func,
        flash_attn_varlen_func=flash_attn_varlen_func,
    )
