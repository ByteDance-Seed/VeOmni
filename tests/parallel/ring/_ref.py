# Copyright 2025 Bytedance Ltd. and/or its affiliates.
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
"""Backend-aware single-device flash-attention references for ring/USP tests.

The ring kernels auto-select their flash-attention backend (``FA_BACKEND`` in
``veomni.distributed.sequence_parallel.ring_attention``): classic FA2 on
Ampere/Hopper, or the FA4 CuTe backend on Blackwell/GB200. The tests must build
their single-device reference with the *same* backend, so this module exposes
thin ``ref_attn_func`` / ``ref_attn_varlen_func`` wrappers plus the matching
``ATTN_IMPL_WITH_SP`` token for the op-level e2e test.

Shape conventions match each backend's public API:

* dense  — both FA2 and FA4 take ``(b, s, h, d)``.
* varlen — both FA2 and FA4 take ``(total, h, d)``.

FA4's high-level functions return ``(out, lse_or_none)`` tuples, which we unpack
so callers always receive a plain output tensor.
"""

from veomni.distributed.sequence_parallel.ring_attention import FA_BACKEND


FA_OK = FA_BACKEND is not None

# Map the detected ring backend to the VeOmni SP attention implementation token
# used by ``flash_attention_forward``.
ATTN_IMPL_WITH_SP = {
    "fa2": "veomni_flash_attention_2_with_sp",
    "fa4": "veomni_flash_attention_4_with_sp",
}.get(FA_BACKEND)


def _import_backend_funcs():
    if FA_BACKEND == "fa4":
        from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func

        return flash_attn_func, flash_attn_varlen_func
    # FA2 (and any future backend exposing the classic top-level API)
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    return flash_attn_func, flash_attn_varlen_func


def _unwrap(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def ref_attn_func(q, k, v, softmax_scale, causal=True):
    """Single-device dense causal attention. ``q/k/v`` are ``(b, s, h, d)``."""
    flash_attn_func, _ = _import_backend_funcs()
    return _unwrap(flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal))


def ref_attn_varlen_func(q, k, v, cu_seqlens, max_seqlen, softmax_scale, causal=True):
    """Single-device packed varlen causal attention. ``q/k/v`` are ``(total, h, d)``."""
    _, flash_attn_varlen_func = _import_backend_funcs()
    return _unwrap(
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    )
