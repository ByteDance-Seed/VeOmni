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

"""Compatibility for FA4 4.0.0b16 SM90 sparse auxiliary mask loads."""

import inspect
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Optional


_retired_compile_caches = ()


def _make_sm90_mask_forwarder(original, cute, cutlass):
    """Guard auxiliary callbacks, retaining full K bounds for ragged sparse tiles."""

    @cute.jit
    def query_bounds_mask(
        batch_idx, head_idx, query_idx, key_idx, seqlen_info, aux_tensors, mask_mod: cutlass.Constexpr[Callable]
    ):
        """Give callbacks valid logical Q indices and then exclude padded Q rows."""
        # Native has already unpacked PackGQA into logical query/head indices.
        q_valid = query_idx[0] < seqlen_info.seqlen_q
        safe_q = cute.make_rmem_tensor(1, cutlass.Int32)
        safe_q[0] = query_idx[0] if q_valid else cutlass.Int32(0)
        result = mask_mod(batch_idx, head_idx, safe_q.load(), key_idx, seqlen_info, aux_tensors)
        valid_result = cute.make_rmem_tensor(1, cutlass.Boolean)
        valid_result[0] = result[0] and q_valid
        return valid_result.load()

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
    ) -> None:
        """Use Q-only guards when K is complete, otherwise preserve native Q/K bounds."""
        # Only later auxiliary partial blocks need an additional guard. The
        # first KV boundary and native backward already pass mask_seqlen=True.
        if cutlass.const_expr(
            not mask_seqlen and not mask_causal and not mask_local and mask_mod is not None and aux_tensors is not None
        ):
            # Sparse lists need not put the ragged K tile first. Preserve
            # full K protection on every Q tile before taking either fast path.
            if self.seqlen_q <= 0 or n_block < 0 or (n_block + 1) * self.tile_n > self.seqlen_k:
                original(
                    self,
                    acc_S,
                    batch_idx,
                    head_idx,
                    m_block,
                    n_block,
                    thr_mma,
                    True,
                    mask_causal,
                    mask_local,
                    mask_mod,
                    aux_tensors,
                    fastdiv_mods,
                )
            # PackGQA expands tokens into physical MMA rows; this branch is
            # CTA-uniform. Only padded Q rows need the Q-only callback.
            elif (m_block + 1) * self.tile_m > self.seqlen_q * self.qhead_per_kvhead_packgqa:
                original(
                    self,
                    acc_S,
                    batch_idx,
                    head_idx,
                    m_block,
                    n_block,
                    thr_mma,
                    False,
                    mask_causal,
                    mask_local,
                    partial(query_bounds_mask, mask_mod=mask_mod),
                    aux_tensors,
                    fastdiv_mods,
                )
            else:
                original(
                    self,
                    acc_S,
                    batch_idx,
                    head_idx,
                    m_block,
                    n_block,
                    thr_mma,
                    False,
                    mask_causal,
                    mask_local,
                    mask_mod,
                    aux_tensors,
                    fastdiv_mods,
                )
        else:
            original(
                self,
                acc_S,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                thr_mma,
                mask_seqlen,
                mask_causal,
                mask_local,
                mask_mod,
                aux_tensors,
                fastdiv_mods,
            )

    apply_mask._veomni_sm90_aux_bounds = True
    return apply_mask


def install_fa4_sm90_aux_bounds_fix():
    """Install the pinned callback fix and isolate its forward/backward caches.

    Return True when installed (including an earlier installation), or False
    when FA4 is absent or has another version. A mismatched pinned signature
    raises before patching. Optional imports and separate SM100 methods stay
    intact; installation must happen before tracing attention calls.
    """
    global _retired_compile_caches
    try:
        if version("flash-attn-4") != "4.0.0b16":
            return False
    except PackageNotFoundError:
        return False

    try:
        import cutlass
        import cutlass.cute as cute
        from flash_attn.cute import interface
        from flash_attn.cute.cache_utils import get_jit_cache
        from flash_attn.cute.mask import AttentionMask
    except ImportError:
        return False

    original = AttentionMask.apply_mask
    if getattr(original, "_veomni_sm90_aux_bounds", False):
        return True
    expected = (
        "self",
        "acc_S",
        "batch_idx",
        "head_idx",
        "m_block",
        "n_block",
        "thr_mma",
        "mask_seqlen",
        "mask_causal",
        "mask_local",
        "mask_mod",
        "aux_tensors",
        "fastdiv_mods",
    )
    if tuple(inspect.signature(original).parameters) != expected:
        raise RuntimeError("FA4 4.0.0b16 AttentionMask.apply_mask signature differs from the supported API.")
    # FA4's persistent fingerprint hashes its own package sources, excluding
    # this shim. A separate namespace prevents reuse of uncorrected kernels,
    # including kernels compiled before VeOmni was imported in this process.
    # Bump this namespace revision whenever the forwarder's semantics change.
    # Prepare both caches before publishing the method/idempotence marker so
    # cache initialization failure leaves the existing installation intact.
    forwarder = _make_sm90_mask_forwarder(original, cute, cutlass)
    fwd_cache = get_jit_cache("veomni_sm90_aux_bounds_q_only_k_bounds_materialized_fwd")
    bwd_cache = get_jit_cache("veomni_sm90_aux_bounds_q_only_k_bounds_materialized_bwd")
    _retired_compile_caches = (interface._flash_attn_fwd.compile_cache, interface._flash_attn_bwd.compile_cache)
    AttentionMask.apply_mask = forwarder
    interface._flash_attn_fwd.compile_cache = fwd_cache
    interface._flash_attn_bwd.compile_cache = bwd_cache
    return True
