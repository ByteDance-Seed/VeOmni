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

"""Opt-in AscendC GDN row retaining forward intermediates for backward."""

from __future__ import annotations

import torch
from torch import Tensor

from .....registry import SavedState
from . import fla_npu


def _reuse_ops(q: Tensor, cu_seqlens_list: list[int] | None):
    """Restrict the decomposed path to its supported kernel contract."""
    if q.device.type != "npu":
        raise RuntimeError("fla_npu_reuse requires an Ascend 910C NPU.")
    device_index = q.device.index if q.device.index is not None else torch.npu.current_device()
    device_name = torch.npu.get_device_name(device_index)
    if not device_name.startswith(("Ascend910C", "Ascend910_93")):
        raise RuntimeError(f"fla_npu_reuse currently supports Ascend 910C only, got {device_name}.")
    if cu_seqlens_list is not None and (
        len(cu_seqlens_list) < 2
        or cu_seqlens_list[0] != 0
        or cu_seqlens_list[-1] != q.shape[2]
        or any(end <= start for start, end in zip(cu_seqlens_list, cu_seqlens_list[1:]))
    ):
        raise ValueError("fla_npu_reuse requires strictly increasing packed boundaries spanning the input.")
    ops = fla_npu._get_gdn_ops()
    names = (
        "chunk_gated_delta_rule_fwd_prepare",
        "chunk_fwd_h",
        "chunk_fwd_o",
        "chunk_bwd_dv_local",
        "chunk_gated_delta_rule_bwd_dhu",
        "chunk_gated_delta_rule_bwd_finalize",
    )
    missing = [name for name in names if not callable(getattr(ops, name, None))]
    if missing:
        raise RuntimeError(f"Installed fla_npu lacks the decomposed GDN entries needed for reuse: {missing}.")
    return ops


def _chunk_fwd_reuse(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    cu_seqlens_list,
    chunk_indices,
    chunk_indices_list,
    chunk_size,
):
    """Retain the 910C prepare/state results for the matching backward."""
    fla_npu._validate_gdn_contract(q, k, v, chunk_size, initial_state)
    ops = _reuse_ops(q, cu_seqlens_list)
    chunks = fla_npu._chunk_list(chunk_indices_list, chunk_size)
    g_h = g.transpose(1, 2).contiguous().float()
    beta_h = beta.transpose(1, 2).contiguous().float()
    _, _, _, _, _, g_cum, w, u, a = ops.chunk_gated_delta_rule_fwd_prepare(
        q,
        k,
        v,
        g_h,
        beta_h,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunks,
        output_a=True,
        use_exp2=False,
    )
    h, v_new, final_state = ops.chunk_fwd_h(
        k,
        w,
        u,
        g=g_cum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunks,
        use_exp2=False,
        state_v_first=False,
    )
    output = ops.chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g_cum,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunks,
        chunk_size=chunk_size,
        transpose_state_layout=False,
        use_exp2=False,
        output_layout="BNSD",
    )
    return g_cum.transpose(1, 2).contiguous(), output.transpose(1, 2).contiguous(), a, final_state, w, h, v_new


def _chunk_bwd_reuse(
    q,
    k,
    v,
    g,
    beta,
    a,
    scale,
    initial_state,
    grad_output,
    cu_seqlens,
    cu_seqlens_list,
    chunk_indices,
    chunk_indices_list,
    chunk_size,
    w,
    h,
    v_new,
):
    """Use the retained states without rerunning prepare or state forward."""
    ops = _reuse_ops(q, cu_seqlens_list)
    chunks = fla_npu._chunk_list(chunk_indices_list, chunk_size)
    g_h = g.transpose(1, 2).contiguous()
    beta_h = beta.transpose(1, 2).contiguous().float()
    do_h = grad_output.transpose(1, 2).contiguous()
    dv_local = ops.chunk_bwd_dv_local(
        q,
        k,
        do_h,
        g_h,
        scale,
        chunk_size,
        A=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunks,
    )
    dh, _, dv = ops.chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do_h,
        dv_local,
        scale,
        chunk_size,
        g=g_h,
        h0=initial_state,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunks,
        use_exp2=False,
        transpose_state_layout=False,
    )
    dq, dk, dv, dbeta, dg = ops.chunk_gated_delta_rule_bwd_finalize(
        q,
        k,
        v,
        v_new,
        do_h,
        dv,
        g_h,
        beta_h,
        h,
        dh,
        a,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunks,
        scale=scale,
        chunk_size=chunk_size,
        use_qk_l2_norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        use_gate_in_kernel=False,
        state_v_first=False,
        use_exp2=False,
    )
    # Finalize returns the raw-gate gradient; no additional reverse cumsum.
    return dq, dk, dv, dbeta.transpose(1, 2).contiguous(), dg.transpose(1, 2).contiguous()


def forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    *,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    cu_seqlens_list: list[int] | None = None,
    chunk_indices: object = None,
    chunk_indices_list: object = None,
    scale: float | None = None,
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    """Trade saved W/H/value activations for two fewer backward forward stages."""
    return fla_npu.forward(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        cu_seqlens,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        chunk_size=chunk_size,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        scale=scale,
        _reuse_intermediates=True,
    )


backward = fla_npu.backward
