"""Lazily imported Ascend compact-index attention kernels."""

from ....utils.import_utils import is_torch_npu_available


def qsa_attn_npu_fused(query, key, value, selected_indices, sm_scale=None, query_chunk_size=2048):
    if query.device.type != "npu" or not is_torch_npu_available():
        raise RuntimeError("QSA npu_fused requires Ascend NPU and torch_npu")
    from .npu_qsa import qsa_attn_npu_fused as impl

    return impl(query, key, value, selected_indices, sm_scale, query_chunk_size)
