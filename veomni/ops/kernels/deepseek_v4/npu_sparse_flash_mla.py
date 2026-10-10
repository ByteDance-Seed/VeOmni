from functools import lru_cache
from typing import Optional

import torch


def get_cmp_cu_seqlens(cu_seqlens, ratio):
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    compressed_lengths = lengths // ratio
    compressed = torch.cat((compressed_lengths.new_zeros(1), compressed_lengths.cumsum(0, dtype=torch.int32)))
    return compressed, int(compressed_lengths.max().item()) if compressed_lengths.numel() else 0


_CUSTOM_OPS = None


def _custom_ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is not None:
        return _CUSTOM_OPS
    try:
        import cann_ops_transformer.ops as custom_ops
    except ImportError:
        custom_ops = None
    _CUSTOM_OPS = custom_ops
    return _CUSTOM_OPS


@lru_cache(maxsize=8)
def get_sparse_attn_sharedkv_metadata(
    B,
    S1,
    S2,
    N1,
    D,
    N2,
    K,
    cmp_S2,
    has_cmp,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
    layout_q,
    layout_kv,
):
    cmp_residual_k = torch.full((B,), S2 % cmp_ratio, dtype=torch.int32, device="npu") if has_cmp else None

    metadata = _custom_ops().sparse_flash_mla_metadata(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        cmp_residual_kv=cmp_residual_k,
        ori_topk_length=None,
        cmp_topk_length=None,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_ori_kv=S2,
        max_seqlen_cmp_kv=cmp_S2,
        cmp_topk=K,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=True,
        has_cmp_kv=has_cmp,
    )
    return metadata


@lru_cache(maxsize=8)
def get_sparse_flash_mla_grad_metadata(
    ctx_N1,
    ctx_N2,
    ctx_D,
    ctx_B,
    ctx_S1,
    ctx_S2,
    cmp_S2,
    cmp_topk,
    ctx_has_cmp,
    ctx_cmp_ratio,
    ctx_ori_mask_mode,
    ctx_cmp_mask_mode,
    ctx_ori_win_left,
    ctx_ori_win_right,
    ctx_layout_q,
    ctx_layout_kv,
):
    cmp_residual_kv = (
        torch.full((ctx_B,), ctx_S2 % ctx_cmp_ratio, dtype=torch.int32, device="npu") if ctx_has_cmp else None
    )

    grad_metadata = _custom_ops().sparse_flash_mla_grad_metadata(
        cu_seqlens_q=None,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_ori_kv=None,
        seqused_cmp_kv=None,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=None,
        cmp_topk_length=None,
        num_heads_q=ctx_N1,
        num_heads_kv=ctx_N2,
        head_dim=ctx_D,
        batch_size=ctx_B,
        max_seqlen_q=ctx_S1,
        max_seqlen_ori_kv=ctx_S2,
        max_seqlen_cmp_kv=cmp_S2,
        ori_topk=0,
        cmp_topk=cmp_topk if ctx_has_cmp else 0,
        cmp_ratio=ctx_cmp_ratio,
        ori_mask_mode=ctx_ori_mask_mode,
        cmp_mask_mode=ctx_cmp_mask_mode,
        ori_win_left=ctx_ori_win_left,
        ori_win_right=ctx_ori_win_right,
        layout_q=ctx_layout_q,
        layout_kv=ctx_layout_kv,
        has_ori_kv=True,
        has_cmp_kv=ctx_has_cmp,
    )
    return grad_metadata


class SparseFlashMlaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        ori_kv: torch.Tensor,
        cmp_kv: torch.Tensor,
        cmp_sparse_indices: torch.Tensor,
        ori_block_table: torch.Tensor,
        cmp_block_table: torch.Tensor,
        cmp_residual_kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_ori_kv: torch.Tensor,
        cu_seqlens_cmp_kv: torch.Tensor,
        sinks: torch.Tensor,
        softmax_scale: float,
        cmp_ratio: int,
        ori_mask_mode: int,
        cmp_mask_mode: int,
        ori_win_left: int,
        ori_win_right: int,
        layout_q: str,
        layout_kv: str,
    ):
        K = cmp_sparse_indices.shape[-1] if cmp_ratio == 4 else 0
        has_cmp = cmp_kv is not None

        if layout_q == "BSND":
            B, S1, N1, D = q.shape
            S2 = ori_kv.shape[1]
            N2 = ori_kv.shape[2]
            max_seqlen_q = S1
            max_seqlen_kv = S2
            max_seqlen_cmp_kv = S2 // cmp_ratio
            metadata = get_sparse_attn_sharedkv_metadata(
                B,
                S1,
                S2,
                N1,
                D,
                N2,
                K,
                max_seqlen_cmp_kv,
                has_cmp,
                cmp_ratio,
                ori_mask_mode,
                cmp_mask_mode,
                ori_win_left,
                ori_win_right,
                layout_q,
                layout_kv,
            )
        else:
            S1, N1, D = q.shape
            S2, N2, _ = ori_kv.shape
            B = len(cu_seqlens_q) - 1
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            max_seqlen_q = seqlens_q.max().item()
            seqlens_kv = cu_seqlens_ori_kv[1:] - cu_seqlens_ori_kv[:-1]
            max_seqlen_kv = seqlens_kv.max().item()
            max_seqlen_cmp_kv = (seqlens_kv // cmp_ratio).max().item()
            metadata = _custom_ops().sparse_flash_mla_metadata(
                num_heads_q=N1,
                num_heads_kv=N2,
                head_dim=D,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                cmp_residual_kv=cmp_residual_kv,
                ori_topk_length=None,
                cmp_topk_length=None,
                batch_size=B,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_ori_kv=max_seqlen_kv,
                max_seqlen_cmp_kv=max_seqlen_cmp_kv,
                cmp_topk=K,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                has_ori_kv=True,
                has_cmp_kv=has_cmp,
            )

        result, softmax_lse = _custom_ops().sparse_flash_mla(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            return_softmax_lse=True,
        )

        ctx.save_for_backward(
            q,
            ori_kv,
            cmp_kv,
            result,
            softmax_lse,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
        )
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.layout_q = layout_q
        ctx.layout_kv = layout_kv
        ctx.has_cmp = has_cmp
        ctx.B, ctx.S1, ctx.S2, ctx.N1, ctx.N2, ctx.D = B, S1, S2, N1, N2, D
        ctx.K = K
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_ori_kv = max_seqlen_kv
        ctx.max_seqlen_cmp_kv = max_seqlen_cmp_kv
        ctx.mark_non_differentiable(softmax_lse)

        return result

    @staticmethod
    def backward(ctx, d_out):
        (
            q,
            ori_kv,
            cmp_kv,
            result,
            softmax_lse,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
        ) = ctx.saved_tensors

        if ctx.layout_q == "BSND":
            cmp_S2 = ctx.S2 // ctx.cmp_ratio
            grad_metadata = get_sparse_flash_mla_grad_metadata(
                ctx.N1,
                ctx.N2,
                ctx.D,
                ctx.B,
                ctx.S1,
                ctx.S2,
                cmp_S2,
                ctx.K,
                ctx.has_cmp,
                ctx.cmp_ratio,
                ctx.ori_mask_mode,
                ctx.cmp_mask_mode,
                ctx.ori_win_left,
                ctx.ori_win_right,
                ctx.layout_q,
                ctx.layout_kv,
            )
        else:
            grad_metadata = _custom_ops().sparse_flash_mla_grad_metadata(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                cmp_residual_kv=cmp_residual_kv,
                ori_topk_length=None,
                cmp_topk_length=None,
                num_heads_q=ctx.N1,
                num_heads_kv=ctx.N2,
                head_dim=ctx.D,
                batch_size=ctx.B,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_ori_kv=ctx.max_seqlen_ori_kv,
                max_seqlen_cmp_kv=ctx.max_seqlen_cmp_kv,
                ori_topk=0,
                cmp_topk=ctx.K,
                cmp_ratio=ctx.cmp_ratio,
                ori_mask_mode=ctx.ori_mask_mode,
                cmp_mask_mode=ctx.cmp_mask_mode,
                ori_win_left=ctx.ori_win_left,
                ori_win_right=ctx.ori_win_right,
                layout_q=ctx.layout_q,
                layout_kv=ctx.layout_kv,
                has_ori_kv=True,
                has_cmp_kv=ctx.has_cmp,
            )

        (
            dq,
            dori_kv,
            dcmp_kv,
            dsinks,
            ori_softmax_l1,
            cmp_softmax_l1,
        ) = _custom_ops().sparse_flash_mla_grad(
            q,
            d_out.contiguous(),
            result,
            softmax_lse,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=None,
            cmp_sparse_indices=cmp_sparse_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            cmp_residual_kv=cmp_residual_kv,
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=sinks,
            metadata=grad_metadata,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.cmp_ratio,
            ori_mask_mode=ctx.ori_mask_mode,
            cmp_mask_mode=ctx.cmp_mask_mode,
            ori_win_left=ctx.ori_win_left,
            ori_win_right=ctx.ori_win_right,
            layout_q=ctx.layout_q,
            layout_kv=ctx.layout_kv,
        )

        # ori/cmp_softmax_l1 are the attn_softmax_out for the kl_div side loss; export them via
        # an external channel (kl is computed outside this Function, not returned through autograd).

        return (
            dq,  # q
            dori_kv,  # ori_kv
            dcmp_kv if ctx.has_cmp else None,  # cmp_kv
            None,  # cmp_sparse_indices
            None,  # ori_block_table
            None,  # cmp_block_table
            None,  # cmp_residual_kv
            None,  # cu_seqlens_q
            None,  # cu_seqlens_ori_kv
            None,  # cu_seqlens_cmp_kv
            dsinks,  # sinks
            None,  # softmax_scale
            None,  # cmp_ratio
            None,  # ori_mask_mode
            None,  # cmp_mask_mode
            None,  # ori_win_left
            None,  # ori_win_right
            None,  # layout_q
            None,  # layout_kv
        )


def npu_sparse_flash_mla(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: Optional[torch.Tensor],
    cmp_sparse_indices: Optional[torch.Tensor],
    *,
    softmax_scale: Optional[float] = None,
    cmp_ratio: int = 4,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    sinks: Optional[torch.Tensor] = None,
    cmp_residual_kv: Optional[torch.Tensor] = None,
    packed_sequence_slices=None,
) -> torch.Tensor:
    """Sparse shared-KV attention with BSND and packed TND dispatch."""
    if _custom_ops() is None:
        raise ImportError("cann_ops_transformer is required for NPU sparse FlashMLA.")
    if cmp_ratio == 0:
        cmp_ratio = 1
    batch, seq_len, _, head_dim = q.shape
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    has_cmp = cmp_kv is not None

    if cmp_sparse_indices is not None and cmp_ratio != 4:
        raise ValueError(f"cmp_sparse_indices are supported only when cmp_ratio=4; got cmp_ratio={cmp_ratio}")
    if has_cmp and cmp_ratio == 4 and cmp_sparse_indices is not None:
        index_topk = cmp_sparse_indices.shape[-1]
        if index_topk > 1024:
            raise ValueError(f"index_topk={index_topk} exceeds sfmla K limit (1024)")
        padded_topk = 512 if index_topk <= 512 else 1024
        indices = cmp_sparse_indices.to(torch.int32)
        if index_topk < padded_topk:
            indices = torch.cat(
                (indices, indices.new_full((*indices.shape[:-1], padded_topk - index_topk), -1)), dim=-1
            )
    else:
        indices = None

    use_tnd = packed_sequence_slices is not None and len(packed_sequence_slices) > 1
    if use_tnd:
        if batch != 1:
            raise ValueError(f"Packed TND sparse FlashMLA expects batch size 1, got {batch}")
        lengths = torch.tensor(
            [end - start for start, end in packed_sequence_slices], dtype=torch.int32, device=q.device
        )
        cu_seqlens_q = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        if int(cu_seqlens_q[-1].item()) != seq_len:
            raise ValueError("Packed sequence slices must span the complete attention sequence")
        cu_seqlens_ori_kv = cu_seqlens_q
        cu_seqlens_cmp_kv, _ = get_cmp_cu_seqlens(cu_seqlens_ori_kv, cmp_ratio)
        if has_cmp and int(cu_seqlens_cmp_kv[-1].item()) != cmp_kv.shape[1]:
            raise ValueError("Packed compressed boundaries do not match compressed KV length")
        cmp_residual_kv = lengths % cmp_ratio if has_cmp else None
        q_in = q[0].contiguous()
        ori_kv_in = ori_kv[0].contiguous()
        cmp_kv_in = cmp_kv[0].contiguous() if has_cmp else None
        indices_in = indices[0].unsqueeze(1).contiguous() if indices is not None else None
        layout = "TND"
    else:
        cu_seqlens_q = cu_seqlens_ori_kv = cu_seqlens_cmp_kv = None
        if cmp_residual_kv is None and has_cmp:
            cmp_residual_kv = torch.full((batch,), seq_len % cmp_ratio, dtype=torch.int32, device=q.device)
        q_in = q
        ori_kv_in = ori_kv
        cmp_kv_in = cmp_kv
        indices_in = indices.unsqueeze(2).contiguous() if indices is not None else None
        layout = "BSND"

    output = SparseFlashMlaFunction.apply(
        q_in,
        ori_kv_in,
        cmp_kv_in,
        indices_in,
        None,
        None,
        cmp_residual_kv,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout,
        layout,
    )
    return output.unsqueeze(0) if use_tnd else output
