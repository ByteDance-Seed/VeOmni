from functools import lru_cache
from typing import Optional

import torch
from cann_ops_transformer import ops


@lru_cache(maxsize=8)
def get_sparse_flash_mla_metadata(
    B,
    N1,
    N2,
    D,
    K,
    has_cmp,
    cmp_ratio,
    max_q,
    max_ori_kv,
    max_cmp_kv,
    residual,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
):
    # cmp_residual_kv must match the tensor passed to the main op so the cmp-side causal
    # mask is rebuilt with pre-compression length = cmp_len * cmp_ratio + residual.
    # Equal-length BSND -> one shared per-batch scalar.
    cmp_residual_kv = torch.full((B,), residual, dtype=torch.int32).npu() if has_cmp else None
    return ops.sparse_flash_mla_metadata(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        batch_size=B,
        max_seqlen_q=max_q,
        max_seqlen_ori_kv=max_ori_kv,  # split out of the old single max_seqlen_kv
        max_seqlen_cmp_kv=max_cmp_kv,  # split out of the old single max_seqlen_kv
        cmp_residual_kv=cmp_residual_kv,  # keep consistent with the main op
        ori_topk=0,
        cmp_topk=K if has_cmp else 0,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q="BSND",
        layout_kv="BSND",
        has_ori_kv=True,
        has_cmp_kv=has_cmp,
    )


class SparseFlashMlaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        cmp_residual_kv,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
    ):
        B, S, N1, D = q.shape
        N2 = ori_kv.shape[2]
        S2 = ori_kv.shape[1]
        S3 = cmp_kv.shape[1] if cmp_kv is not None else 0
        has_cmp = cmp_kv is not None
        K = cmp_sparse_indices.shape[-1] if (has_cmp and cmp_ratio == 4 and cmp_sparse_indices is not None) else 0
        residual = int(cmp_residual_kv[0].item()) if (has_cmp and cmp_residual_kv is not None) else 0

        metadata = get_sparse_flash_mla_metadata(
            B,
            N1,
            N2,
            D,
            K,
            has_cmp,
            cmp_ratio,
            S,
            S2,
            S3,
            residual,
            ori_mask_mode,
            cmp_mask_mode,
            ori_win_left,
            ori_win_right,
        )

        result, softmax_lse = ops.sparse_flash_mla(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            cmp_residual_kv=cmp_residual_kv,
            # cmp_topk_length is reserved in this prototype and must stay None.
            sinks=sinks,
            metadata=metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q="BSND",
            layout_kv="BSND",
            return_softmax_lse=True,
        )

        ctx.save_for_backward(q, ori_kv, cmp_kv, result, softmax_lse, cmp_sparse_indices, cmp_residual_kv, sinks)
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.has_cmp = has_cmp
        ctx.K = K
        ctx.mark_non_differentiable(softmax_lse)
        ctx.B, ctx.S1, ctx.S2 = B, S, S2
        ctx.N1, ctx.N2, ctx.D = N1, N2, D

        return result, softmax_lse

    @staticmethod
    def backward(ctx, d_out, d_softmax_lse):
        (q, ori_kv, cmp_kv, result, softmax_lse, cmp_sparse_indices, cmp_residual_kv, sinks) = ctx.saved_tensors

        cmp_S2 = ctx.S2 // ctx.cmp_ratio
        cmp_residual_kv_meta = (
            torch.tensor([int(cmp_residual_kv[0].item())], dtype=torch.int32, device=q.device)
            if ctx.has_cmp and cmp_residual_kv is not None
            else None
        )
        grad_metadata = ops.sparse_flash_mla_grad_metadata(
            num_heads_q=ctx.N1,
            num_heads_kv=ctx.N2,
            head_dim=ctx.D,
            cmp_residual_kv=cmp_residual_kv_meta,
            batch_size=ctx.B,
            max_seqlen_q=ctx.S1,
            max_seqlen_ori_kv=ctx.S2,
            max_seqlen_cmp_kv=cmp_S2,
            ori_topk=0,
            cmp_topk=ctx.K if ctx.has_cmp else 0,
            cmp_ratio=ctx.cmp_ratio,
            ori_mask_mode=ctx.ori_mask_mode,
            cmp_mask_mode=ctx.cmp_mask_mode,
            ori_win_left=ctx.ori_win_left,
            ori_win_right=ctx.ori_win_right,
            layout_q="BSND",
            layout_kv="BSND",
            has_ori_kv=True,
            has_cmp_kv=ctx.has_cmp,
        )

        (dq, dori_kv, dcmp_kv, dsinks, _ori_l1, _cmp_l1) = ops.sparse_flash_mla_grad(
            q,
            d_out.contiguous(),  # dout (positional)
            result,  # attn_out (positional)
            softmax_lse,  # softmax_lse (positional)
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=None,
            cmp_sparse_indices=cmp_sparse_indices,
            cu_seqlens_q=None,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
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
            layout_q="BSND",
            layout_kv="BSND",
        )

        return (
            dq,  # q
            dori_kv,  # ori_kv
            dcmp_kv if ctx.has_cmp else None,  # cmp_kv
            None,  # cmp_sparse_indices
            None,  # cmp_residual_kv
            dsinks,  # sinks
            None,
            None,
            None,
            None,
            None,
            None,  # softmax_scale .. ori_win_right
        )


def npu_sparse_flash_mla(
    q: torch.Tensor,  # [B, S, N1, D]
    ori_kv: torch.Tensor,  # [B, S2, 1, D]
    cmp_kv: Optional[torch.Tensor],  # [B, S3, 1, D] or None
    cmp_sparse_indices: Optional[torch.Tensor],  # [B, S, k] int32, -1=invalid, or None
    *,
    softmax_scale: Optional[float] = None,
    cmp_ratio: int = 4,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    sinks: Optional[torch.Tensor] = None,
    cmp_residual_kv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SCFA, BSND. fwd: sparse_flash_mla / bwd: sparse_flash_mla_grad. Returns [B, S, N1, D]."""
    if cmp_ratio == 0:
        cmp_ratio = 1
    B, S, N1, D = q.shape
    if softmax_scale is None:
        softmax_scale = D**-0.5

    has_cmp = cmp_kv is not None

    # SCFA: [B,S,k] -> [B,S,1,K], K in {512, 1024}, pad invalid slots with -1.
    if has_cmp and cmp_ratio == 4 and cmp_sparse_indices is not None:
        k = cmp_sparse_indices.shape[-1]
        K = 512 if k <= 512 else 1024
        assert k <= 1024, f"index_topk={k} exceeds sfmla K limit (1024)"
        idx = cmp_sparse_indices.to(torch.int32)
        if k < K:
            idx = torch.cat([idx, idx.new_full((B, S, K - k), -1)], dim=-1)
        cmp_sparse_indices = idx.unsqueeze(2).contiguous()  # [B, S, 1, K]
    else:
        cmp_sparse_indices = None

    if cmp_residual_kv is None and has_cmp:
        # pre-compression length = cmp_len * cmp_ratio + residual; equal-length BSND.
        cmp_residual_kv = torch.full((B,), S % cmp_ratio, dtype=torch.int32, device=q.device)

    result, _ = SparseFlashMlaFunction.apply(
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        cmp_residual_kv,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
    )
    return result
