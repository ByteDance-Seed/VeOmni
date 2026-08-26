from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
from cann_ops_transformer import ops

from veomni.ops.kernels.deepseek_v4.npu_sparse_flash_mla import get_sparse_flash_mla_metadata


@lru_cache(maxsize=8)
def _kl_grad_metadata(B, S, T, N1_idx, D_idx, K, residual, mask_mode, cmp_ratio):
    # Uses the indexer's N/D (index_head_dim=128, index_n_heads), NOT the main attn 512/N.
    cmp_res = torch.full((B,), residual, dtype=torch.int32).npu()
    return ops.sparse_lightning_indexer_kl_loss_grad_metadata(
        N1_idx,
        1,
        D_idx,
        cmp_residual_k=cmp_res,
        batch_size=B,
        max_seqlen_q=S,
        max_seqlen_k=T,
        topk=K,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )


def _compute_indexer_loss(attn_softmax_out, indexer_softmax_out, eps=1e-9):
    y = attn_softmax_out
    Y = indexer_softmax_out
    reduce_target = torch.sum(y, dim=-1, keepdim=True)
    norm_target = torch.div(y, reduce_target + eps)

    log_p = torch.clamp(norm_target, min=eps).log()
    log_Y = (Y + eps).log()
    tmp = log_p - log_Y
    result = tmp * y
    return result.sum(dim=-1).mean()


class SparseFlashMlaWithIndexerLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        query_index,
        key_index,
        weights,
        cmp_residual_kv,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        indexer_loss_coeff,
        loss_tracker,
    ):
        B, S, N1, D = q.shape
        N2 = ori_kv.shape[2]
        S2 = ori_kv.shape[1]
        S3 = cmp_kv.shape[1] if cmp_kv is not None else 0
        has_cmp = cmp_kv is not None
        K = cmp_sparse_indices.shape[-1] if (has_cmp and cmp_ratio == 4) else 0
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
            # cmp_topk_length reserved -> None
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

        ctx.save_for_backward(
            q,
            ori_kv,
            cmp_kv,
            result,
            softmax_lse,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            query_index,
            key_index,
            weights,
        )
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.has_cmp = has_cmp
        ctx.indexer_loss_coeff = indexer_loss_coeff
        ctx.loss_tracker = loss_tracker
        ctx.B, ctx.S, ctx.N1, ctx.N2, ctx.D, ctx.K = B, S, N1, N2, D, K
        ctx.S2 = S2
        ctx.mark_non_differentiable(softmax_lse)
        return result, softmax_lse

    @staticmethod
    def backward(ctx, d_out, d_softmax_lse):
        (
            q,
            ori_kv,
            cmp_kv,
            result,
            softmax_lse,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            query_index,
            key_index,
            weights,
        ) = ctx.saved_tensors

        # grads for the indexer triple; stay None when there is no compressed branch
        d_query_index = d_key_index = d_weights = None

        # Grad metadata must carry the full config (mask modes, cmp ratio, seq lens,
        # residual), otherwise the grad kernel rebuilds the cmp-side causal mask with
        # defaults and both the gradients and cmp_softmax_l1 come out wrong.
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
            max_seqlen_q=ctx.S,
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

        # 1) main sparse-attn backward -> also yields cmp_softmax_l1 (the KL target distribution p)
        (dq, dori_kv, dcmp_kv, dsinks, _ori_l1, cmp_softmax_l1) = ops.sparse_flash_mla_grad(
            q,
            d_out.contiguous(),
            result,
            softmax_lse,
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

        # 2) indexer KL backward: target = cmp_softmax_l1, indices shared with the main attn
        if ctx.has_cmp and ctx.cmp_ratio == 4 and cmp_softmax_l1 is not None:
            N1_idx, D_idx = query_index.shape[2], query_index.shape[3]  # indexer N/D (!= main attn)
            T = key_index.shape[1]
            residual = int(cmp_residual_kv[0].item()) if cmp_residual_kv is not None else 0

            kl_metadata = _kl_grad_metadata(
                ctx.B,
                ctx.S,
                T,
                N1_idx,
                D_idx,
                ctx.K,
                residual,
                ctx.cmp_mask_mode,
                ctx.cmp_ratio,
            )

            d_query_index, d_key_index, d_weights, softmax_out = ops.sparse_lightning_indexer_kl_loss_grad(
                query_index,
                key_index,
                weights,
                cmp_sparse_indices,
                cmp_softmax_l1,
                cmp_residual_k=cmp_residual_kv,
                metadata=kl_metadata,
                layout_q="BSND",
                layout_k="BSND",
                mask_mode=ctx.cmp_mask_mode,
                cmp_ratio=ctx.cmp_ratio,
            )

            # KL op returns grads SUMMED over all query tokens; the eager loss is a per-token
            # mean. Divide by B*S so the indexer grad magnitude matches the eager path.
            scale = ctx.indexer_loss_coeff / (ctx.B * ctx.S)
            d_query_index = d_query_index * scale
            d_key_index = d_key_index * scale
            d_weights = d_weights * scale

            if ctx.loss_tracker is not None:
                loss = _compute_indexer_loss(F.normalize(cmp_softmax_l1, p=1, dim=-1), softmax_out)
                ctx.loss_tracker(loss * ctx.indexer_loss_coeff)

        return (
            dq,
            dori_kv,
            dcmp_kv if ctx.has_cmp else None,
            None,  # cmp_sparse_indices
            d_query_index,  # query_index
            d_key_index,  # key_index
            d_weights,  # weights
            None,  # cmp_residual_kv
            dsinks,  # sinks
            None,
            None,
            None,
            None,
            None,
            None,  # softmax_scale .. ori_win_right
            None,  # indexer_loss_coeff
            None,  # loss_tracker
        )


def npu_sparse_flash_mla_with_indexer_loss(
    q: torch.Tensor,  # [B, S, N1, 512]
    ori_kv: torch.Tensor,  # [B, S2, 1, 512]
    cmp_kv: torch.Tensor,  # [B, T, 1, 512]
    top_k_indices: torch.Tensor,  # [B, S, k] int32, -1=invalid
    query_index: torch.Tensor,  # [B, S, N1_idx, 128]  (indexer)
    key_index: torch.Tensor,  # [B, T, 1, 128]
    weights: torch.Tensor,  # [B, S, N1_idx]
    *,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    cmp_ratio: int = 4,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    indexer_loss_coeff: float = 1.0,
    loss_tracker=None,
) -> torch.Tensor:
    """SCFA + indexer KL loss (BSND). fwd: sparse_flash_mla, bwd: sparse_flash_mla_grad + KL grad.
    Returns [B, S, N1, 512].
    """
    B, S, N1, D = q.shape
    if softmax_scale is None:
        softmax_scale = D**-0.5

    # K pad to {512, 1024}; the main attn and KL share the same indices.
    k = top_k_indices.shape[-1]
    K = 512 if k <= 512 else 1024
    assert k <= 1024, f"index_topk={k} exceeds K limit (1024)"
    idx = top_k_indices.to(torch.int32)
    if k < K:
        idx = torch.cat([idx, idx.new_full((B, S, K - k), -1)], dim=-1)
    cmp_sparse_indices = idx.unsqueeze(2).contiguous()  # [B, S, 1, K]

    if cmp_ratio == 0:
        cmp_ratio = 1
    cmp_residual_kv = torch.full((B,), S % cmp_ratio, dtype=torch.int32, device=q.device)

    # KL op requires w in float32; keep the indexer triple contiguous.
    query_index = query_index.to(torch.bfloat16).contiguous()
    key_index = key_index.contiguous()
    weights = weights.float().contiguous()

    result, _ = SparseFlashMlaWithIndexerLossFunction.apply(
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        query_index,
        key_index,
        weights,
        cmp_residual_kv,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        indexer_loss_coeff,
        loss_tracker,
    )
    return result
