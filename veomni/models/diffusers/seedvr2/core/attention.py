"""Native packed attention preserving SeedVR2's per-window sequence boundaries."""

import torch
import torch.nn.functional as F
from torch import nn


class FlashAttentionVarlen(nn.Module):
    """Source-compatible interface backed by native SDPA on CPU/GPU/NPU."""

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None):
        q_bounds = cu_seqlens_q.tolist()
        k_bounds = cu_seqlens_k.tolist()
        if len(q_bounds) != len(k_bounds) or len(q_bounds) < 2:
            raise ValueError("Packed query/key boundaries must describe the same nonempty batch.")
        if q_bounds[0] != 0 or q_bounds[-1] != q.shape[0] or k_bounds[0] != 0 or k_bounds[-1] != k.shape[0]:
            raise ValueError("Packed attention boundaries do not cover the input tensors.")
        outputs = []
        for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]):
            if qb <= qa or kb <= ka:
                raise ValueError("Packed attention segments must have positive lengths.")
            outputs.append(
                F.scaled_dot_product_attention(
                    q[qa:qb].transpose(0, 1).unsqueeze(0),
                    k[ka:kb].transpose(0, 1).unsqueeze(0),
                    v[ka:kb].transpose(0, 1).unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        return torch.cat(outputs, dim=0)
