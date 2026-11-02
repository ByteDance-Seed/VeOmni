# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Weighted EP token combine with an explicit FP32 accumulation contract."""

import torch


def eager(tokens, weights, indices, *, num_tokens):
    """Multiply in the promoted dtype, scatter-add in FP32, then cast back."""
    weighted = tokens * weights[:, None]
    output = torch.zeros((num_tokens, tokens.shape[1]), dtype=torch.float32, device=tokens.device)
    output.scatter_add_(0, indices[:, None].expand_as(tokens), weighted.float())
    return output.to(weighted.dtype)


def triton(tokens, weights, indices, *, num_tokens):
    """Fuse weighting and unpermutation without a full FP32 routed-token copy."""
    from .unpermute_triton import WeightedUnpermute

    return WeightedUnpermute.apply(tokens, weights, indices, num_tokens)
