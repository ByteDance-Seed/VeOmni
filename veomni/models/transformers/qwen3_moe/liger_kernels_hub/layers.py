"""Stateless wrappers around liger kernel classes for kernels-hub compatibility.

The ``kernels`` library expects replacement layers to be stateless
``nn.Module`` subclasses (no custom ``__init__``, no extra members).  These
thin wrappers satisfy that contract while delegating to the underlying liger
triton kernels.  Missing attributes that the liger forward expects (e.g.
``offset``, ``casting_mode``) are lazily patched with sensible defaults so
the wrapper can be bound to the original model module via ``MethodType``.
"""

import torch.nn as nn
from liger_kernel.transformers.rms_norm import LigerRMSNorm as _OrigLigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb as _orig_liger_rotary_pos_emb


class LigerRMSNorm(nn.Module):
    def forward(self, hidden_states):
        return _OrigLigerRMSNorm.forward(self, hidden_states)


def liger_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Wrapper that adapts liger's ``liger_rotary_pos_emb`` to the Qwen3MoE signature.

    The upstream transformers library defines ``apply_rotary_pos_emb`` with six
    parameters (``q, k, cos, sin, position_ids=None, unsqueeze_dim=1``), but the
    patched Qwen3MoE modeling code drops the ``position_ids`` argument and only
    passes five (``q, k, cos, sin, unsqueeze_dim=1``).  This wrapper bridges the
    gap so the liger kernel can be used as a drop-in replacement.
    """
    return _orig_liger_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=unsqueeze_dim)
