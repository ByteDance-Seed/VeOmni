import torch
from torch import nn

from veomni.utils.import_utils import is_torch_npu_available


"""
This function is adapted from an open-source project, OpenDiT
For more details, see: https://github.com/NUS-HPC-AI-Lab/OpenDiT/blob/master/opendit/modules/layers.py
"""


def get_layernorm(hidden_size: torch.Tensor, eps: float = 1e-5, affine: bool = True, fused: bool = True):
    if fused and not is_torch_npu_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)
