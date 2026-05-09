import torch
import torch.nn as nn
import torch_npu


class NPUFusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        hidden_states = torch.cat([gate, hidden_states], dim=-1)
        hidden_states = torch_npu.npu_swiglu(hidden_states, dim=-1)

        return hidden_states
