# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import torch.nn as nn


class ByT5Mapper(nn.Module):
    """
    ByT5Mapper: Maps ByT5 encoder outputs to a new space, with optional residual connection.

    Args:
        in_dim (int): Input dimension (must equal out_dim if use_residual).
        out_dim (int): Output dimension after second linear layer.
        hidden_dim (int): Hidden dimension for intermediate layer.
        out_dim1 (int): Final output dimension.
        use_residual (bool): Whether to use residual connection (default: True).
    """

    def __init__(self, in_dim, out_dim, hidden_dim, out_dim1, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim1)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        """
        Forward pass for ByT5Mapper.

        Args:
            x (Tensor): Input tensor of shape (..., in_dim).

        Returns:
            Tensor: Output tensor of shape (..., out_dim1).
        """
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x2 = self.act_fn(x)
        x2 = self.fc3(x2)
        if self.use_residual:
            x2 = x2 + residual
        return x2
