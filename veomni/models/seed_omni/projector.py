# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from transformers import ViTConfig, ViTModel


def build_feature_projector(
    feature_size: int,
    hidden_size: int,
    mlp_depth: int = 2,
) -> "torch.nn.Module":
    """
    Builds feature projector containing several linear layers.
    """
    layers = [torch.nn.Linear(feature_size, hidden_size)]
    for _ in range(mlp_depth - 1):
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Linear(hidden_size, hidden_size))

    return torch.nn.Sequential(*layers)


def build_vit_decoder(
    input_dim: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    output_dim: int,
):
    class ViT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            vit_config = ViTConfig(
                hidden_size=hidden_dim,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                intermediate_size=hidden_dim * 4,
                image_size=32,
                patch_size=1,
                num_channels=input_dim,  # Input feature channel size
            )
            self.projector = torch.nn.Linear(hidden_dim, output_dim)
            self.model = ViTModel(vit_config)

        def forward(self, x):
            x = self.model(x).last_hidden_state
            x = x[:, 1:, :]  # skip cls_token
            return self.projector(x)

    return ViT()
