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

from transformers import PretrainedConfig


class MoVQGANConfig(PretrainedConfig):
    model_type = "movqgan"

    def __init__(
        self,
        embed_dim=4,
        n_embed=16384,
        double_z=False,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=256,
        ch_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(32,),
        dropout=0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        # base config
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        # ddconfig
        self.double_z = double_z
        self.z_channels = z_channels
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = out_ch
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        # init config
        self.initializer_range = initializer_range
        super().__init__(**kwargs)
