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

# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import List, Optional, Tuple, Union

from transformers import LlamaConfig, PretrainedConfig


class JanusVisionConfig(PretrainedConfig):
    model_type = "janus"

    def __init__(
        self,
        width: int = 1152,
        layers: Union[Tuple[int, int, int, int], int] = 27,
        heads: int = 16,
        patch_size: int = 14,
        image_size: Union[Tuple[int, int], int] = 336,
        global_pool: str = "map",
        mlp_ratio: float = 3.7362,
        class_token: bool = False,
        num_classes: int = 0,
        use_checkpoint: bool = False,
        select_feature: str = "patch",
        select_layer: int = -2,
        pixel_mean: Optional[List[float]] = None,
        pixel_std: Optional[List[float]] = None,
        ignore_head: bool = True,
        weight_init: str = "skip",
        **kwargs,
    ):
        self.width = width
        self.layers = layers
        self.heads = heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.global_pool = global_pool
        self.mlp_ratio = mlp_ratio
        self.class_token = class_token
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.select_feature = select_feature
        self.select_layer = select_layer

        # https://github.com/deepseek-ai/Janus/blob/1daa72fa409002d40931bd7b36a9280362469ead/janus/models/clip_encoder.py#L61
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        # https://github.com/deepseek-ai/Janus/blob/1daa72fa409002d40931bd7b36a9280362469ead/janus/models/siglip_vit.py#L653
        self.ignore_head = ignore_head
        self.weight_init = weight_init
        if select_layer <= 0:
            self.layers = min(self.layers, self.layers + select_layer + 1)
        else:
            self.layers = min(self.layers, select_layer)

        super().__init__(**kwargs)


class JanusGenVisionConfig(PretrainedConfig):
    model_type = "janus"

    def __init__(
        self,
        codebook_size: int = 16384,
        codebook_embed_dim: int = 8,
        codebook_l2_norm: bool = True,
        codebook_show_usage: bool = True,
        commit_loss_beta: float = 0.25,
        entropy_loss_ratio: float = 0.0,
        encoder_ch_mult: List[int] = [1, 1, 2, 2, 4],
        decoder_ch_mult: List[int] = [1, 1, 2, 2, 4],
        z_channels: int = 256,
        dropout_p: float = 0.0,
        **kwargs,
    ):
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.codebook_l2_norm = codebook_l2_norm
        self.codebook_show_usage = codebook_show_usage
        self.commit_loss_beta = commit_loss_beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.encoder_ch_mult = encoder_ch_mult
        self.decoder_ch_mult = decoder_ch_mult
        self.z_channels = z_channels
        self.dropout_p = dropout_p
        super().__init__(**kwargs)


class JanusConfig(PretrainedConfig):
    model_type = "janus"

    def __init__(
        self,
        vision_config=None,
        gen_vision_config=None,
        language_config=None,
        aligner_depth=2,
        aligner_projector_type="mlp_gelu",
        gen_aligner_depth=2,
        gen_aligner_projector_type="mlp_gelu",
        gen_head_embed=2048,
        **kwargs,
    ):
        if vision_config is None:
            self.vision_config = JanusVisionConfig()
        else:
            self.vision_config = JanusVisionConfig(**vision_config)

        if gen_vision_config is None:
            self.gen_vision_config = JanusGenVisionConfig()
        else:
            self.gen_vision_config = JanusGenVisionConfig(**gen_vision_config)

        if language_config is None:
            self.language_config = LlamaConfig()
        else:
            self.language_config = LlamaConfig(**language_config)

        self.n_embed = self.language_config.hidden_size
        self.aligner_input_dim = self.vision_config.width
        self.gen_aligner_input_dim = self.gen_vision_config.codebook_embed_dim
        self.image_token_size = self.gen_vision_config.codebook_size

        self.aligner_depth = aligner_depth
        self.aligner_projector_type = aligner_projector_type
        self.gen_aligner_depth = gen_aligner_depth
        self.gen_aligner_projector_type = gen_aligner_projector_type

        self.gen_head_embed = gen_head_embed
        super().__init__(**kwargs)
