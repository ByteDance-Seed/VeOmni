# Copyright 2024-2025 The Black-forest-labs Authors. All rights reserved.
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


class FluxConfig(PretrainedConfig):
    model_type = "flux"

    def __init__(
        self,
        disable_guidance_embedder=False,
        input_dim=64,
        output_dim=64,
        num_blocks=19,
        num_single_layers=38,
        num_attention_heads=24,
        attention_head_dim=128,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        timestep_embedding_dim=256,
        axes_dims_rope=(16, 56, 56),
        rope_theta=10000,
        **kwargs,
    ):
        self.disable_guidance_embedder = disable_guidance_embedder
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.num_single_layers = num_single_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.timestep_embedding_dim = timestep_embedding_dim
        self.axes_dims_rope = list(axes_dims_rope)
        self.rope_theta = rope_theta

        super().__init__(**kwargs)
