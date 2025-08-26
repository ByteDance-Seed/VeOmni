# Copyright 2024-2025 The Alibaba Qwen Team Authors. All rights reserved.
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

class QwenImageConfig(PretrainedConfig):
    model_type = "qwen_image"

    def __init__(
        self,
        attention_head_dim=128,
        axes_dims_rope=[16, 56, 56],
        guidance_embeds=False,
        in_channels=64,
        joint_attention_dim=3584,
        num_attention_heads=24,
        num_layers=60,
        out_channels=16,
        patch_size=2,
        pooled_projection_dim=768,
        **kwargs
    ):
        self.attention_head_dim = attention_head_dim
        self.axes_dims_rope = axes_dims_rope
        self.guidance_embeds = guidance_embeds
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.pooled_projection_dim = pooled_projection_dim

        super().__init__(** kwargs)