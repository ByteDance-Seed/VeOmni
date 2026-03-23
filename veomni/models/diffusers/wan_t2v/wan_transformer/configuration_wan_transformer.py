# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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

import inspect
from typing import Optional, Tuple

import diffusers
from diffusers import WanTransformer3DModel
from transformers import PretrainedConfig


WAN_INIT_SIGNATURE = inspect.signature(WanTransformer3DModel.__init__)


diffusers_version = diffusers.__version__


class WanTransformer3DModelConfig(PretrainedConfig):
    model_type = "WanTransformer3DModel"
    condition_model_type = "WanTransformer3DConditionModel"

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        **kwargs,
    ):
        self.patch_size = patch_size

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.cross_attn_norm = cross_attn_norm
        self.qk_norm = qk_norm
        self.eps = eps
        self.image_dim = image_dim
        self.added_kv_proj_dim = added_kv_proj_dim
        self.rope_max_seq_len = rope_max_seq_len
        self.pos_embed_seq_len = pos_embed_seq_len
        super().__init__(**kwargs)

    def to_diffuser_dict(self):
        return {key: getattr(self, key) for key in WAN_INIT_SIGNATURE.parameters.keys() if key != "self"}

    def to_dict(self):
        return_dict = super().to_dict()
        return_dict["_class_name"] = "WanTransformer3DModel"
        return_dict["_diffusers_version"] = diffusers_version
        del return_dict["dtype"]
        return return_dict
