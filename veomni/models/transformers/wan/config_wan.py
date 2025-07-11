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

import json
import os

from transformers import PretrainedConfig


try:
    import diffusers

    diffusers_version = diffusers.__version__

except ModuleNotFoundError:
    raise ImportError("diffusers is not installed")


class WanConfig(PretrainedConfig):
    model_type = "wan"

    def __init__(
        self,
        patch_size=[1, 2, 2],
        dim=5120,
        eps=1e-06,
        ffn_dim=13824,
        freq_dim=256,
        in_dim=36,
        num_heads=40,
        num_layers=40,
        out_dim=16,
        text_dim=4096,
        text_len=512,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.dim = dim
        self.eps = eps
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.text_dim = text_dim
        self.text_len = text_len
        super().__init__(**kwargs)

    # @classmethod
    # def from_pretrained(cls, path):
    #     config = AutoConfig.from_pretrained(path)
    #     return cls(**config.to_dict())

    def save_pretrained(self, path):
        config = {
            "_class_name": "WanModel",
            "_diffusers_version": diffusers_version,
            "dim": self.dim,
            "eps": self.eps,
            "ffn_dim": self.ffn_dim,
            "freq_dim": self.freq_dim,
            "in_dim": self.in_dim,
            "model_type": "t2v" if self.has_image_input == "false" else "i2v",
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "out_dim": self.out_dim,
            "text_len": self.text_len,
        }

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
