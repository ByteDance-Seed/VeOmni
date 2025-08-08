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

    def __init__(self, disable_guidance_embedder=False, input_dim=64, num_blocks=19, **kwargs):
        self.disable_guidance_embedder = disable_guidance_embedder
        self.input_dim = input_dim
        self.num_blocks = num_blocks

        super().__init__(**kwargs)
