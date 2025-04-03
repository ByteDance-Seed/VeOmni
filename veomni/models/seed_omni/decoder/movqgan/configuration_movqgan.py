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

from ....transformers.movqgan.configuration_movqgan import MoVQGANConfig
from ..base import BaseDecoderConfigMixin


class MoVQGANDecoderConfig(BaseDecoderConfigMixin, MoVQGANConfig):
    model_type = "movqgan_decoder"

    def __init__(
        self,
        freeze_codebook=True,
        use_semantic_decoder=False,
        semantic_dim=1280,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.freeze_codebook = freeze_codebook
        self.use_semantic_decoder = use_semantic_decoder
        self.semantic_dim = semantic_dim
