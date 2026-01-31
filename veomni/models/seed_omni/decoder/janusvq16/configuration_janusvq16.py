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

from ....transformers.janus.configuration_janus import JanusGenVisionConfig
from ..base import BaseDecoderConfigMixin


class JanusVQ16DecoderConfig(BaseDecoderConfigMixin, JanusGenVisionConfig):
    model_type = "janusvq16_decoder"

    def __init__(
        self,
        gen_aligner_depth: int = 2,
        gen_aligner_input_dim: int = 8,
        n_embed: int = 2048,
        gen_aligner_projector_type: str = "mlp_gelu",
        gen_head_embed: int = 2048,
        projector_train_from_scratch: bool = False,
        train_origin_projector: bool = False,
        **kwargs,
    ):
        self.gen_aligner_depth = gen_aligner_depth
        self.gen_aligner_input_dim = gen_aligner_input_dim
        self.n_embed = n_embed
        self.gen_aligner_projector_type = gen_aligner_projector_type
        self.gen_head_embed = gen_head_embed
        self.projector_train_from_scratch = projector_train_from_scratch
        self.train_origin_projector = train_origin_projector
        super().__init__(**kwargs)
