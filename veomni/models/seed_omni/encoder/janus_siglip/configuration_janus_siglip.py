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

from ....transformers.janus.configuration_janus import JanusVisionConfig
from ..base import BaseEncoderConfigMixin


class JanusSigLIPEncoderConfig(BaseEncoderConfigMixin, JanusVisionConfig):
    model_type = "janussiglip_encoder"

    def __init__(
        self,
        aligner_depth: int = 2,
        aligner_input_dim: int = 1024,
        n_embed: int = 2048,
        aligner_projector_type: str = "mlp_gelu",
        **kwargs,
    ):
        self.aligner_depth = aligner_depth
        self.aligner_input_dim = aligner_input_dim
        self.n_embed = n_embed
        self.aligner_projector_type = aligner_projector_type
        super().__init__(**kwargs)
