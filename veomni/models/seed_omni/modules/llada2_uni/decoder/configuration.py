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

"""Config for :class:`LladaUniDecoder` (VQ tokens → image: ZImage DiT + Flux VAE)."""

from ...module_configuration_base import OmniModuleConfig


class LladaUniDecoderConfig(OmniModuleConfig):
    """Omni-module config for the LLaDA2.0-Uni image decoder (hyper-params only)."""

    model_type = "llada2_uni_decoder"

    def __init__(
        self,
        image_token_offset: int = 157184,
        resolution_multiplier: int = 2,
        decoder_num_steps: int = 50,
        decode_mode: str = "normal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_token_offset = image_token_offset
        self.resolution_multiplier = resolution_multiplier
        self.decoder_num_steps = decoder_num_steps
        self.decode_mode = decode_mode


__all__ = ["LladaUniDecoderConfig"]
