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
"""Configuration for LLaDA-V: the LLaDA-8B mask predictor with a SigLIP image encoder.

The text fields are `LLaDAConfig`'s. ``vision_config`` describes the SigLIP tower
*as stored*: the upstream checkpoint drops every layer after the one LLaDA-V reads
(``mm_vision_select_layer: -2`` of 27), so the converter records the 26 layers that
are present and ``mm_vision_select_layer: -1``.
"""

from huggingface_hub.dataclasses import strict
from transformers import PreTrainedConfig, SiglipVisionConfig

from ..llada.configuration_llada import LLaDAConfig


# google/siglip2-so400m-patch14-384, truncated to the layers LLaDA-V reads.
_SIGLIP_SO400M_VISION = {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_attention_heads": 16,
    "num_hidden_layers": 26,
    "image_size": 384,
    "patch_size": 14,
    "vision_use_head": False,
}


@strict
class LLaDAVConfig(LLaDAConfig):
    r"""LLaDA text config plus the SigLIP tower, the ``mlp2x_gelu`` projector input width and
    the tower layer whose patch features feed the projector.

    Each image becomes ``side * (side + 1)`` placeholder tokens (``side = image_size //
    patch_size``): the patch grid with one ``image_newline`` appended to every row.
    """

    model_type = "llada_v"
    sub_configs = {"vision_config": SiglipVisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    mm_hidden_size: int | None = None
    mm_vision_select_layer: int = -1

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = SiglipVisionConfig(**_SIGLIP_SO400M_VISION)
        elif isinstance(self.vision_config, dict):
            self.vision_config = SiglipVisionConfig(**self.vision_config)
        if self.mm_hidden_size is None:
            self.mm_hidden_size = self.vision_config.hidden_size
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if self.mm_hidden_size != self.vision_config.hidden_size:
            raise ValueError(
                f"`mm_hidden_size` ({self.mm_hidden_size}) must equal the vision tower width "
                f"({self.vision_config.hidden_size}): the projector reads the tower's patch features."
            )

    @property
    def image_seq_len(self) -> int:
        side = self.vision_config.image_size // self.vision_config.patch_size
        return side * (side + 1)


__all__ = ["LLaDAVConfig"]
