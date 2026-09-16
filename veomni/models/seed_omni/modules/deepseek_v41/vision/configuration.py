# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Config for :class:`DeepseekV41VisionEncoder` -- ViT + aligner + span delimiters."""

from typing import Optional

from transformers import PretrainedConfig


class DeepseekV41VisionEncoderConfig(PretrainedConfig):
    """DeepSeek-V4.1's vision tower.

    Fields are the checkpoint's ``vision_config`` verbatim, plus ``out_hidden_size``
    -- the aligner projects into the *backbone's* hidden size, which the released
    config only states once, under ``text_config.hidden_size``. It is declared here
    rather than read across because a module config has to stand alone: the split
    checkpoint gives this module its own ``config.json`` and nothing in it would
    otherwise pin the output width.

    Unlike Qwen3-VL there is no nested HF sub-config: transformers 5.16.1 ships no
    ``deepseek_v41_vision`` model, so the ViT is written out in ``modeling.py`` and
    reads these fields directly.
    """

    model_type = "deepseek_v41_vision"

    def __init__(
        self,
        num_hidden_layers: int = 32,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        intermediate_size: int = 2816,
        patch_size: int = 14,
        rope_theta: float = 10000.0,
        downsample_ratio: int = 3,
        max_image_tokens: int = 1024,
        min_pixels: int = 295936,
        max_wh_ratio: Optional[float] = None,
        out_hidden_size: int = 5120,
        freeze: bool = False,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.downsample_ratio = downsample_ratio
        self.max_image_tokens = max_image_tokens
        self.min_pixels = min_pixels
        self.max_wh_ratio = max_wh_ratio
        self.out_hidden_size = out_hidden_size
        self.freeze = freeze
        super().__init__(**kwargs)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rope_dim(self) -> int:
        """Half the head dim: 2D RoPE spends one half on rows and one on columns."""
        return self.hidden_size // self.num_attention_heads // 2


__all__ = ["DeepseekV41VisionEncoderConfig"]
