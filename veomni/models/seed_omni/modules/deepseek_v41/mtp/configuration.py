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

"""Configuration for the standalone DeepSeek-V4.1 DSpark / MTP OmniModule."""

from __future__ import annotations

from ..llm.configuration import DeepseekV41LlmConfig


class DeepseekV41MtpConfig(DeepseekV41LlmConfig):
    """Config for ``deepseek_v41_mtp`` (the DSpark draft stages).

    Inherits the shared attention / MoE / mHC / RoPE / quantization fields from
    :class:`DeepseekV41LlmConfig` so ``DeepseekV41DSparkBlock`` and its inner
    layers read the exact same schema and per-site override indices
    (``layers.{num_hidden_layers + stage}.*``) as before the module split.
    """

    model_type = "deepseek_v41_mtp"

    def __init__(self, freeze: bool = False, **kwargs) -> None:
        self.freeze = freeze
        super().__init__(**kwargs)
        if not isinstance(self.freeze, bool):
            raise ValueError(f"freeze must be a bool, got {self.freeze!r}")


__all__ = ["DeepseekV41MtpConfig"]
