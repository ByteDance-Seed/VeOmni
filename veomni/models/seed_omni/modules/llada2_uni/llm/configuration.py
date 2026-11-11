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

"""Config for :class:`LladaUniLlm` (the LLaDA2.0-Uni bidirectional MoE backbone)."""

from typing import Any, Dict, Optional

from veomni.models.transformers.llada2_uni.configuration_llada2_uni import LLaDA2UniConfig

from ...module_configuration_base import OmniModuleConfig


class LladaUniLlmConfig(OmniModuleConfig):
    """Top-level config for the LLaDA2.0-Uni mask-predictor backbone (no wte / lm_head)."""

    model_type = "llada2_uni_llm"
    sub_configs = {"text_config": LLaDA2UniConfig}

    def __init__(
        self,
        text_config: Optional[Dict[str, Any] | LLaDA2UniConfig] = None,
        freeze: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if text_config is None:
            self.text_config = LLaDA2UniConfig()
        elif isinstance(text_config, LLaDA2UniConfig):
            self.text_config = text_config
        else:
            # Drop vision_config / vq_config if present in the source dict and not needed by the backbone,
            # or let LLaDA2UniConfig accept them cleanly.
            allowed = set(LLaDA2UniConfig.__dataclass_fields__)
            filtered = {k: v for k, v in text_config.items() if k in allowed}
            self.text_config = LLaDA2UniConfig(**filtered)
        self.freeze = freeze


__all__ = ["LladaUniLlmConfig"]
