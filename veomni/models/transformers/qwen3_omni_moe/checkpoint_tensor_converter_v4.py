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

"""
Runtime per-expert -> stacked converter for transformers v4 Qwen3-Omni-MoE.

The converter is registered on three classes that can each be the load entry,
and they expose different parameter prefixes:

    Qwen3OmniMoeForConditionalGeneration          → ``thinker.model.layers.*``
    Qwen3OmniMoeThinkerForConditionalGeneration   → ``model.layers.*``
    Qwen3OmniMoeThinkerTextModel                  → ``layers.*``

We pick the pattern at factory time from config introspection so each load
target only matches its own keys. The top-level pattern intentionally excludes
the talker tower: in v4 the talker keeps ``nn.ModuleList`` experts and consumes
HF per-expert keys natively, so passing them through unchanged is required.

The converter is also a no-op when ``_moe_implementation != "fused"`` because
the thinker uses ``nn.ModuleList`` in eager mode too — the factory returns
``None`` in that case.
"""

import re

from .._moe_v4_converter import MoEV4StackingConverter


_PROJ_SUFFIX = r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"

# Top-level Qwen3OmniMoeForConditionalGeneration: thinker only, talker is nn.ModuleList in v4.
_TOP_LEVEL_PATTERN = re.compile(r"^(thinker\.model\.layers\.\d+\.mlp)" + _PROJ_SUFFIX)
# Standalone Qwen3OmniMoeThinkerForConditionalGeneration: experts under `model.*`.
_THINKER_PATTERN = re.compile(r"^(model\.layers\.\d+\.mlp)" + _PROJ_SUFFIX)
# Standalone Qwen3OmniMoeThinkerTextModel: experts under `layers.*` (the model class IS the root).
_TEXT_MODEL_PATTERN = re.compile(r"^(layers\.\d+\.mlp)" + _PROJ_SUFFIX)


def create_qwen3_omni_moe_v4_checkpoint_tensor_converter(model):
    """Factory registered on v4 Qwen3-Omni-MoE classes via `_create_checkpoint_tensor_converter`.

    Returns ``None`` outside fused mode: in eager mode the thinker uses
    ``nn.ModuleList`` whose per-expert FQNs already match the HF checkpoint.
    """
    config = model.config
    # Resolve text config and the matching key-prefix pattern from whichever
    # config shape the model carries.
    if hasattr(config, "thinker_config"):
        # Top-level Qwen3OmniMoeConfig.
        text_config = config.thinker_config.text_config
        pattern = _TOP_LEVEL_PATTERN
    elif hasattr(config, "text_config"):
        # Qwen3OmniMoeThinkerConfig — standalone thinker.
        text_config = config.text_config
        pattern = _THINKER_PATTERN
    else:
        # Qwen3OmniMoeThinkerTextConfig — text model is the root.
        text_config = config
        pattern = _TEXT_MODEL_PATTERN

    moe_implementation = getattr(text_config, "_moe_implementation", None) or getattr(
        config, "_moe_implementation", "eager"
    )
    if moe_implementation != "fused":
        return None

    num_experts = text_config.num_experts
    return MoEV4StackingConverter(
        pattern=pattern,
        num_experts_for=lambda _prefix: num_experts,
    )
