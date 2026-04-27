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

Only fired for the thinker tower's experts. The talker tower in v4 always runs
in eager mode (`nn.ModuleList`) and consumes per-expert HF keys natively, so
the converter intentionally does not match talker prefixes. The converter is
also a no-op when `_moe_implementation != "fused"` because eager mode also
uses `nn.ModuleList` for the thinker; the factory returns `None` in that case.
"""

import re

from .._moe_v4_converter import MoEV4StackingConverter


# Match thinker-tower per-expert keys only; talker stays in nn.ModuleList format on v4.
_EXPERT_PATTERN = re.compile(
    r"^(thinker\.model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def create_qwen3_omni_moe_v4_checkpoint_tensor_converter(model):
    """Factory registered on v4 Qwen3-Omni-MoE classes via `_create_checkpoint_tensor_converter`.

    Returns ``None`` outside fused mode: in eager mode the thinker uses
    ``nn.ModuleList`` whose per-expert FQNs already match the HF checkpoint.
    """
    config = model.config
    # The factory may be attached to either the top-level conditional-generation class
    # or to the inner thinker text model loaded standalone. Resolve the text config that
    # carries `num_experts` accordingly.
    if hasattr(config, "thinker_config"):
        text_config = config.thinker_config.text_config
    elif hasattr(config, "text_config"):
        text_config = config.text_config
    else:
        text_config = config

    moe_implementation = getattr(text_config, "_moe_implementation", None) or getattr(
        config, "_moe_implementation", "eager"
    )
    if moe_implementation != "fused":
        return None

    num_experts = text_config.num_experts
    return MoEV4StackingConverter(
        pattern=_EXPERT_PATTERN,
        num_experts_for=lambda _prefix: num_experts,
    )
