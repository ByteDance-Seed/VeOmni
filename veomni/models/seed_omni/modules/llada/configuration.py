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

"""Config for the LLaDA single-module Omni wrapper.

The mask-predictor's own ``LLaDAConfig`` already carries everything the model
needs (``mask_token_id``, attention heads, vocabulary). Re-declaring those fields
here would add a second source of truth, so this module declares only the
``model_type`` slot the SeedOmni registries key on and defers the rest to the
vendored config on load.
"""

from __future__ import annotations

from ....transformers.llada.configuration_llada import LLaDAConfig
from ..module_configuration_base import OmniModuleConfig


class LladaOmniConfig(OmniModuleConfig, LLaDAConfig):
    """``OmniModuleConfig`` view over the vendored LLaDA mask-predictor config.

    Kept as an ``OmniModuleConfig`` subclass so the SeedOmni machinery
    (``_module_entries``, dummy-input binding, the ``VeOmniPretrainedConfig``
    lifecycle) applies, and a ``LLaDAConfig`` subclass so no field is restated.
    """

    model_type = "llada_omni"


__all__ = ["LladaOmniConfig"]
