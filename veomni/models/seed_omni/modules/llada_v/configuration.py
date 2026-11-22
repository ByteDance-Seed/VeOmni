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

"""Config for LLaDA-V as a SeedOmni module.

The module keeps the vendored ``LLaDAVConfig`` fields unchanged -- the text
backbone, the SigLIP ``vision_config``, ``mm_hidden_size`` and the projector's
source layer -- and only adds the ``model_type`` the SeedOmni registries key on.
``llada_v_omni`` stays disjoint from the vendored ``llada_v`` entry the non-Omni
trainer resolves.
"""

from __future__ import annotations

from ....transformers.llada_v.configuration_llada_v import LLaDAVConfig
from ..module_configuration_base import OmniModuleConfig


class LladaVOmniConfig(OmniModuleConfig, LLaDAVConfig):
    """``OmniModuleConfig`` view over the vendored LLaDA-V config.

    ``image_seq_len`` stays the property it is on the vendored config -- it is
    derived from the tower geometry (``side * (side + 1)`` placeholder tokens per
    image) and restating it here would create a second source of truth the
    ``image_mask`` check would then disagree with.
    """

    model_type = "llada_v_omni"


__all__ = ["LladaVOmniConfig"]
