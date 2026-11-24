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

"""Config for DiffusionGemma as a SeedOmni module.

The module keeps the vendored ``DiffusionGemmaConfig`` fields unchanged -- the
canvas geometry, the encoder/decoder split, and every ``Eq. 13`` switch
(``encoder_ar_loss``, ``compute_all_canvas_df_loss``, ``canvas_batching``, ...) --
and only adds the ``model_type`` the SeedOmni registries key on.
``diffusion_gemma_omni`` is the only ``model_type`` for this family now: the
non-Omni entry that used to resolve ``diffusion_gemma`` was served by the removed
trainer and is not reachable from any launcher.
"""

from __future__ import annotations

from ....transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
from ..module_configuration_base import OmniModuleConfig


class DiffusionGemmaOmniConfig(OmniModuleConfig, DiffusionGemmaConfig):
    """``OmniModuleConfig`` view over the vendored block-diffusion config.

    ``canvas_length`` and the ``canvas_batching`` mode are read straight off this
    object by the module's batching hook, so they are deliberately not restated:
    the collator and the model must agree on one source of truth.
    """

    model_type = "diffusion_gemma_omni"


__all__ = ["DiffusionGemmaOmniConfig"]
