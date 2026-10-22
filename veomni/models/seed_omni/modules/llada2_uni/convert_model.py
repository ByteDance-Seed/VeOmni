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

"""Split a LLaDA2.0-Uni checkpoint into SeedOmni module subfolders.

Supports both:
* ``model_type == "llada2_uni"`` (converted by ``scripts/convert_llada2uni_to_veomni.py``),
  handled automatically by ``LLADA2_UNI_HF_LAYOUT`` via ``convert_with_hf_layout``;
* ``model_type == "llada2_moe"`` (upstream ``inclusionAI/LLaDA2.0-Uni`` directory with
  ``image_tokenizer/`` subfolder), which converts into a temporary ``llada2_uni`` view
  first and then splits via ``convert_with_hf_layout``.
"""

from __future__ import annotations

import tempfile
from typing import Any

from ...utils.convert_registry import OMNI_CONVERT_REGISTRY
from ...utils.hf_layout import convert_with_hf_layout


def convert_llada2_uni_from_raw(model_path: str, **kwargs: Any) -> dict[str, Any]:
    """Convert raw upstream ``inclusionAI/LLaDA2.0-Uni`` (``model_type: llada2_moe``) into SeedOmni modules."""
    from scripts.convert_llada2uni_to_veomni import convert_checkpoint as convert_raw_to_veomni

    with tempfile.TemporaryDirectory(prefix="llada2_uni_veomni_") as veomni_dir:
        convert_raw_to_veomni(model_path, veomni_dir, copy_tokenizer=True)
        return convert_with_hf_layout(veomni_dir, **kwargs)


@OMNI_CONVERT_REGISTRY.register("llada2_moe")
def _register_llada2_moe_convert():
    return convert_llada2_uni_from_raw


__all__ = ["convert_llada2_uni_from_raw"]
