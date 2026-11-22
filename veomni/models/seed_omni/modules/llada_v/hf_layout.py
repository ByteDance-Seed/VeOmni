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

"""``OmniHFLayout`` for LLaDA-V (``model_type: llava_llada`` in the source).

Lets ``OmniTrainer`` / ``OmniModel.from_pretrained`` read a ``LLaDA-V-VeOmni``
checkpoint in place, without an offline split.

The family is a single module, so the layout is the degenerate case: one entry
that claims every weight key unchanged. ``key_prefixes=(("", ""),)`` is what
expresses that -- the empty *source* prefix matches every key, including the
``model.vision_tower.`` / ``model.mm_projector`` names the tower and projector
own. An empty tuple would mean "this module owns no keys at all", and the loader
would take its weight-free branch and leave the model uninitialised.
"""

from __future__ import annotations

from typing import Any

from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .configuration import LladaVOmniConfig


def _load_tokenizer(model_path: str) -> dict[str, Any]:
    from transformers import AutoImageProcessor, AutoTokenizer

    return {
        "tokenizer": AutoTokenizer.from_pretrained(model_path),
        "image_processor": AutoImageProcessor.from_pretrained(model_path),
    }


def _build_llada_v_config(hf: Any) -> LladaVOmniConfig:
    """Turn the converted config dict into this module's config.

    ``LLaDAVConfig`` is a ``@strict`` dataclass, so ``from_dict`` materialises every
    key in the source dict as an *instance* attribute -- including
    ``model_type: "llada_v"``, which then shadows the class attribute
    ``"llada_v_omni"`` the SeedOmni registries key on. ``convert_model.py`` reads
    that attribute to pick the module class, so leaving it as ``llada_v`` makes the
    conversion fail with ``Unknown OmniModel name``.
    """
    config = LladaVOmniConfig.from_dict(hf.to_dict())
    config.model_type = LladaVOmniConfig.model_type
    return config


def _load_llada_v_config(model_path: str) -> LladaVOmniConfig:
    """Read the checkpoint's ``config.json`` and restore the registry key."""
    cfg = LladaVOmniConfig.from_pretrained(model_path)
    cfg.model_type = LladaVOmniConfig.model_type
    cfg._name_or_path = str(model_path)
    return cfg


LLADA_V_HF_LAYOUT = OmniHFLayout(
    modules={
        "llada_v_omni": OmniHFModuleLayout(
            # `("", "")` -- the empty source prefix matches every key, so this one
            # module claims the whole checkpoint under its own names.
            key_prefixes=(("", ""),),
            build_config=_build_llada_v_config,
            build_assets=_load_tokenizer,
        ),
    },
    graph_dir="configs/seed_omni/Llada_v/llada_v_8b",
    training_graphs="train/graph_train.yaml",
    generation_graphs="infer/graph_infer_gen.yaml",
    load_hf_config=_load_llada_v_config,
)


@OMNI_HF_LAYOUT_REGISTRY.register("llada_v")
def _register_llada_v_hf_layout() -> OmniHFLayout:
    return LLADA_V_HF_LAYOUT


__all__ = ["LLADA_V_HF_LAYOUT"]
