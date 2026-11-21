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

"""``OmniHFLayout`` for LLaDA (``model_type: llada``).

Lets ``OmniTrainer`` / ``OmniModel.from_pretrained`` read a
``LLaDA-8B-Instruct-veomni`` checkpoint in place, without an offline split.

The family is a single module, so the layout is the degenerate case: one entry
that claims every weight key unchanged. ``key_prefixes=()`` means "no rewriting"
-- the ``OmniHFModuleLayout`` docstring reserves an empty tuple for a module whose
source names already match, which is what a one-module family always is.
"""

from __future__ import annotations

from typing import Any

from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .configuration import LladaOmniConfig


def _load_tokenizer(model_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    return {"tokenizer": AutoTokenizer.from_pretrained(model_path)}


def _build_llada_config(hf: Any) -> LladaOmniConfig:
    """Turn the upstream config dict into this module's config.

    ``LLaDAConfig`` is a ``@strict`` dataclass, so ``from_dict`` materialises
    every key in the source dict as an *instance* attribute -- including
    ``model_type: "llada"``, which then shadows the class attribute
    ``"llada_omni"`` the SeedOmni registries key on. ``convert_model.py`` reads
    that attribute to pick the module class, so leaving it as ``llada`` makes the
    conversion fail with ``Unknown OmniModel name``.
    """
    config = LladaOmniConfig.from_dict(hf.to_dict())
    config.model_type = LladaOmniConfig.model_type
    return config


def _load_llada_config(model_path: str) -> LladaOmniConfig:
    """Read the checkpoint's own config through the Omni subclass.

    ``from_pretrained`` on the subclass picks up the vendored ``LLaDAConfig``
    fields (``mask_token_id``, head counts, vocabulary) and stamps the
    ``llada_omni`` ``model_type`` the registries key on.
    """
    cfg = LladaOmniConfig.from_pretrained(model_path)
    cfg.model_type = LladaOmniConfig.model_type
    cfg._name_or_path = str(model_path)
    return cfg


LLADA_HF_LAYOUT = OmniHFLayout(
    modules={
        "llada_omni": OmniHFModuleLayout(
            # `("", "")` -- the empty source prefix matches every key, so this
            # one module claims the whole checkpoint under its own names. An
            # empty *tuple* would mean "this module owns no keys at all":
            # `_rules` flattens the pairs, so `()` yields no rule, `route()`
            # returns None for every key, and `load_module_from_hf_source` takes
            # its weight-free branch and leaves the model uninitialised.
            key_prefixes=(("", ""),),
            build_config=_build_llada_config,
            build_assets=_load_tokenizer,
        ),
    },
    graph_dir="configs/seed_omni/Llada/llada_8b",
    training_graphs="train/graph_train.yaml",
    generation_graphs="infer/graph_infer_gen.yaml",
    load_hf_config=_load_llada_config,
)


@OMNI_HF_LAYOUT_REGISTRY.register("llada")
def _register_llada_hf_layout() -> OmniHFLayout:
    return LLADA_HF_LAYOUT


__all__ = ["LLADA_HF_LAYOUT"]
