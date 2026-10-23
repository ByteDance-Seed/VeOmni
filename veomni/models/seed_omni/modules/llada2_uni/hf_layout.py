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

"""``OmniHFLayout`` for LLaDA2.0-Uni (``model_type: llada2_uni``).

Allows ``OmniTrainer``, ``OmniInferencer``, and ``OmniModel.from_pretrained`` to
load directly from a ``LLaDA2.0-Uni-veomni`` checkpoint without an offline split.
"""

from __future__ import annotations

from typing import Any

from veomni.models.transformers.llada2_uni.configuration_llada2_uni import LLaDA2UniConfig

from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .decoder.configuration import LladaUniDecoderConfig
from .image_tokenizer.configuration import LladaUniImageTokenizerConfig
from .image_tokenizer.processing import LladaUniImageProcessor
from .llm.configuration import LladaUniLlmConfig
from .text_encoder.configuration import LladaUniTextEncoderConfig


def _load_tokenizer(model_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    return {"tokenizer": AutoTokenizer.from_pretrained(model_path)}


def _load_image_processor(model_path: str) -> dict[str, Any]:
    return {"image_processor": LladaUniImageProcessor.from_pretrained(model_path)}


def _load_llada2_uni_config(model_path: str) -> LLaDA2UniConfig:
    cfg = LLaDA2UniConfig.from_pretrained(model_path)
    cfg._name_or_path = str(model_path)
    return cfg


LLADA2_UNI_HF_LAYOUT = OmniHFLayout(
    modules={
        "llada2_uni_image_tokenizer": OmniHFModuleLayout(
            key_prefixes=(
                ("model.visual.", "visual."),
                ("model.vqmodel.", "vqmodel."),
            ),
            build_config=lambda hf: LladaUniImageTokenizerConfig(
                vision_config=hf.vision_config,
                vq_config=hf.vq_config,
                image_token_offset=hf.image_token_offset,
            ),
            build_assets=_load_image_processor,
        ),
        "llada2_uni_text_encoder": OmniHFModuleLayout(
            key_prefixes=(
                ("model.word_embeddings.", "embed_tokens."),
                ("lm_head.", "lm_head."),
            ),
            build_config=lambda hf: LladaUniTextEncoderConfig(
                vocab_size=hf.vocab_size,
                hidden_size=hf.hidden_size,
                tie_word_embeddings=hf.tie_word_embeddings,
                lm_head_bias=False,
                pad_token_id=hf.pad_token_id,
                mask_token_id=hf.mask_token_id,
                image_token_offset=hf.image_token_offset,
            ),
            build_assets=_load_tokenizer,
        ),
        "llada2_uni_llm": OmniHFModuleLayout(
            key_prefixes=(("model.", "language_model."),),
            build_config=lambda hf: LladaUniLlmConfig(text_config=hf.to_dict()),
        ),
        "llada2_uni_decoder": OmniHFModuleLayout(
            key_prefixes=(),
            build_config=lambda hf: LladaUniDecoderConfig(
                image_token_offset=hf.image_token_offset,
                asset_dir=getattr(hf, "_name_or_path", None) or None,
            ),
        ),
    },
    graph_dir="configs/seed_omni/Llada2_uni/llada2_uni_16b",
    training_graphs="graph_train.yaml",
    generation_graphs="graph_infer.yaml",
    infer_type="infer_und",
    load_hf_config=_load_llada2_uni_config,
)


@OMNI_HF_LAYOUT_REGISTRY.register("llada2_uni")
def _register_llada2_uni_hf_layout() -> OmniHFLayout:
    return LLADA2_UNI_HF_LAYOUT


__all__ = ["LLADA2_UNI_HF_LAYOUT"]
