# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Run DeepSeek-V4.1 straight off the released checkpoint.

Registered under ``OMNI_HF_LAYOUT_REGISTRY["deepseek_v41"]``. Point
``model.model_path`` at the release and the four modules load from it with no
offline split; the HF export is written back in the release's own keys, dtypes
and shards.

The rename is the offline splitter's (:mod:`.released_names`), listed key by
key because most of it sits mid-key (``attn_norm`` -> ``input_layernorm``) and
depends on the layer count, which is why the layout is built from the source's
``config.json``. Each module's checkpoint tensor converter does the rest at
load: the llm stacks the per-expert ``w1``/``w2``/``w3`` and, for training,
dequantizes fp8/fp4; the n-gram dequantizes and pads its tables. The n-gram's
``token_map`` is not in the release -- it is built from the tokenizer, as the
offline splitter bakes it.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any

import torch
from transformers import PretrainedConfig

from ...utils.hf_layout import (
    OMNI_HF_LAYOUT_REGISTRY,
    OmniHFLayout,
    OmniHFModuleLayout,
    source_tensor_specs,
    source_weight_files,
)
from .convert_model import _build_configs, build_ngram_token_map, build_vision_image_processor
from .export import export_llm_state, refuse_ngram_export
from .quant_scheme import QUANT_BLOCK_SIZE
from .released_names import (
    LLM,
    NGRAM,
    TEXT_ENCODER,
    VISION,
    release_key_prefixes,
    released_scale_name,
    tied_mtp_names,
)


_GRAPH_DIR = "configs/seed_omni/DeepSeek/deepseek_v41_flash"
_MODULE_INDEX = {VISION: 0, TEXT_ENCODER: 1, NGRAM: 2, LLM: 3}


def _read_release_config(source_path: str) -> PretrainedConfig:
    """``AutoConfig`` does not know ``deepseek_v41``; the module configs only need the parsed file."""
    with open(os.path.join(source_path, "config.json"), encoding="utf-8") as f:
        return PretrainedConfig.from_dict(json.load(f))


def _module_config(name: str):
    def build(hf_config: PretrainedConfig):
        return _build_configs(hf_config.to_dict())[_MODULE_INDEX[name]]

    return build


def _tokenizer(source_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    return {"tokenizer": AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)}


def _image_processor(source_path: str) -> dict[str, Any]:
    vision_config = _build_configs(_read_release_config(source_path).to_dict())[_MODULE_INDEX[VISION]]
    return {"image_processor": build_vision_image_processor(vision_config)}


def _token_map(source_path: str, config) -> dict[str, Any]:
    return {"token_map": build_ngram_token_map(source_path, config)}


def _release_blocks(source_path: str, layout: OmniHFLayout) -> tuple[dict[str, tuple[int, int]], int]:
    """The llm's quantization blocks as the release's scales have them, from its headers only.

    Returns each fp8 dense weight's ``(rows, cols)`` block under its module key,
    and the fp4 experts' column block.
    """
    specs = source_tensor_specs(source_path, source_weight_files(source_path))
    fp8_blocks: dict[str, tuple[int, int]] = {}
    fp4_blocks: set[int] = set()
    for name, (dtype, shape) in specs.items():
        scale = specs.get(released_scale_name(name)) if name.endswith(".weight") else None
        routed = layout.route(name) if scale is not None else None
        if routed is None or routed[0] != LLM:
            continue
        scale_shape = scale[1]
        if ".mlp.experts." in routed[1]:
            if dtype != torch.int8:
                raise ValueError(f"{source_path}: expert '{name}' is {dtype}, export expects packed fp4 (int8).")
            fp4_blocks.add(2 * shape[1] // scale_shape[1])
        else:
            fp8_blocks[routed[1]] = (shape[0] // scale_shape[0], shape[1] // scale_shape[1])
    if len(fp4_blocks) > 1:
        raise ValueError(f"{source_path}: experts mix fp4 column blocks {sorted(fp4_blocks)}.")
    return fp8_blocks, fp4_blocks.pop() if fp4_blocks else QUANT_BLOCK_SIZE


def build_deepseek_v41_hf_layout(source_path: str) -> OmniHFLayout:
    """The layout for the release at ``source_path``, its renames sized by its ``config.json``."""
    hf_config = _read_release_config(source_path)
    text_config = hf_config.to_dict()["text_config"]
    prefixes = release_key_prefixes(text_config)
    num_stages = text_config.get("num_nextn_predict_layers", 0) or 0

    @functools.cache
    def release_blocks() -> tuple[dict[str, tuple[int, int]], int]:
        return _release_blocks(source_path, layout)

    def llm_export(model, state):
        fp8_blocks, fp4_block = release_blocks()
        return export_llm_state(model, state, fp8_blocks=fp8_blocks, fp4_block=fp4_block)

    layout = OmniHFLayout(
        modules={
            VISION: OmniHFModuleLayout(
                key_prefixes=prefixes[VISION],
                build_config=_module_config(VISION),
                build_assets=_image_processor,
            ),
            TEXT_ENCODER: OmniHFModuleLayout(
                key_prefixes=prefixes[TEXT_ENCODER],
                build_config=_module_config(TEXT_ENCODER),
                build_assets=_tokenizer,
            ),
            NGRAM: OmniHFModuleLayout(
                key_prefixes=prefixes[NGRAM],
                build_config=_module_config(NGRAM),
                generated_tensors=_token_map,
                export_state=refuse_ngram_export,
            ),
            LLM: OmniHFModuleLayout(
                key_prefixes=prefixes[LLM],
                build_config=_module_config(LLM),
                export_state=llm_export,
            ),
        },
        tied_source_keys=tied_mtp_names(num_stages),
        graph_dir=_GRAPH_DIR,
        training_graphs="train/graph_train.yaml",
        generation_graphs={"vision_understanding": "infer/graph_infer.yaml"},
        infer_type="vision_understanding",
        load_hf_config=_read_release_config,
    )
    return layout


@OMNI_HF_LAYOUT_REGISTRY.register("deepseek_v41")
def _register_deepseek_v41_hf_layout(source_path: str | None = None) -> OmniHFLayout:
    if source_path is None:
        raise ValueError("The deepseek_v41 HF layout is sized by the release's config.json; it needs its directory.")
    return build_deepseek_v41_hf_layout(source_path)


__all__ = ["build_deepseek_v41_hf_layout"]
