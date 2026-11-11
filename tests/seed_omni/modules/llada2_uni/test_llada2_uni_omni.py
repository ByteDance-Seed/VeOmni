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

"""Unit tests for the LLaDA2.0-Uni SeedOmni modules, HF layout, training DAG, and inference FSM."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image

from tests.tools.training_utils import make_eager_ops_config
from veomni.models.seed_omni.accelerated.utils.executor import TrainNodeRunner
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
)
from veomni.models.seed_omni.modules.llada2_uni.hf_layout import LLADA2_UNI_HF_LAYOUT
from veomni.models.seed_omni.modules.llada2_uni.image_tokenizer.processing import (
    LladaUniImageProcessor,
    LladaUniImageTokenizerPreprocessor,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.models.transformers.llada2_uni.configuration_llada2_uni import LLaDA2UniConfig
from veomni.models.transformers.llada2_uni.modeling_llada2_uni import LLaDA2UniForConditionalGeneration


TOY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "toy_config", "llada2_uni_toy")


class _DummyTokenizer:
    eos_token_id = 156892
    pad_token_id = 156892

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        if text == "<|image|>":
            return {"input_ids": [156901]}
        if text == "<|/image|>":
            return {"input_ids": [156902]}
        if text == "<boi>":
            return {"input_ids": [156904]}
        if text.startswith("<|reserved_token_") and text.endswith("|>"):
            n = int(text[len("<|reserved_token_") : -2])
            return {"input_ids": [156895 + n]}
        ids = [10 + (ord(c) % 80) for c in text] or [10]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        if len(ids) == 1 and 156895 <= int(ids[0]) < 157184:
            return f"<|reserved_token_{int(ids[0]) - 156895}|>"
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


def test_llada2_uni_omni_modules_are_registered():
    for model_type in (
        "llada2_uni_image_tokenizer",
        "llada2_uni_text_encoder",
        "llada2_uni_llm",
        "llada2_uni_decoder",
    ):
        assert model_type in OMNI_CONFIG_REGISTRY.valid_keys()
        assert model_type in OMNI_MODEL_REGISTRY.valid_keys()
        assert model_type in OMNI_ACCELERATED_MODEL_REGISTRY.valid_keys()
    assert "llada2_uni_image_tokenizer" in OMNI_PROCESSOR_REGISTRY.valid_keys()


def test_llada2_uni_hf_layout_routes_all_toy_checkpoint_keys(tmp_path: Path):
    cfg = LLaDA2UniConfig.from_pretrained(TOY_CONFIG_PATH, ops_implementation=make_eager_ops_config())
    model = LLaDA2UniForConditionalGeneration(cfg)

    for key in model.state_dict():
        routed = LLADA2_UNI_HF_LAYOUT.route(key)
        assert routed is not None, f"Unrouted key: {key}"
        module_name, module_key = routed
        assert LLADA2_UNI_HF_LAYOUT.source_key(module_name, module_key) == key


def test_llada2_uni_image_preprocessor_snaps_odd_grids():
    proc = LladaUniImageProcessor(
        size={"shortest_edge": 32 * 32, "longest_edge": 64 * 64},
        patch_size=16,
        merge_size=1,
    )
    preprocessor = LladaUniImageTokenizerPreprocessor(proc, dtype=torch.float32)
    img = Image.new("RGB", (48, 48), color=(128, 64, 32))
    item = ConversationItem(type="image", value=img, role="user")
    batch = {"conversation_list": [[item, ConversationItem(type="text", value="what is this?", role="user")]]}
    preprocessor(batch, inference=False)

    assert isinstance(item.value, torch.Tensor)
    grid = item.meta["_omni_grid"]
    assert grid[0] == 1
    assert grid[1] % 2 == 0 and grid[2] % 2 == 0
    assert item.value.shape[0] == grid[0] * grid[1] * grid[2]


def test_llada2_uni_training_dag_forward_and_backward():
    hf_cfg = LLaDA2UniConfig.from_pretrained(TOY_CONFIG_PATH, ops_implementation=make_eager_ops_config())
    hf_cfg.vocab_size = 157250
    hf_cfg.pad_token_id = 156892
    hf_cfg.eos_token_id = 156892
    hf_cfg.mask_token_id = 156895
    hf_cfg.image_token_offset = 157184
    hf_cfg.vq_config.num_embeddings = 32

    it_cfg = LLADA2_UNI_HF_LAYOUT.modules["llada2_uni_image_tokenizer"].build_config(hf_cfg)
    te_cfg = LLADA2_UNI_HF_LAYOUT.modules["llada2_uni_text_encoder"].build_config(hf_cfg)
    llm_cfg = LLADA2_UNI_HF_LAYOUT.modules["llada2_uni_llm"].build_config(hf_cfg)
    for cfg in (it_cfg, te_cfg, llm_cfg):
        cfg._set_ops_implementation(make_eager_ops_config())

    it_cls = OMNI_ACCELERATED_MODEL_REGISTRY["llada2_uni_image_tokenizer"]()
    te_cls = OMNI_ACCELERATED_MODEL_REGISTRY["llada2_uni_text_encoder"]()
    llm_cls = OMNI_ACCELERATED_MODEL_REGISTRY["llada2_uni_llm"]()

    it_mod = it_cls(it_cfg)
    te_mod = te_cls(te_cfg)
    te_mod.tokenizer = _DummyTokenizer()
    llm_mod = llm_cls(llm_cfg)

    training_graphs, generation_graphs = LLADA2_UNI_HF_LAYOUT.load_graphs()
    omni_cfg = OmniConfig(
        _module_entries={
            "llada2_uni_image_tokenizer": {"model_path": "llada2_uni_image_tokenizer"},
            "llada2_uni_text_encoder": {"model_path": "llada2_uni_text_encoder"},
            "llada2_uni_llm": {"model_path": "llada2_uni_llm"},
        },
        training_graphs=training_graphs,
        generation_graphs=generation_graphs,
        infer_type="infer_und",
    )
    omni = OmniModel(
        omni_cfg,
        {
            "llada2_uni_image_tokenizer": it_mod,
            "llada2_uni_text_encoder": te_mod,
            "llada2_uni_llm": llm_mod,
        },
    )

    patch_dim = hf_cfg.vision_config.in_channels * (hf_cfg.vision_config.patch_size**2)
    img_item = ConversationItem(
        type="image",
        value=torch.zeros(4, patch_dim, dtype=torch.float32),
        role="user",
        meta={"_omni_grid": [1, 2, 2], "llada2_uni_image_tokenizer_pixels": True},
    )
    user_text = ConversationItem(type="text", value="describe", role="user")
    asst_text = ConversationItem(type="text", value="a cat on a mat", role="assistant")

    batch = {"conversation_list": [[img_item, user_text, asst_text]]}
    runner = TrainNodeRunner()
    result = omni(batch, node_runner=runner)

    assert "loss" in result and result["loss"] is not None
    result["loss"].backward()

    assert te_mod.embed_tokens.weight.grad is not None
    assert te_mod.lm_head.weight.grad is not None
    assert any(p.grad is not None for p in llm_mod.parameters())
    assert all(p.grad is None for p in it_mod.parameters())


def test_llada2_uni_generation_fsm_i2t_and_t2i():
    hf_cfg = LLaDA2UniConfig.from_pretrained(TOY_CONFIG_PATH, ops_implementation=make_eager_ops_config())
    hf_cfg.vocab_size = 157250
    hf_cfg.pad_token_id = 156892
    hf_cfg.eos_token_id = 156892
    hf_cfg.mask_token_id = 156895
    hf_cfg.image_token_offset = 157184
    hf_cfg.vq_config.num_embeddings = 32

    it_cfg = LLADA2_UNI_HF_LAYOUT.modules["llada2_uni_image_tokenizer"].build_config(hf_cfg)
    te_cfg = LLADA2_UNI_HF_LAYOUT.modules["llada2_uni_text_encoder"].build_config(hf_cfg)
    llm_cfg = LLADA2_UNI_HF_LAYOUT.modules["llada2_uni_llm"].build_config(hf_cfg)
    for cfg in (it_cfg, te_cfg, llm_cfg):
        cfg._set_ops_implementation(make_eager_ops_config())

    it_mod = OMNI_MODEL_REGISTRY["llada2_uni_image_tokenizer"]()(it_cfg).eval()
    te_mod = OMNI_MODEL_REGISTRY["llada2_uni_text_encoder"]()(te_cfg).eval()
    te_mod.tokenizer = _DummyTokenizer()
    llm_mod = OMNI_MODEL_REGISTRY["llada2_uni_llm"]()(llm_cfg).eval()

    training_graphs, generation_graphs = LLADA2_UNI_HF_LAYOUT.load_graphs()
    omni_cfg = OmniConfig(
        _module_entries={
            "llada2_uni_image_tokenizer": {"model_path": "llada2_uni_image_tokenizer"},
            "llada2_uni_text_encoder": {"model_path": "llada2_uni_text_encoder"},
            "llada2_uni_llm": {"model_path": "llada2_uni_llm"},
        },
        training_graphs=training_graphs,
        generation_graphs=generation_graphs,
        infer_type="infer_und",
    )
    omni = OmniModel(
        omni_cfg,
        {
            "llada2_uni_image_tokenizer": it_mod,
            "llada2_uni_text_encoder": te_mod,
            "llada2_uni_llm": llm_mod,
        },
    )

    patch_dim = hf_cfg.vision_config.in_channels * (hf_cfg.vision_config.patch_size**2)
    img_item = ConversationItem(
        type="image",
        value=torch.zeros(4, patch_dim, dtype=torch.float32),
        role="user",
        meta={"_omni_grid": [1, 2, 2], "llada2_uni_image_tokenizer_pixels": True},
    )
    req = {"conversation_list": [img_item, ConversationItem(type="text", value="hi", role="user")]}
    omni.reset()
    generated = omni.generate(
        req,
        generation_kwargs={"block_length": 4, "gen_length": 4, "diffusion_steps": 2, "max_new_tokens": 10},
    )
    assert len(generated) == 1
    assert generated[0]["type"] == "text"
