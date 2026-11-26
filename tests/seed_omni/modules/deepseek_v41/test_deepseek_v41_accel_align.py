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

"""Accelerated runtime hooks, shipped YAML recipe validation, and ViT Ulysses SP parity for DeepSeek-V4.1."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from veomni.arguments.arguments_types import AcceleratorConfig
from veomni.arguments.omni_arguments_types import (
    OmniModuleRuntimeArguments,
    _validate_omni_dp_size,
    build_omni_module_runtime_args,
)
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.distributed.sequence_parallel import sp_pad_and_slice
from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import (
    DeepseekV41LlmAccelerated,
    build_packed_image_mask,
    fold_indexer_kl_loss,
)
from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
from veomni.models.seed_omni.modules.deepseek_v41.vision.configuration import DeepseekV41VisionEncoderConfig
from veomni.models.seed_omni.modules.deepseek_v41.vision.modeling import (
    DeepseekV41VisionEncoder,
    build_deepseek_v41_vit_metadata,
)
from veomni.models.seed_omni.modules.deepseek_v41.vision.processing import image_token_types
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.ops import OpsSelection


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = REPO_ROOT / "configs/seed_omni/DeepSeek/deepseek_v41_flash"
HIDDEN = 64
N_LAYERS = 8


def make_llm_config(**overrides) -> DeepseekV41LlmConfig:
    kwargs = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=48,
        o_lora_rank=16,
        o_groups=2,
        sliding_window=6,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=4,
        rms_norm_eps=1e-20,
        hc_mult=4,
        compress_ratios=[2] * N_LAYERS + [0, 0, 0],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0, 4],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        engram_layer_ids=[1, 5],
        dspark_target_layer_ids=[5, 6, 7],
        dspark_n_routed_experts=8,
        max_position_embeddings=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_shared_experts=1,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=8,
    )
    kwargs.update(overrides)
    return DeepseekV41LlmConfig(**kwargs)


def _text(length: int, role: str = "user") -> ConversationItem:
    return ConversationItem(
        type="text",
        value=torch.randn(length, HIDDEN),
        role=role,
        meta={"input_ids": torch.randint(1, 100, (length,))},
    )


class TestAcceleratedHooks:
    def test_image_mask_and_indexer_kl_folding(self):
        img = ConversationItem(type="image", value=torch.randn(3, HIDDEN), role="user")
        mask = build_packed_image_mask([[_text(4), img, _text(2)]], torch.device("cpu"))
        assert torch.equal(mask, torch.tensor([[False] * 4 + [True] * 3 + [False] * 2]))
        assert build_packed_image_mask([[_text(4), _text(2)]], torch.device("cpu")) is None

        loss = torch.tensor(0.84, requires_grad=True)
        folded = fold_indexer_kl_loss({"indexer_kl_total": loss, "indexer_query_tokens": 2}, coef=0.5)
        assert folded is not None
        torch.testing.assert_close(folded, torch.tensor(0.21))
        assert fold_indexer_kl_loss({"indexer_kl_total": loss, "indexer_query_tokens": 2}, coef=0.0) is None

    def test_dummy_image_gets_zero_gradient_in_training(self, monkeypatch):
        import veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated as mod

        monkeypatch.setattr(mod, "get_parallel_state", lambda: type("PS", (), {"cp_size": 1})())
        model = DeepseekV41LlmAccelerated(make_llm_config()).train()

        anchor = torch.randn(9, HIDDEN, requires_grad=True)
        conv = [
            [
                _text(4),
                _text(3, role="assistant"),
                ConversationItem(type="image", value=anchor, role="user", is_dummy=True),
            ]
        ]
        packed = torch.cat([conv[0][0].value, conv[0][1].value])
        out = model.forward_pre(conversation_list=conv)
        out["inputs_embeds"].sum().backward()

        torch.testing.assert_close(out["inputs_embeds"][0], packed, rtol=0, atol=0)
        assert anchor.grad is not None and not anchor.grad.any()

    def test_cp_allowlist_and_vision_ulysses_split(self, monkeypatch):
        import veomni.models.auto as auto_module
        from veomni.models.auto import CONTEXT_PARALLEL_MODEL_TYPES, check_context_parallel_supported
        from veomni.models.seed_omni.modules.deepseek_v41.convert_model import MODULES

        cp_modules = {"deepseek_v41_llm", "deepseek_v41_mtp", "deepseek_v41_ngram", "deepseek_v41_text_encoder"}
        assert set(MODULES) == cp_modules | {"deepseek_v41_vision"}
        assert cp_modules <= CONTEXT_PARALLEL_MODEL_TYPES
        assert "deepseek_v41_vision" not in CONTEXT_PARALLEL_MODEL_TYPES

        state = type("PS", (), {"cp_enabled": False, "cp_size": 1, "ulysses_enabled": True, "ulysses_size": 4})()
        monkeypatch.setattr(auto_module, "is_parallel_state_initialized", lambda: True)
        monkeypatch.setattr(auto_module, "get_parallel_state", lambda: state)
        check_context_parallel_supported(DeepseekV41VisionEncoderConfig())

        state.cp_enabled, state.cp_size, state.ulysses_enabled, state.ulysses_size = True, 4, False, 1
        with pytest.raises(NotImplementedError, match="deepseek_v41_vision"):
            check_context_parallel_supported(DeepseekV41VisionEncoderConfig())


class TestShippedRecipes:
    @pytest.mark.parametrize(
        "recipe_rel",
        [
            "bf16/base.yaml",
            "qat/base.yaml",
            "deepgemm/base.yaml",
            "deepgemm/no_ngram_mtp.yaml",
        ],
    )
    def test_shipped_recipe_resolves_uniform_dp_and_valid_ops(self, monkeypatch, recipe_rel):
        monkeypatch.setenv("WORLD_SIZE", "128")
        recipe = yaml.safe_load((CONFIG_ROOT / recipe_rel).read_text())
        modules_path = REPO_ROOT / recipe["model"]["model_config"]["modules"]
        graph_path = REPO_ROOT / recipe["model"]["model_config"]["train_graph"]
        data_path = REPO_ROOT / recipe["data"]["train_path"]
        assert modules_path.is_file() and graph_path.is_file() and data_path.is_file()

        raw_modules = yaml.safe_load(modules_path.read_text())
        edges = yaml.safe_load(graph_path.read_text())
        graph_nodes = {edge[k].split(".")[0] for edge in edges for k in ("from", "to") if edge[k] != "end"}
        assert set(raw_modules) == graph_nodes

        global_args = OmniModuleRuntimeArguments(
            model_path="/tmp/deepseek_v41",
            accelerator=AcceleratorConfig(cp_size=recipe["model"]["accelerator"]["cp_size"]),
        )
        resolved = build_omni_module_runtime_args(global_args, "/tmp/deepseek_v41", raw_modules)
        _validate_omni_dp_size(global_args.accelerator, resolved)
        assert global_args.accelerator.dp_size == 8
        vision_acc = resolved["deepseek_v41_vision"].accelerator
        assert (vision_acc.ulysses_size, vision_acc.cp_size, vision_acc.dp_size) == (16, 1, 8)

        for mod_name in ("deepseek_v41_llm", "deepseek_v41_mtp"):
            if mod_name not in raw_modules:
                continue
            ops = raw_modules[mod_name]["ops_implementation"]
            sel = OpsSelection.from_config_value(ops["moe_implementation"])
            if recipe_rel.startswith("deepgemm/"):
                assert sel.name == "deepgemm" and sel.kwargs == {"dtype": "fp8"}


_GRIDS = [[3, 4], [5, 3]]
_OUT_HIDDEN = 32


def _tiny_vision_sp_config() -> DeepseekV41VisionEncoderConfig:
    return DeepseekV41VisionEncoderConfig(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        patch_size=2,
        downsample_ratio=2,
        max_image_tokens=64,
        min_pixels=16,
        out_hidden_size=_OUT_HIDDEN,
        ops_implementation={"attn_implementation": "sdpa"},
    )


def _vision_sp_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        from veomni.models.seed_omni.modules.deepseek_v41.vision import modeling as vision_modeling

        vision_modeling.USE_FLASH_ATTN_VARLEN = False
        config = _tiny_vision_sp_config()
        torch.manual_seed(0)
        model = DeepseekV41VisionEncoder(config).float().train()
        with torch.no_grad():
            model.image_start.copy_(torch.linspace(-0.2, 0.2, _OUT_HIDDEN))
            model.image_end.copy_(torch.linspace(0.1, -0.3, _OUT_HIDDEN))
            model.image_newline.copy_(torch.linspace(-0.1, 0.3, _OUT_HIDDEN))

        gen = torch.Generator().manual_seed(7)
        total_patches = sum(h * w for h, w in _GRIDS)
        pixel_values = torch.randn(total_patches, 3, config.patch_size, config.patch_size, generator=gen)
        token_types = [
            image_token_types(math.ceil(h / config.downsample_ratio), math.ceil(w / config.downsample_ratio))
            for h, w in _GRIDS
        ]
        weights = torch.randn(sum(int(t.numel()) for t in token_types), _OUT_HIDDEN, generator=gen)

        _init_parallel_state(dp_size=world_size, ulysses_size=1, device_type="cpu", name=None)
        ref_pixels = pixel_values.clone().requires_grad_(True)
        ref_meta = build_deepseek_v41_vit_metadata(_GRIDS, config.rope_dim, config.rope_theta, ref_pixels.device)
        ref_out = model(pixel_values=ref_pixels, grids=_GRIDS, token_types=token_types, vit_metadata=ref_meta)[
            "image_embeds"
        ]
        (ref_out * weights).sum().backward()
        ref_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        clear_parallel_state()

        _init_parallel_state(dp_size=1, ulysses_size=world_size, device_type="cpu", name=None)
        sp_pixels = pixel_values.clone().requires_grad_(True)
        sp_meta = build_deepseek_v41_vit_metadata(_GRIDS, config.rope_dim, config.rope_theta, sp_pixels.device)
        sp_out = model(
            pixel_values=sp_pad_and_slice(sp_pixels, dim=0, pad_value=0),
            grids=_GRIDS,
            token_types=token_types,
            vit_metadata=sp_meta,
        )["image_embeds"]
        (sp_out * weights).sum().backward()

        torch.testing.assert_close(sp_out, ref_out, rtol=1e-5, atol=1e-5)
        for name, p in model.named_parameters():
            assert p.grad is not None
            grad = p.grad.clone()
            dist.all_reduce(grad)
            torch.testing.assert_close(grad / world_size, ref_grads[name], rtol=1e-4, atol=1e-5)
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo required")
def test_vision_ulysses_sp_forward_and_gradients_match_unsharded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_vision_sp_worker, args=(2, os.path.join(tmp, "init")), nprocs=2, join=True)
