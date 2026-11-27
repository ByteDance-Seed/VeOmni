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

"""GLM-5.3-Flash registry, EP plan, and Hugging Face text parity tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel as HFGlm5NextTextModel

from tests.models.compare import assert_outputs_and_grads_match, eager_ops_config
from tests.models.tiny_configs import tiny_glm5_next_config as _tiny_mm_config
from tests.models.tiny_configs import tiny_glm5_next_text_config as _tiny_config
from veomni.models.auto import build_foundation_model
from veomni.models.transformers.glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextTextModel,
)


def _glm5_cls(architecture: str):
    if architecture == "Glm5NextForConditionalGeneration":
        return Glm5NextForConditionalGeneration
    return Glm5NextTextModel


def _build_ours(config, ops: SimpleNamespace | None = None, architecture: str = "Glm5NextTextModel"):
    config._set_ops_implementation(ops if ops is not None else eager_ops_config())
    return _glm5_cls(architecture)(config)


def test_glm5_next_text_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFGlm5NextTextModel(config)
    ours = _build_ours(config)
    assert hasattr(ours.layers[0].mlp, "veomni_gated_mlp")
    assert hasattr(ours.layers[2].mlp.experts, "veomni_moe")
    assert ours.layers[1].self_attn.next_skip_topk
    assert ours.layers[2].self_attn.skip_topk
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    gradient_weights = torch.randn(2, 8, config.hidden_size)
    assert_outputs_and_grads_match(
        hf,
        ours,
        lambda model: model(input_ids=input_ids, use_cache=False).last_hidden_state * gradient_weights,
        grad_atol=1e-4,
        grad_rtol=1e-4,
    )


def test_glm5_next_text_ep_plan_shards_routed_experts():
    config = _tiny_config()
    ours = _build_ours(config)
    plan = ours.get_parallel_plan()
    ep = plan.extra_parallel_plan["ep"]
    assert "layers.*.mlp.experts.gate_up_proj" in ep
    assert "layers.*.mlp.experts.down_proj" in ep


def test_glm5_next_forcg_ep_plan_shards_routed_experts():
    config = _tiny_mm_config()
    ours = _build_ours(config, architecture="Glm5NextForConditionalGeneration")
    plan = ours.get_parallel_plan()
    ep = plan.extra_parallel_plan["ep"]
    assert "model.language_model.layers.*.mlp.experts.gate_up_proj" in ep
    assert "model.language_model.layers.*.mlp.experts.down_proj" in ep


def test_glm5_next_registry_builds_text_and_conditional():
    text = build_foundation_model(
        config_path=_tiny_config(),
        weights_path=None,
        init_device="cpu",
        torch_dtype="float32",
        ops_implementation=eager_ops_config(),
    )
    assert text.__class__.__name__ == "Glm5NextTextModel"

    cg = build_foundation_model(
        config_path=_tiny_mm_config(),
        weights_path=None,
        init_device="cpu",
        torch_dtype="float32",
        ops_implementation=eager_ops_config(),
    )
    assert cg.__class__.__name__ == "Glm5NextForConditionalGeneration"
    assert hasattr(cg, "veomni_ce")
    assert hasattr(cg.model.language_model.layers[2].mlp.experts, "veomni_moe")


def test_glm5_next_vlm_hooks_and_vision_ops():
    config = _tiny_mm_config()
    ours = _build_ours(config, architecture="Glm5NextForConditionalGeneration")
    assert callable(ours.get_position_id_func())
    assert callable(ours.get_metadata_collate_func())
    assert hasattr(ours.model.visual, "dummy_forward")
    vision_attn = ours.model.visual.blocks[0].attn
    assert hasattr(vision_attn, "veomni_attn")
    assert hasattr(vision_attn.q_norm, "veomni_rms_norm")
    pos = ours.get_position_id_func()(input_ids=torch.ones(1, 8, dtype=torch.long))
    assert pos["position_ids"].shape == (1, 8)
    batch = {"image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long)}
    ours.get_metadata_collate_func()(batch, {"pixel_values": 0, "pixel_values_videos": 0})
    assert "multimodal_metadata" in batch
    assert batch["multimodal_metadata"]["vit_image_max_seqlen"] == 16


def test_glm5_next_shared_dsa_and_attn_ops_sync():
    config = _tiny_config()
    ours = _build_ours(config)
    assert config._attn_implementation == config.ops_config.attn_implementation
    assert ours.layers[1].block_type == "deepseek_sparse_attention"
    assert ours.layers[2].block_type == "deepseek_sparse_attention"
    assert ours.layers[1].self_attn.indexer is not None
    assert ours.layers[2].self_attn.indexer is None
    assert ours.layers[1].self_attn.next_skip_topk
    assert not any(p.requires_grad for p in ours.layers[1].self_attn.indexer.parameters())


def test_glm5_next_rejects_packed_multi_sample():
    config = _tiny_config()
    ours = _build_ours(config)
    input_ids = torch.randint(3, config.vocab_size, (1, 8))
    try:
        ours(
            input_ids=input_ids,
            use_cache=False,
            cu_seq_lens_q=torch.tensor([0, 3, 8], dtype=torch.long),
        )
    except ValueError as exc:
        assert "packed multi-sample" in str(exc)
    else:
        raise AssertionError("expected packed multi-sample rejection")


def test_glm5_next_rejects_orphan_shared_indexer():
    config = _tiny_config()
    config.layer_types = [
        "linear_attention",
        "deepseek_sparse_attention",
        "linear_attention",
        "deepseek_sparse_attention",
    ]
    config.indexer_types = ["full", "full", "full", "shared"]
    try:
        _build_ours(config)
    except ValueError as exc:
        assert "shared requires the previous layer" in str(exc)
    else:
        raise AssertionError("expected orphan shared indexer rejection")
