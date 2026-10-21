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

from types import SimpleNamespace

import pytest
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config, Gemma3TextConfig
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from tests.models.compare import eager_ops_config, ops_config_scope, stamp_attn_implementation
from tests.models.tiny_configs import tiny_gemma3_text_config
from veomni.models.transformers.gemma3.generated import patched_modeling_gemma3_npu as gemma3_modeling
from veomni.ops import VeomniOp


def _build_tiny_gemma3_vlm():
    text_config = tiny_gemma3_text_config()
    vision_config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=8,
        patch_size=4,
    )
    config = Gemma3Config(
        text_config=text_config,
        vision_config=vision_config,
        mm_tokens_per_image=4,
        image_token_index=120,
        tie_word_embeddings=False,
    )
    ops = eager_ops_config()
    stamp_attn_implementation(config, ops.attn_implementation)
    with ops_config_scope(ops):
        return gemma3_modeling.Gemma3ForConditionalGeneration(config)


def test_gemma3_vlm_binds_instance_local_ops_and_runs_text_loss():
    model = _build_tiny_gemma3_vlm()

    assert isinstance(model.veomni_ce, VeomniOp)
    assert model.veomni_ce.impl == "eager"
    assert model.model.language_model.layers[0].input_layernorm.veomni_rms_norm.variant == "offset"
    assert not hasattr(gemma3_modeling, "veomni_causal_lm_loss")

    input_ids = torch.tensor([[2, 5, 6, 7]])
    output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    assert torch.isfinite(output.loss)


def test_gemma3_vlm_text_only_distributed_batch_touches_visual_parameters(monkeypatch):
    model = _build_tiny_gemma3_vlm().eval()
    parallel_state = SimpleNamespace(fsdp_enabled=False)
    monkeypatch.setattr(gemma3_modeling, "get_parallel_state", lambda: parallel_state)
    input_ids = torch.tensor([[2, 5, 6, 7]])

    with torch.no_grad():
        baseline_logits = model(input_ids=input_ids, use_cache=False).logits
        parallel_state.fsdp_enabled = True
        distributed_logits = model(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(distributed_logits, baseline_logits)

    output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    output.loss.backward()
    visual_parameters = list(model.model.vision_tower.parameters()) + list(
        model.model.multi_modal_projector.parameters()
    )
    assert visual_parameters
    assert all(parameter.grad is not None for parameter in visual_parameters)
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in visual_parameters)


def test_gemma3_vlm_sdpa_mask_preserves_packed_boundaries():
    config = Gemma3TextConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
        sliding_window=3,
    )
    config._attn_implementation = "sdpa"

    masks = gemma3_modeling.create_gemma3_masks(
        config=config,
        inputs_embeds=torch.zeros(1, 5, 8),
        attention_mask=torch.ones(1, 5),
        past_key_values=None,
        position_ids=None,
        token_type_ids=torch.tensor([[0, 1, 1, 0, 0]]),
        cu_seq_lens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
        attention_impl="sdpa",
    )

    for mask in masks.values():
        assert mask[0, 0, 2, 0]
        assert mask[0, 0, 1, 2]  # Image tokens remain bidirectional within their packed sample.
        assert not mask[0, 0, 3, 2]
        assert mask[0, 0, 4, 3]


def test_gemma3_vlm_rejects_sequence_parallelism(monkeypatch):
    monkeypatch.setattr(
        gemma3_modeling,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_enabled=True),
    )

    with pytest.raises(ValueError, match="does not support sequence parallelism"):
        gemma3_modeling.Gemma3ForConditionalGeneration.get_extra_collate_infos(None)
