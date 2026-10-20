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
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from veomni.models.transformers.gemma3.generated import patched_modeling_gemma3_npu as gemma3_modeling


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
