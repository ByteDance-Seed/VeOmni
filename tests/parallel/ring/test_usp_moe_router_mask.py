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

"""Qwen3-VL-MoE aux loss under USP: the padding mask must follow the router-logit rows.

The collator pads ``attention_mask`` to the full sequence without slicing it, while each
cp rank routes only its zig-zag shard. A tiny model runs one cp rank's shard on CPU with
``output_router_logits=True`` and checks the aux loss uses exactly that rank's mask rows.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3VLMoeConfig

from veomni.models.auto import build_foundation_model
from veomni.models.transformers.qwen3_vl_moe.generated import patched_modeling_qwen3_vl_moe_gpu as modeling
from veomni.ops.kernels.load_balancing_loss.eager import load_balancing_loss_pytorch

from ...tools.training_utils import make_eager_ops_config


SEQ_LEN = 16
CP_SIZE = 2


def _tiny_config() -> Qwen3VLMoeConfig:
    return Qwen3VLMoeConfig(
        architectures=["Qwen3VLMoeForConditionalGeneration"],
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [0],
        },
    )


def _cp_state(cp_rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        sp_enabled=True,
        sp_size=CP_SIZE,
        sp_rank=cp_rank,
        cp_enabled=True,
        cp_size=CP_SIZE,
        cp_rank=cp_rank,
        ulysses_enabled=False,
        ulysses_size=1,
        ulysses_rank=-1,
        fsdp_enabled=False,
        async_enabled=False,
    )


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_router_aux_loss_uses_this_ranks_zigzag_mask(monkeypatch, cp_rank):
    torch.manual_seed(0)
    model = build_foundation_model(
        config_path=_tiny_config(),
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=make_eager_ops_config(),
    ).eval()
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: _cp_state(cp_rank))

    # Zig-zag shard of one dense sequence: cp rank r owns blocks r and 2*cp-1-r.
    block = SEQ_LEN // (2 * CP_SIZE)
    blocks = (cp_rank, 2 * CP_SIZE - 1 - cp_rank)
    rows = torch.cat([torch.arange(b * block, (b + 1) * block) for b in blocks])

    full_ids = torch.randint(0, 64, (1, SEQ_LEN))
    full_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)
    full_mask[:, -3:] = 0  # padding tail, so the slice decides which rows count
    no_visual = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    position_ids = rows.view(1, 1, -1).expand(3, 1, -1)

    with torch.no_grad():
        out = model(
            input_ids=full_ids[:, rows],
            attention_mask=full_mask,
            position_ids=position_ids,
            image_mask=no_visual,
            video_mask=no_visual,
            output_router_logits=True,
        )

    assert all(logits.shape[0] == rows.numel() for logits in out.router_logits)
    text_config = model.config.text_config
    expected = load_balancing_loss_pytorch(
        out.router_logits, text_config.num_experts, text_config.num_experts_per_tok, full_mask[:, rows]
    )
    torch.testing.assert_close(out.aux_loss, expected)


def test_ulysses_only_router_mask_is_a_contiguous_chunk(monkeypatch):
    state = SimpleNamespace(sp_enabled=True, sp_size=2, sp_rank=1, cp_enabled=False)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: state)
    mask = torch.arange(SEQ_LEN).view(1, -1)
    torch.testing.assert_close(modeling._sp_local_router_mask(mask, None), mask[:, SEQ_LEN // 2 :])
