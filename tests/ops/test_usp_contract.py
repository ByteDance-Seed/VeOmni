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

"""USP must preserve existing CP layouts and reject unsupported objectives."""

from types import SimpleNamespace

import pytest
import torch

from veomni.arguments import AcceleratorConfig
from veomni.models import auto
from veomni.ops.kernels.attention import flash
from veomni.trainer.text_dpo_trainer import TextDPOTrainer


def test_usp_layout_is_explicit(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(NotImplementedError, match="cannot be combined"):
        AcceleratorConfig(cp_size=2, ulysses_size=2)
    config = AcceleratorConfig(cp_size=2, ulysses_size=2, cp_layout="zigzag")
    assert config.dp_size == 1


@pytest.mark.parametrize(
    "model_type,layout,backend,allowed",
    [
        ("deepseek_v4", "contiguous", "eager", True),
        ("deepseek_v4", "zigzag", "veomni_flash_attention_2_with_sp", False),
        ("qwen3", "contiguous", "veomni_flash_attention_2_with_sp", False),
        ("qwen3", "zigzag", "veomni_flash_attention_2_with_sp", True),
        ("qwen3", "zigzag", "veomni_flash_attention_4_with_sp", True),
        ("qwen3", "zigzag", "veomni_flex_attention_with_sp", False),
        ("qwen3_next", "zigzag", "veomni_flash_attention_2_with_sp", False),
    ],
)
def test_cp_model_and_backend_gate(monkeypatch, model_type, layout, backend, allowed):
    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "is_torch_npu_available", lambda: False)
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_layout=layout))
    config = SimpleNamespace(model_type=model_type, _attn_implementation=backend)
    if allowed:
        auto.check_context_parallel_supported(config)
    else:
        with pytest.raises(NotImplementedError):
            auto.check_context_parallel_supported(config)


@pytest.mark.parametrize(
    "modifier",
    [
        {"dropout": 0.1},
        {"sliding_window": 4},
        {"softcap": 1.0},
        {"s_aux": torch.ones(2)},
    ],
)
def test_usp_rejects_attention_modifiers_before_collectives(monkeypatch, modifier):
    monkeypatch.setattr(
        flash, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_layout="zigzag", ulysses_enabled=True)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported attention entered an Ulysses collective")

    monkeypatch.setattr(flash, "prepare_ulysses_qkv", forbidden)
    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="veomni_flash_attention_2_with_sp"), is_causal=True
    )
    q = torch.zeros(1, 2, 8, 4, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="does not support"):
        flash.flash_attention_forward(module, q, q, q, None, **modifier)


def test_dpo_rejects_zigzag_before_setup():
    args = SimpleNamespace(model=SimpleNamespace(accelerator=SimpleNamespace(cp_size=2, cp_layout="zigzag")))
    with pytest.raises(NotImplementedError, match="DPO"):
        TextDPOTrainer(args)


@pytest.mark.parametrize("extra_outputs", [(), (None, None)])
def test_fa4_forward_accepts_auxiliary_outputs(monkeypatch, extra_outputs):
    from veomni.distributed.sequence_parallel import ring_attention as ring

    q = torch.zeros(1, 8, 2, 4)
    lse = torch.zeros(1, 2, 8)
    monkeypatch.setattr(ring, "FA_BACKEND", "fa4")
    monkeypatch.setattr(ring, "_fa4_fwd", lambda *args, **kwargs: (q, lse, *extra_outputs), raising=False)
    out, actual_lse = ring._fa_forward(q, q, q, 0.5, True)
    assert out is q and actual_lse is lse
    cu = torch.tensor([0, 8], dtype=torch.int32)
    out, actual_lse = ring._fa_varlen_forward(q, q, q, cu, cu, 8, 8, 0.5, True)
    assert out is q and actual_lse is lse


def test_rl_rejects_zigzag_for_composed_trainers(monkeypatch):
    from veomni.trainer import base_rl_trainer

    monkeypatch.setattr(
        base_rl_trainer, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_layout="zigzag")
    )
    trainer = base_rl_trainer.BaseRLTrainer.__new__(base_rl_trainer.BaseRLTrainer)
    with pytest.raises(NotImplementedError, match="RL postprocessing"):
        trainer._build_preforward_postforward()
