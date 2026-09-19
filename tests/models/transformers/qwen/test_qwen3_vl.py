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
# See the License for the specific language governing limitations
# under the License.

"""Qwen3-VL text, image, and video Hugging Face parity tests.

Direct-import the generated classes. Compare a toy model against Hugging Face on
the text-only, image+text, and video+text paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration,
)

from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    ops_config_scope,
    qwen_image_inputs,
    qwen_video_inputs,
    stamp_attn_implementation,
)
from tests.models.tiny_configs import tiny_qwen3_vl_config as _tiny_config


IMAGE_TOKEN_ID = 120
VIDEO_TOKEN_ID = 121


def _qwen3_vl_cls():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_npu import (
            Qwen3VLForConditionalGeneration,
        )
    else:
        from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
            Qwen3VLForConditionalGeneration,
        )
    return Qwen3VLForConditionalGeneration


def _build_ours(config: Qwen3VLConfig, ops: SimpleNamespace | None = None):
    ops = ops if ops is not None else eager_ops_config()
    stamp_attn_implementation(config, ops.attn_implementation)
    with ops_config_scope(ops):
        return _qwen3_vl_cls()(config)


def test_qwen3_vl_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids, logits_equal=True)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_qwen3_vl_eager_matches_hf_vision_and_text(modality):
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 20))
    vision_inputs = (
        qwen_image_inputs(config, input_ids)
        if modality == "image"
        else qwen_video_inputs(config, input_ids, split_frames=True)
    )
    ids = vision_inputs.pop("input_ids")
    labels = vision_inputs.pop("labels")
    # Vision embed / RoPE ULP (~2e-7); the text-only path above is bitwise.
    assert_eager_matches_hf(hf, ours, input_ids=ids, labels=labels, fwd_kwargs=vision_inputs)


def test_qwen3_vl_async_text_skips_ulysses_but_vision_stays_sync():
    """Text async gathers outside attention; vision must keep sync Ulysses."""
    import inspect

    from veomni.models.transformers.qwen3_vl.generated import (
        patched_modeling_qwen3_vl_gpu,
        patched_modeling_qwen3_vl_npu,
    )
    from veomni.models.transformers.qwen3_vl.qwen3_vl_gpu_patch_gen_config import (
        _qwen3_vl_async_ulysses_attention_forward,
        qwen3_vl_vision_attention_forward_patched,
    )
    from veomni.models.transformers.qwen3_vl_moe.generated import (
        patched_modeling_qwen3_vl_moe_gpu,
        patched_modeling_qwen3_vl_moe_npu,
    )

    assert "skip_ulysses=True" in inspect.getsource(_qwen3_vl_async_ulysses_attention_forward)
    assert "rms_norm=self.q_norm.veomni_rms_norm" in inspect.getsource(_qwen3_vl_async_ulysses_attention_forward)
    assert "skip_ulysses=True" not in inspect.getsource(qwen3_vl_vision_attention_forward_patched)
    for module in (
        patched_modeling_qwen3_vl_gpu,
        patched_modeling_qwen3_vl_npu,
        patched_modeling_qwen3_vl_moe_gpu,
        patched_modeling_qwen3_vl_moe_npu,
    ):
        source = inspect.getsource(module._qwen3_vl_async_ulysses_attention_forward)
        assert "skip_ulysses=True" in source
        assert "rms_norm=self.q_norm.veomni_rms_norm" in source


def test_qwen3_vl_async_helper_forwards_skip_ulysses(monkeypatch):
    from veomni.models.transformers.qwen3_vl import qwen3_vl_gpu_patch_gen_config as cfg

    captured = {}

    class _FakeOp:
        def __init__(self, op, variant, impl=None):
            self.op = op

        def __call__(self, *args, **kwargs):
            if self.op == "async_ulysses_qkv":
                captured["rms_norm"] = kwargs.get("rms_norm")
                hidden = kwargs["hidden_states"]
                batch, seq, _ = hidden.shape
                qkv = hidden.new_zeros(batch, seq * 2, 2, 4)
                return qkv, qkv.clone(), qkv.clone()
            return kwargs["hidden_states"]

    def fake_attn(module, query, key, value, attention_mask, **kwargs):
        captured["skip_ulysses"] = kwargs.get("skip_ulysses")
        captured["query_seq"] = query.shape[2]
        return query.transpose(1, 2), None

    monkeypatch.setattr(cfg, "VeomniOp", _FakeOp)
    monkeypatch.setattr(cfg, "gather_outputs", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(cfg, "get_ulysses_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(cfg, "get_parallel_state", lambda: SimpleNamespace(sp_group=object()))
    monkeypatch.setattr(cfg, "is_flash_attention_requested", lambda _config: True)

    rms_handle = object()
    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="flash_attention_2", rms_norm_eps=1e-6),
        q_proj=SimpleNamespace(weight=None, bias=None),
        k_proj=SimpleNamespace(weight=None, bias=None),
        v_proj=SimpleNamespace(weight=None, bias=None),
        o_proj=SimpleNamespace(weight=None, bias=None),
        q_norm=SimpleNamespace(weight=None, veomni_rms_norm=rms_handle),
        k_norm=SimpleNamespace(weight=None),
        veomni_rope=lambda q, k, cos, sin: (q, k),
        veomni_attn=fake_attn,
        training=False,
        attention_dropout=0.0,
        scaling=1.0,
        head_dim=4,
    )
    hidden = torch.zeros(1, 4, 8)
    cos = torch.zeros(1, 4, 4)
    cfg._qwen3_vl_async_ulysses_attention_forward(module, hidden, None, (cos, cos))
    assert captured["skip_ulysses"] is True
    assert captured["query_seq"] == 8
    assert captured["rms_norm"] is rms_handle
