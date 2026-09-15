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

"""Qwen3 models consume tests.

Direct-import the generated classes. Compare a toy model against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFQwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForTokenClassification as HFQwen3ForTokenClassification
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model as HFQwen3Model

from tests.models.compare import assert_sequence_classification_matches_hf, ops_config_scope
from tests.models.tiny_configs import tiny_qwen3_config as _tiny_config
from tests.ops.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from tests.tools.training_utils import make_eager_ops_config
from veomni.data.data_collator import MainCollator
from veomni.models import build_foundation_model
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


def _eager_kernels_config() -> SimpleNamespace:
    return SimpleNamespace(
        attn_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        rotary_pos_emb_implementation="eager",
        swiglu_mlp_implementation="eager",
    )


def _qwen3_classes():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_npu import (
            Qwen3ForCausalLM,
            Qwen3ForSequenceClassification,
            Qwen3ForTokenClassification,
            Qwen3Model,
        )
    else:
        from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import (
            Qwen3ForCausalLM,
            Qwen3ForSequenceClassification,
            Qwen3ForTokenClassification,
            Qwen3Model,
        )
    return Qwen3ForCausalLM, Qwen3ForSequenceClassification, Qwen3ForTokenClassification, Qwen3Model


def _build_qwen3(config: Qwen3Config, ops: SimpleNamespace | None = None):
    with ops_config_scope(ops if ops is not None else _eager_kernels_config()):
        causal_cls, _, _, _ = _qwen3_classes()
        return causal_cls(config)


def _named_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def test_qwen3_eager_matches_hf_logits_and_loss():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3ForCausalLM(config)
    ours = _build_qwen3(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    hf_logits = hf(input_ids=input_ids, use_cache=False).logits
    ours_logits = ours(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(ours_logits, hf_logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    assert ours_out.logits is None

    hf_out.loss.backward()
    ours_out.loss.backward()
    hf_grads = _named_trainable(hf)
    ours_grads = _named_trainable(ours)
    assert hf_grads.keys() == ours_grads.keys()
    for name, param in hf_grads.items():
        assert param.grad is not None, name
        assert ours_grads[name].grad is not None, name
        torch.testing.assert_close(
            ours_grads[name].grad, param.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL, msg=name
        )


def test_qwen3_base_model_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3Model(config)
    with ops_config_scope(_eager_kernels_config()):
        *_, model_cls = _qwen3_classes()
        ours = model_cls(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    hf_out = hf(input_ids=input_ids, use_cache=False)
    ours_out = ours(input_ids=input_ids, use_cache=False)
    torch.testing.assert_close(ours_out.last_hidden_state, hf_out.last_hidden_state, atol=EAGER_ATOL, rtol=EAGER_RTOL)


@pytest.mark.parametrize("supervision", ["last-valid", "selected-tokens"])
def test_qwen3_seq_cls_matches_hf(supervision):
    torch.manual_seed(0)
    _, seq_cls, _, _ = _qwen3_classes()
    with ops_config_scope(_eager_kernels_config()):
        model = seq_cls(_tiny_config(num_labels=4))
    assert_sequence_classification_matches_hf(model, supervision=supervision)


def test_qwen3_token_cls_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config(num_labels=4)
    hf = HFQwen3ForTokenClassification(config)
    with ops_config_scope(_eager_kernels_config()):
        _, _, token_cls, _ = _qwen3_classes()
        ours = token_cls(config)
    ours.load_state_dict(hf.state_dict())
    hf.eval()
    ours.eval()

    input_ids = torch.randint(3, config.vocab_size, (2, 6))
    labels = torch.randint(0, config.num_labels, input_ids.shape)
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False)
    torch.testing.assert_close(ours_out.logits, hf_out.logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=EAGER_ATOL, rtol=EAGER_RTOL)


def test_qwen3_loss_matches_with_padded_packed_input(monkeypatch):
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for flash-attn")
    pytest.importorskip("flash_attn")

    monkeypatch.setattr(
        "veomni.data.data_collator.get_parallel_state",
        lambda: type("PS", (), {"sp_enabled": False, "sp_size": 1, "sp_rank": 0})(),
    )

    device = torch.device(get_device_type())
    torch.manual_seed(0)
    model = build_foundation_model(
        config_path="tests/toy_config/qwen3_toy",
        weights_path=None,
        torch_dtype="float16",
        init_device=get_device_type(),
        ops_implementation=make_eager_ops_config(attn_implementation="flash_attention_2"),
    ).eval()

    features = [
        {
            "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([11, 12, 13], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([21, 22], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "labels": torch.tensor([21, 22], dtype=torch.long),
        },
    ]

    unpadded = MainCollator()(features)
    padded = MainCollator(pad_to_length=16)(features)

    def to_device(batch):
        return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}

    def forward(batch):
        batch = to_device(batch)
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            position_ids=batch.get("position_ids"),
            cu_seq_lens_q=batch.get("cu_seq_lens_q"),
            cu_seq_lens_k=batch.get("cu_seq_lens_k"),
            max_length_q=batch.get("max_length_q"),
            max_length_k=batch.get("max_length_k"),
            labels=batch.get("labels"),
        )

    with torch.no_grad():
        unpadded_loss = forward(unpadded).loss
        padded_loss = forward(padded).loss

    torch.testing.assert_close(padded_loss, unpadded_loss, rtol=1e-3, atol=1e-3)
