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

"""Qwen3 loss integration and Hugging Face task-head parity tests.

Direct-import the generated classes. Compare a toy model against Hugging Face.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFQwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForTokenClassification as HFQwen3ForTokenClassification
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model as HFQwen3Model

from tests.models.compare import assert_sequence_classification_matches_hf, eager_ops_config, ops_config_scope
from tests.models.tiny_configs import tiny_qwen3_config as _tiny_config
from tests.ops.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from tests.tools.training_utils import make_eager_ops_config
from veomni.data.data_collator import MainCollator
from veomni.models import build_foundation_model
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


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
    with ops_config_scope(ops if ops is not None else eager_ops_config()):
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
    assert torch.equal(ours_logits, hf_logits)

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
    with ops_config_scope(eager_ops_config()):
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
    with ops_config_scope(eager_ops_config()):
        model = seq_cls(_tiny_config(num_labels=4))
    assert_sequence_classification_matches_hf(model, supervision=supervision)


def test_qwen3_token_cls_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config(num_labels=4)
    hf = HFQwen3ForTokenClassification(config)
    with ops_config_scope(eager_ops_config()):
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


@pytest.mark.parametrize("impl", ("eager", "sdpa", "veomni_sdpa"))
def test_qwen3_packed_dense_matches_independent_samples(impl):
    """Packed ``cu_seq_lens_q`` isolates samples under dense attention backends."""
    import copy

    torch.manual_seed(123)
    config = _tiny_config()
    config._attn_implementation = impl
    ops = eager_ops_config()
    ops.attn_implementation = impl
    ours = _build_qwen3(config, ops).eval()
    ours_sep = copy.deepcopy(ours).eval()

    first_input_ids = torch.tensor([[5, 6, 7]])
    second_input_ids = torch.tensor([[8, 9, 10, 11, 12]])
    packed_input_ids = torch.cat((first_input_ids, second_input_ids), dim=1)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    packed_logits = ours(
        input_ids=packed_input_ids,
        attention_mask=torch.ones_like(packed_input_ids),
        position_ids=torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]]),
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        use_cache=False,
    ).logits
    first_logits = ours_sep(
        input_ids=first_input_ids,
        attention_mask=torch.ones_like(first_input_ids),
        position_ids=torch.arange(3).unsqueeze(0),
        cu_seq_lens_q=torch.tensor([0, 3], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 3], dtype=torch.int32),
        use_cache=False,
    ).logits
    second_logits = ours_sep(
        input_ids=second_input_ids,
        attention_mask=torch.ones_like(second_input_ids),
        position_ids=torch.arange(5).unsqueeze(0),
        cu_seq_lens_q=torch.tensor([0, 5], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 5], dtype=torch.int32),
        use_cache=False,
    ).logits

    torch.testing.assert_close(packed_logits[:, :3], first_logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(packed_logits[:, 3:], second_logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    w1 = torch.randn_like(first_logits)
    w2 = torch.randn_like(second_logits)
    ((packed_logits[:, :3] * w1).sum() + (packed_logits[:, 3:] * w2).sum()).backward()
    ((first_logits * w1).sum() + (second_logits * w2).sum()).backward()

    for (name, p_packed), (_, p_sep) in zip(ours.named_parameters(), ours_sep.named_parameters(), strict=True):
        if p_sep.grad is None:
            assert p_packed.grad is None, name
            continue
        torch.testing.assert_close(p_packed.grad, p_sep.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL, msg=name)
