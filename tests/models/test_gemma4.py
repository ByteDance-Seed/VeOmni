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

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration as HFGemma4

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.models.transformers.gemma4.generated.patched_modeling_gemma4_gpu import (
    Gemma4ForConditionalGeneration as VeOmniGemma4,
)
from veomni.ops.kernels import attention as veomni_attention


_TOY_CONFIG = Path(__file__).parents[1] / "toy_config" / "gemma4_toy"
_FLEX_IMPLEMENTATION = "veomni_flex_attention_with_sp"


def _eager_ops_config(attn_implementation: str) -> OpsImplementationConfig:
    return OpsImplementationConfig(
        attn_implementation=attn_implementation,
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        rotary_pos_emb_vision_implementation="eager",
        load_balancing_loss_implementation="eager",
    )


def _toy_config() -> Gemma4Config:
    return Gemma4Config.from_pretrained(_TOY_CONFIG)


def test_gemma4_state_dict_matches_huggingface():
    config = _toy_config()
    with torch.device("meta"):
        hf_model = HFGemma4(config)
        veomni_model = VeOmniGemma4(config)

    hf_state = {name: tuple(tensor.shape) for name, tensor in hf_model.state_dict().items()}
    veomni_state = {name: tuple(tensor.shape) for name, tensor in veomni_model.state_dict().items()}
    assert veomni_state == hf_state


def test_gemma4_eager_training(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torch.manual_seed(17)
    model = build_foundation_model(
        _TOY_CONFIG,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_eager_ops_config("eager"),
    )
    input_ids = torch.randint(3, model.config.text_config.vocab_size, (1, 8))

    output = model(input_ids=input_ids, labels=input_ids.clone(), logits_to_keep=1, use_cache=False)
    output.loss.backward()

    assert output.logits.shape == (1, 8, model.config.text_config.vocab_size)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(model.model.language_model.layers[0].self_attn.q_proj.weight.grad).all()


def test_gemma4_text_eager_training_ignores_logits_to_keep_with_labels(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    config = _toy_config().text_config
    config.save_pretrained(tmp_path)
    model = build_foundation_model(
        tmp_path,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_eager_ops_config("eager"),
    )
    input_ids = torch.randint(3, config.vocab_size, (1, 8))

    output = model(input_ids=input_ids, labels=input_ids.clone(), logits_to_keep=1, use_cache=False)
    output.loss.backward()

    assert output.logits.shape == (1, 8, config.vocab_size)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(model.model.layers[0].self_attn.q_proj.weight.grad).all()


@pytest.mark.parametrize("text_only", [False, True], ids=["multimodal", "text"])
@pytest.mark.parametrize("loss_implementation", ["eager", "chunk_loss"])
@pytest.mark.parametrize("softcap", [None, 30.0], ids=["uncapped", "softcapped"])
def test_gemma4_training_loss_matches_capped_logits(monkeypatch, tmp_path, text_only, loss_implementation, softcap):
    """A non-eager loss selection must preserve softcapped loss and gradients."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torch.manual_seed(23)
    config = _toy_config()
    config.text_config.final_logit_softcapping = softcap
    if text_only:
        config = config.text_config
    config.save_pretrained(tmp_path)
    ops_config = _eager_ops_config("eager")
    # chunk_loss exercises real non-eager dispatch on CPU without CUDA mocks.
    ops_config.cross_entropy_loss_implementation = loss_implementation
    model = build_foundation_model(tmp_path, torch_dtype="float32", init_device="cpu", ops_implementation=ops_config)
    with torch.no_grad():
        model.lm_head.weight.mul_(100)
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    labels = input_ids.clone()
    labels[:, 3] = -100
    backbone = model.model if text_only else model.model.language_model
    hidden_states = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    raw_logits = model.lm_head(hidden_states)
    expected_logits = raw_logits if softcap is None else torch.tanh(raw_logits / softcap) * softcap
    expected_loss = F.cross_entropy(
        expected_logits[:, :-1].reshape(-1, expected_logits.shape[-1]), labels[:, 1:].reshape(-1)
    )
    parameters = (model.lm_head.weight, backbone.layers[0].self_attn.q_proj.weight)
    expected_gradients = torch.autograd.grad(expected_loss, parameters)

    output = model(input_ids=input_ids, labels=labels, use_cache=False)
    output.loss.backward()

    torch.testing.assert_close(output.loss, expected_loss)
    for parameter, expected_gradient in zip(parameters, expected_gradients):
        torch.testing.assert_close(parameter.grad, expected_gradient, atol=1e-5, rtol=1e-4)
    if softcap is not None:
        assert not torch.allclose(expected_logits, raw_logits)
        torch.testing.assert_close(output.logits, expected_logits)
    elif loss_implementation == "chunk_loss":
        assert output.logits is None


@pytest.mark.parametrize("has_features,has_mask", [(False, False), (True, False), (False, True)])
def test_gemma4_incomplete_audio_inputs_return_no_audio_states(monkeypatch, has_features, has_mask):
    """Audio states are absent when either encoder input is missing."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    model = build_foundation_model(
        _TOY_CONFIG,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_eager_ops_config("eager"),
    ).eval()
    input_ids = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False)
        output = model(
            input_ids=input_ids,
            input_features=torch.randn(1, 16, 128) if has_features else None,
            input_features_mask=torch.ones(1, 16, dtype=torch.bool) if has_mask else None,
            use_cache=False,
        )

    assert output.audio_hidden_states is None
    torch.testing.assert_close(output.logits, expected.logits)


def test_gemma4_builds_pack_aware_full_and_sliding_masks(monkeypatch):
    captured = []

    def fake_flex_adapter(module, query, key, value, attention_mask, **kwargs):
        captured.append((module.layer_type, attention_mask))
        return torch.zeros_like(query).transpose(1, 2), None

    monkeypatch.setitem(
        veomni_attention._ATTENTION_FORWARD_DISPATCH,
        _FLEX_IMPLEMENTATION,
        fake_flex_adapter,
    )
    config = _toy_config()
    model = VeOmniGemma4._from_config(config, attn_implementation=_FLEX_IMPLEMENTATION).eval()
    input_ids = torch.randint(3, config.text_config.vocab_size, (1, 8))
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    cu_seq_lens_q = torch.tensor([0, 3, 8], dtype=torch.int32)

    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=position_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            use_cache=False,
        )

    assert [layer_type for layer_type, _ in captured] == ["sliding_attention", "full_attention"]
    zero = torch.tensor(0)
    first_token_in_second_pack = torch.tensor(3)
    last_token_in_first_pack = torch.tensor(2)
    for _, block_mask in captured:
        assert isinstance(block_mask, BlockMask)
        assert not block_mask.mask_mod(zero, zero, first_token_in_second_pack, last_token_in_first_pack)


def test_gemma4_multimodal_masks_preserve_packed_boundaries(monkeypatch):
    captured = {}

    def fake_flex_adapter(module, query, key, value, attention_mask, **kwargs):
        captured[module.layer_type] = attention_mask
        return torch.zeros_like(query).transpose(1, 2), None

    monkeypatch.setitem(
        veomni_attention._ATTENTION_FORWARD_DISPATCH,
        _FLEX_IMPLEMENTATION,
        fake_flex_adapter,
    )
    config = _toy_config()
    config.text_config.use_bidirectional_attention = "vision"
    model = VeOmniGemma4._from_config(config, attn_implementation=_FLEX_IMPLEMENTATION).eval()
    input_ids = torch.randint(3, config.text_config.vocab_size, (1, 8))
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    mm_token_type_ids = torch.tensor([[0, 1, 1, 0, 0, 1, 1, 0]])
    cu_seq_lens_q = torch.tensor([0, 4, 8], dtype=torch.int32)

    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            use_cache=False,
        )

    zero = torch.tensor(0)
    for block_mask in captured.values():
        assert isinstance(block_mask, BlockMask)
        assert not block_mask.mask_mod(zero, zero, torch.tensor(5), torch.tensor(1))
    assert captured["sliding_attention"].mask_mod(zero, zero, torch.tensor(1), torch.tensor(2))
