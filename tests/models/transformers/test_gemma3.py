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

"""Gemma 3 text models consume tests.

Direct-import the generated CausalLM. Compare a toy model against HuggingFace.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import BlockMask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM as HFGemma3ForCausalLM

from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    named_trainable,
    ops_config_scope,
)
from tests.models.tiny_configs import tiny_gemma3_text_config as _tiny_config
from tests.ops.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


def _build_ours(config: Gemma3TextConfig, ops: SimpleNamespace | None = None):
    from veomni.models.transformers.gemma3.generated.patched_modeling_gemma3_gpu import (
        Gemma3ForCausalLM,
    )

    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return Gemma3ForCausalLM(config)


def _flex_ops_config() -> SimpleNamespace:
    config = eager_ops_config()
    config.attn_implementation = "veomni_flex_attention"
    return config


def test_gemma3_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFGemma3ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_gemma3_eager_matches_hf_softcap():
    """Softcap-on labeled path keeps logits. The helper sees `logits=`, not fused hidden+weight."""
    torch.manual_seed(1)
    config = _tiny_config(final_logit_softcapping=30.0)
    hf = HFGemma3ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    hf_logits = hf(input_ids=input_ids, use_cache=False).logits
    ours_logits = ours(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(ours_logits, hf_logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    assert ours_out.logits is not None
    torch.testing.assert_close(ours_out.logits, hf_out.logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    hf_out.loss.backward()
    ours_out.loss.backward()
    hf_grads = named_trainable(hf)
    ours_grads = named_trainable(ours)
    assert hf_grads.keys() == ours_grads.keys()
    for name, param in hf_grads.items():
        if param.grad is None:
            assert ours_grads[name].grad is None, name
            continue
        assert ours_grads[name].grad is not None, name
        torch.testing.assert_close(
            ours_grads[name].grad,
            param.grad,
            atol=EAGER_GRAD_ATOL,
            rtol=EAGER_GRAD_RTOL,
            msg=name,
        )


def test_gemma3_packed_eager_matches_independent_samples():
    """Packed ``cu_seq_lens_q`` uses ``packed_causal_mask`` / ``sliding_window_mask``."""
    torch.manual_seed(123)
    config = _tiny_config()
    ours = _build_ours(config).eval()
    first_input_ids = torch.tensor([[5, 6, 7]])
    second_input_ids = torch.tensor([[8, 9, 10, 11, 12]])
    packed_input_ids = torch.cat((first_input_ids, second_input_ids), dim=1)

    with torch.no_grad():
        packed_logits = ours(
            input_ids=packed_input_ids,
            attention_mask=torch.ones_like(packed_input_ids),
            position_ids=torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]]),
            cu_seq_lens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
            use_cache=False,
        ).logits
        first_logits = ours(
            input_ids=first_input_ids,
            attention_mask=torch.ones_like(first_input_ids),
            position_ids=torch.arange(3).unsqueeze(0),
            cu_seq_lens_q=torch.tensor([0, 3], dtype=torch.int32),
            use_cache=False,
        ).logits
        second_logits = ours(
            input_ids=second_input_ids,
            attention_mask=torch.ones_like(second_input_ids),
            position_ids=torch.arange(5).unsqueeze(0),
            cu_seq_lens_q=torch.tensor([0, 5], dtype=torch.int32),
            use_cache=False,
        ).logits

    torch.testing.assert_close(packed_logits[:, :3], first_logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(packed_logits[:, 3:], second_logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)


def test_gemma3_routes_full_and_sliding_flex_masks(monkeypatch):
    captured = []

    def fake_flex(module, query, key, value, attention_mask, *, sliding_window=None, **kwargs):
        captured.append((module.layer_type, attention_mask, sliding_window))
        return torch.zeros_like(query).transpose(1, 2), None

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "veomni_flex_attention", fake_flex)
    ops = _flex_ops_config()
    config = _tiny_config()
    config.sliding_window = 4
    model = _build_ours(config, ops).eval()
    input_ids = torch.randint(3, model.config.vocab_size, (1, 8))

    with ops_config_scope(ops), torch.no_grad():
        output = model(input_ids=input_ids, use_cache=False)

    assert torch.isfinite(output.logits).all()
    assert [layer_type for layer_type, _, _ in captured] == config.layer_types
    assert all(isinstance(attention_mask, BlockMask) for _, attention_mask, _ in captured)
    assert [sliding_window for _, _, sliding_window in captured] == [
        config.sliding_window if layer_type == "sliding_attention" else None for layer_type in config.layer_types
    ]

    zero = torch.tensor(0)
    query_idx = torch.tensor(7)
    sliding_mask = next(mask for layer_type, mask, _ in captured if layer_type == "sliding_attention")
    full_mask = next(mask for layer_type, mask, _ in captured if layer_type == "full_attention")
    assert not sliding_mask.mask_mod(zero, zero, query_idx, torch.tensor(0))
    assert sliding_mask.mask_mod(zero, zero, query_idx, torch.tensor(6))
    assert full_mask.mask_mod(zero, zero, query_idx, torch.tensor(0))


def test_gemma3_packed_flex_matches_independent_samples_on_cpu():
    torch.compiler.reset()
    torch.manual_seed(123)
    ops = _flex_ops_config()
    model = _build_ours(_tiny_config(), ops).eval()
    first_input_ids = torch.tensor([[5, 6, 7]])
    second_input_ids = torch.tensor([[8, 9, 10, 11, 12]])
    packed_input_ids = torch.cat((first_input_ids, second_input_ids), dim=1)

    with ops_config_scope(ops), torch.no_grad():
        packed_logits = model(
            input_ids=packed_input_ids,
            attention_mask=torch.ones_like(packed_input_ids),
            position_ids=torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]]),
            cu_seq_lens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
            use_cache=False,
        ).logits
        first_logits = model(
            input_ids=first_input_ids,
            attention_mask=torch.ones_like(first_input_ids),
            position_ids=torch.arange(3).unsqueeze(0),
            cu_seq_lens_q=torch.tensor([0, 3], dtype=torch.int32),
            use_cache=False,
        ).logits
        second_logits = model(
            input_ids=second_input_ids,
            attention_mask=torch.ones_like(second_input_ids),
            position_ids=torch.arange(5).unsqueeze(0),
            cu_seq_lens_q=torch.tensor([0, 5], dtype=torch.int32),
            use_cache=False,
        ).logits

    torch.testing.assert_close(packed_logits[:, :3], first_logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(packed_logits[:, 3:], second_logits, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Gemma 3 packed FlexAttention requires CUDA")
def test_gemma3_packed_flex_matches_independent_samples_on_cuda():
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    torch.manual_seed(127)
    ops = _flex_ops_config()
    model = _build_ours(_tiny_config(), ops).to(device=device, dtype=dtype).eval()
    packed_inputs = torch.randn(
        1,
        8,
        model.config.hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    with ops_config_scope(ops):
        with torch.no_grad():
            first_logits = model(
                inputs_embeds=packed_inputs[:, :3].detach(),
                attention_mask=torch.ones(1, 3, device=device, dtype=torch.long),
                position_ids=torch.arange(3, device=device).unsqueeze(0),
                cu_seq_lens_q=torch.tensor([0, 3], device=device, dtype=torch.int32),
                use_cache=False,
            ).logits
            second_logits = model(
                inputs_embeds=packed_inputs[:, 3:].detach(),
                attention_mask=torch.ones(1, 5, device=device, dtype=torch.long),
                position_ids=torch.arange(5, device=device).unsqueeze(0),
                cu_seq_lens_q=torch.tensor([0, 5], device=device, dtype=torch.int32),
                use_cache=False,
            ).logits

        packed_logits = model(
            inputs_embeds=packed_inputs,
            attention_mask=torch.ones(1, 8, device=device, dtype=torch.long),
            position_ids=torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]], device=device),
            cu_seq_lens_q=torch.tensor([0, 3, 8], device=device, dtype=torch.int32),
            use_cache=False,
        ).logits

    torch.testing.assert_close(packed_logits[:, :3], first_logits, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(packed_logits[:, 3:], second_logits, rtol=3e-2, atol=3e-2)

    packed_logits.float().square().mean().backward()
    gradients = (
        packed_inputs.grad,
        model.model.layers[0].self_attn.q_proj.weight.grad,
        model.model.layers[0].self_attn.k_proj.weight.grad,
        model.model.layers[0].self_attn.v_proj.weight.grad,
        model.model.layers[0].self_attn.o_proj.weight.grad,
        model.lm_head.weight.grad,
    )
    for gradient in gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max() > 0


def test_gemma3_builds_global_pack_aware_flex_masks_with_ulysses(monkeypatch):
    from veomni.ops.kernels.attention import ulysses

    captured = []

    def fake_flex(module, query, key, value, attention_mask, **kwargs):
        captured.append(attention_mask)
        return torch.zeros_like(query).transpose(1, 2), None

    monkeypatch.setattr(
        ulysses,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_size=2, async_enabled=False),
    )
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "veomni_flex_attention", fake_flex)
    ops = _flex_ops_config()
    config = _tiny_config()
    model = _build_ours(config, ops).eval()
    local_input_ids = torch.randint(3, model.config.vocab_size, (1, 4))

    with ops_config_scope(ops), torch.no_grad():
        model(
            input_ids=local_input_ids,
            attention_mask=torch.ones(1, 8, dtype=torch.long),
            position_ids=torch.tensor([[0, 1, 2, 0]]),
            cu_seq_lens_q=torch.tensor([0, 3, 5, 8], dtype=torch.int32),
            use_cache=False,
        )

    assert len(captured) == config.num_hidden_layers
    zero = torch.tensor(0)
    for block_mask in captured:
        assert block_mask.shape == (1, 1, 8, 8)
        assert not block_mask.mask_mod(zero, zero, torch.tensor(5), torch.tensor(4))


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Gemma 3 FlexAttention backward requires CUDA")
def test_gemma3_flex_matches_math_sdpa_forward_and_backward():
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    torch.manual_seed(29)
    config = _tiny_config()
    math_model = HFGemma3ForCausalLM._from_config(copy.deepcopy(config), attn_implementation="sdpa")
    ops = _flex_ops_config()
    flex_model = _build_ours(copy.deepcopy(config), ops)
    flex_model.load_state_dict(math_model.state_dict())
    math_model.to(device=device, dtype=dtype).train()
    flex_model.to(device=device, dtype=dtype).train()

    math_inputs = torch.randn(1, 8, config.hidden_size, device=device, dtype=dtype, requires_grad=True)
    flex_inputs = math_inputs.detach().clone().requires_grad_(True)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        math_logits = math_model(inputs_embeds=math_inputs, use_cache=False).logits
    with ops_config_scope(ops):
        flex_logits = flex_model(inputs_embeds=flex_inputs, use_cache=False).logits

    torch.testing.assert_close(flex_logits, math_logits, rtol=3e-2, atol=3e-2)

    math_logits.float().square().mean().backward()
    flex_logits.float().square().mean().backward()
    torch.testing.assert_close(flex_inputs.grad, math_inputs.grad, rtol=5e-2, atol=5e-2)

    parameter_names = (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "lm_head.weight",
    )
    math_parameters = dict(math_model.named_parameters())
    flex_parameters = dict(flex_model.named_parameters())
    for name in parameter_names:
        math_gradient = math_parameters[name].grad
        flex_gradient = flex_parameters[name].grad
        assert math_gradient is not None and torch.isfinite(math_gradient).all()
        assert flex_gradient is not None and torch.isfinite(flex_gradient).all()
        torch.testing.assert_close(flex_gradient, math_gradient, rtol=7e-2, atol=7e-2)
