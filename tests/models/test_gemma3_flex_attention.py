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

import copy
import gc
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.functional import scaled_dot_product_attention
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM as HFGemma3ForCausalLM

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.models.transformers.gemma3.generated import patched_modeling_gemma3_gpu as gemma3_modeling
from veomni.models.transformers.gemma3.generated.patched_modeling_gemma3_gpu import (
    Gemma3ForCausalLM as VeOmniGemma3ForCausalLM,
)
from veomni.models.transformers.gemma3.generated.patched_modeling_gemma3_gpu import (
    Gemma3TextModel as VeOmniGemma3TextModel,
)
from veomni.ops.kernels import attention as veomni_attention
from veomni.ops.kernels.attention import flex as flex_attention
from veomni.utils.device import IS_CUDA_AVAILABLE, empty_cache, get_device_type, get_torch_device, synchronize


_TOY_CONFIG = Path(__file__).parents[1] / "toy_config" / "gemma3_toy"
_FLEX_IMPLEMENTATION = "veomni_flex_attention_with_sp"
_EFFICIENT_PROFILE_IMPLEMENTATION = "gemma3_efficient_attention_profile"
_RUN_PROFILE = os.environ.get("RUN_GEMMA3_FLEX_PROFILE") == "1"
_PROFILE_SEQUENCE_LENGTHS = (4096, 8192, 20000)
_PROFILE_ITERATIONS = 5


def _eager_ops_config(
    attn_implementation: str,
    *,
    cross_entropy_loss_implementation: str = "eager",
) -> OpsImplementationConfig:
    return OpsImplementationConfig(
        attn_implementation=attn_implementation,
        moe_implementation="eager",
        cross_entropy_loss_implementation=cross_entropy_loss_implementation,
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
    )


def _toy_config() -> Gemma3TextConfig:
    return Gemma3TextConfig.from_pretrained(_TOY_CONFIG)


def _profile_config() -> Gemma3TextConfig:
    return Gemma3TextConfig(
        vocab_size=128,
        hidden_size=640,
        intermediate_size=2048,
        num_hidden_layers=18,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=256,
        max_position_embeddings=32768,
        sliding_window=512,
        layer_types=["full_attention" if (layer_idx + 1) % 6 == 0 else "sliding_attention" for layer_idx in range(18)],
        query_pre_attn_scalar=256,
        tie_word_embeddings=False,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        use_cache=False,
    )


def _efficient_attention_profile_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    **kwargs,
):
    repeat_count = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeat_count, dim=1)
    value = value.repeat_interleave(repeat_count, dim=1)
    is_causal = query.shape[2] > 1 and attention_mask is None and module.is_causal
    with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
    return output.transpose(1, 2).contiguous(), None


def test_gemma3_text_registry_builds_generated_flex_model(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    model = build_foundation_model(
        _TOY_CONFIG,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_eager_ops_config("flex_attention"),
    )

    assert isinstance(model, VeOmniGemma3ForCausalLM)
    assert model.config.model_type == "gemma3_text"
    assert model.config._attn_implementation == _FLEX_IMPLEMENTATION


def test_gemma3_patched_causal_lm_forward_supports_eager_training(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torch.manual_seed(17)
    model = build_foundation_model(
        _TOY_CONFIG,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_eager_ops_config("eager"),
    )
    input_ids = torch.randint(3, model.config.vocab_size, (1, 8))

    output = model(input_ids=input_ids, labels=input_ids.clone(), use_cache=False)
    output.loss.backward()

    assert output.logits.shape == (1, 8, model.config.vocab_size)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(model.model.layers[0].self_attn.q_proj.weight.grad).all()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Liger fused linear cross entropy requires CUDA")
def test_gemma3_patched_causal_lm_forward_supports_fused_loss(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torch.manual_seed(23)
    model = build_foundation_model(
        _TOY_CONFIG,
        torch_dtype="bfloat16",
        init_device=get_device_type(),
        ops_implementation=_eager_ops_config(
            "eager",
            cross_entropy_loss_implementation="liger_kernel",
        ),
    ).train()
    input_ids = torch.randint(3, model.config.vocab_size, (2, 16), device=model.device)
    labels = input_ids.clone()
    labels[0, 0] = -100

    assert gemma3_modeling.veomni_causal_lm_loss.use_non_eager_impl
    with torch.no_grad():
        reference_logits = model(input_ids=input_ids, use_cache=False).logits
        reference_loss = F.cross_entropy(
            reference_logits[:, :-1].float().reshape(-1, model.config.vocab_size),
            labels[:, 1:].reshape(-1),
        )

    model.zero_grad(set_to_none=True)
    output = model(input_ids=input_ids, labels=labels, use_cache=False)

    assert output.logits is None
    assert output.fused_linear_aux is None
    assert torch.isfinite(output.loss)
    torch.testing.assert_close(output.loss, reference_loss, rtol=1e-2, atol=1e-2)

    output.loss.backward()
    for parameter in (
        model.model.layers[0].self_attn.q_proj.weight,
        model.lm_head.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0


def test_gemma3_routes_native_full_and_sliding_block_masks(monkeypatch):
    captured = []

    def fake_flex_adapter(module, query, key, value, attention_mask, *, sliding_window=None, **kwargs):
        captured.append((module.layer_type, attention_mask, sliding_window))
        return torch.zeros_like(query).transpose(1, 2), None

    monkeypatch.setitem(
        veomni_attention._ATTENTION_FORWARD_DISPATCH,
        _FLEX_IMPLEMENTATION,
        fake_flex_adapter,
    )
    config = _toy_config()
    model = VeOmniGemma3ForCausalLM._from_config(config, attn_implementation=_FLEX_IMPLEMENTATION).eval()
    input_ids = torch.randint(3, config.vocab_size, (1, 8))

    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=False)

    assert torch.isfinite(output.logits).all()
    assert [layer_type for layer_type, _, _ in captured] == ["sliding_attention", "full_attention"]
    assert all(isinstance(attention_mask, BlockMask) for _, attention_mask, _ in captured)
    assert [sliding_window for _, _, sliding_window in captured] == [4, None]

    sliding_mask = captured[0][1]
    full_mask = captured[1][1]
    zero = torch.tensor(0)
    query_idx = torch.tensor(7)
    distant_key_idx = torch.tensor(0)
    nearby_key_idx = torch.tensor(6)
    assert not sliding_mask.mask_mod(zero, zero, query_idx, distant_key_idx)
    assert sliding_mask.mask_mod(zero, zero, query_idx, nearby_key_idx)
    assert full_mask.mask_mod(zero, zero, query_idx, distant_key_idx)


def test_gemma3_builds_pack_aware_full_and_sliding_block_masks(monkeypatch):
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
    model = VeOmniGemma3ForCausalLM._from_config(config, attn_implementation=_FLEX_IMPLEMENTATION).eval()
    input_ids = torch.randint(3, config.vocab_size, (1, 8))
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
        assert not block_mask.mask_mod(zero, zero, first_token_in_second_pack, last_token_in_first_pack)


def test_gemma3_packed_flex_matches_independent_samples_on_cpu():
    torch.compiler.reset()
    torch.manual_seed(123)
    config = _toy_config()
    model = VeOmniGemma3ForCausalLM._from_config(config, attn_implementation=_FLEX_IMPLEMENTATION).eval()
    first_input_ids = torch.tensor([[5, 6, 7]])
    second_input_ids = torch.tensor([[8, 9, 10, 11, 12]])
    packed_input_ids = torch.cat((first_input_ids, second_input_ids), dim=1)

    with torch.no_grad():
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
    config = _toy_config()
    model = VeOmniGemma3ForCausalLM._from_config(
        config,
        attn_implementation=_FLEX_IMPLEMENTATION,
    ).to(device=device, dtype=dtype)
    model.eval()
    packed_inputs = torch.randn(
        1,
        8,
        config.hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

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

    assert model.config._attn_implementation == _FLEX_IMPLEMENTATION
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


def test_gemma3_builds_global_pack_aware_masks_with_ulysses(monkeypatch):
    captured = []

    def fake_flex_adapter(module, query, key, value, attention_mask, **kwargs):
        captured.append(attention_mask)
        return torch.zeros_like(query).transpose(1, 2), None

    monkeypatch.setattr(
        flex_attention,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=True, ulysses_size=2),
    )
    monkeypatch.setitem(
        veomni_attention._ATTENTION_FORWARD_DISPATCH,
        _FLEX_IMPLEMENTATION,
        fake_flex_adapter,
    )
    config = _toy_config()
    model = VeOmniGemma3ForCausalLM._from_config(config, attn_implementation=_FLEX_IMPLEMENTATION).eval()
    local_input_ids = torch.randint(3, config.vocab_size, (1, 4))

    with torch.no_grad():
        model(
            input_ids=local_input_ids,
            attention_mask=torch.ones(1, 8, dtype=torch.long),
            position_ids=torch.tensor([[0, 1, 2, 0]]),
            cu_seq_lens_q=torch.tensor([0, 3, 5, 8], dtype=torch.int32),
            use_cache=False,
        )

    assert len(captured) == 2
    zero = torch.tensor(0)
    for block_mask in captured:
        assert block_mask.shape == (1, 1, 8, 8)
        assert not block_mask.mask_mod(zero, zero, torch.tensor(5), torch.tensor(4))


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Gemma 3 FlexAttention backward requires CUDA")
def test_gemma3_flex_matches_math_sdpa_forward_and_backward():
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    torch.manual_seed(29)
    config = _toy_config()
    math_model = HFGemma3ForCausalLM._from_config(copy.deepcopy(config), attn_implementation="sdpa")
    flex_model = VeOmniGemma3ForCausalLM._from_config(
        copy.deepcopy(config),
        attn_implementation=_FLEX_IMPLEMENTATION,
    )
    flex_model.load_state_dict(math_model.state_dict())
    math_model.to(device=device, dtype=dtype).train()
    flex_model.to(device=device, dtype=dtype).train()

    math_inputs = torch.randn(1, 8, config.hidden_size, device=device, dtype=dtype, requires_grad=True)
    flex_inputs = math_inputs.detach().clone().requires_grad_(True)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        math_logits = math_model(inputs_embeds=math_inputs, use_cache=False).logits
    flex_logits = flex_model(inputs_embeds=flex_inputs, use_cache=False).logits

    assert torch.isfinite(math_logits).all()
    assert torch.isfinite(flex_logits).all()
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


def _run_profile_iteration(model, inputs_embeds):
    device_api = get_torch_device()
    model.zero_grad(set_to_none=True)
    inputs_embeds.grad = None
    synchronize()
    start = device_api.Event(enable_timing=True)
    end = device_api.Event(enable_timing=True)
    start.record()
    output = model(inputs_embeds=inputs_embeds, use_cache=False).last_hidden_state
    output.float().square().mean().backward()
    end.record()
    synchronize()

    gradients = (
        inputs_embeds.grad,
        model.layers[0].self_attn.q_proj.weight.grad,
        model.layers[0].self_attn.k_proj.weight.grad,
        model.layers[0].self_attn.v_proj.weight.grad,
        model.layers[0].self_attn.o_proj.weight.grad,
    )
    all_finite = torch.isfinite(output).all() and all(
        gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
    )
    return start.elapsed_time(end), bool(all_finite)


def _profile_gemma3_backend(sequence_length, backend):
    device_api = get_torch_device()
    gc.collect()
    empty_cache()
    if backend == "flex_attention":
        torch.compiler.reset()

    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    config = _profile_config()
    if backend == "efficient_attention":
        ALL_ATTENTION_FUNCTIONS.register(_EFFICIENT_PROFILE_IMPLEMENTATION, _efficient_attention_profile_forward)
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            _EFFICIENT_PROFILE_IMPLEMENTATION,
            ALL_MASK_ATTENTION_FUNCTIONS["sdpa"],
        )
        implementation = _EFFICIENT_PROFILE_IMPLEMENTATION
    else:
        implementation = _FLEX_IMPLEMENTATION
    torch.manual_seed(41)
    model = VeOmniGemma3TextModel._from_config(config, attn_implementation=implementation)
    model.to(device=device, dtype=dtype).train()
    inputs_embeds = torch.randn(
        1,
        sequence_length,
        config.hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    device_api.reset_peak_memory_stats()

    try:
        first_iteration_ms, first_finite = _run_profile_iteration(model, inputs_embeds)
        first_iteration_peak_allocated_gib = device_api.max_memory_allocated() / 1024**3
        warmup_ms, warmup_finite = _run_profile_iteration(model, inputs_embeds)
        device_api.reset_peak_memory_stats()
        steady_state_times_ms = []
        all_finite = first_finite and warmup_finite
        for _ in range(_PROFILE_ITERATIONS):
            elapsed_ms, iteration_finite = _run_profile_iteration(model, inputs_embeds)
            steady_state_times_ms.append(elapsed_ms)
            all_finite = all_finite and iteration_finite
        result = {
            "backend": backend,
            "status": "completed",
            "first_iteration_ms": first_iteration_ms,
            "first_iteration_peak_allocated_gib": first_iteration_peak_allocated_gib,
            "post_first_warmup_ms": warmup_ms,
            "steady_state_iterations": _PROFILE_ITERATIONS,
            "steady_state_times_ms": steady_state_times_ms,
            "steady_state_median_ms": statistics.median(steady_state_times_ms),
            "peak_allocated_gib": device_api.max_memory_allocated() / 1024**3,
            "all_outputs_and_gradients_finite": all_finite,
        }
    except device_api.OutOfMemoryError as error:
        free_bytes, total_bytes = device_api.mem_get_info()
        result = {
            "backend": backend,
            "status": "oom",
            "error": str(error),
            "allocated_gib": device_api.memory_allocated() / 1024**3,
            "reserved_gib": device_api.memory_reserved() / 1024**3,
            "free_gib": free_bytes / 1024**3,
            "total_gib": total_bytes / 1024**3,
        }
    finally:
        del model
        del inputs_embeds
        gc.collect()
        empty_cache()

    return result


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Gemma 3 profiling requires CUDA")
@pytest.mark.skipif(
    not _RUN_PROFILE,
    reason="Set RUN_GEMMA3_FLEX_PROFILE=1 to run the Gemma 3 CUDA profile",
)
@pytest.mark.benchmark
@pytest.mark.parametrize("sequence_length", _PROFILE_SEQUENCE_LENGTHS)
def test_gemma3_profiles_efficient_sdpa_against_veomni_flex(sequence_length):
    results = {
        "sequence_length": sequence_length,
        "dtype": str(torch.bfloat16),
        "batch_size": 1,
        "geometry": {
            "hidden_size": 640,
            "intermediate_size": 2048,
            "layers": 18,
            "query_heads": 4,
            "kv_heads": 1,
            "head_dim": 256,
            "sliding_window": 512,
            "full_attention_every": 6,
            "efficient_attention_expanded_kv_heads": 4,
        },
        "efficient_attention": _profile_gemma3_backend(sequence_length, "efficient_attention"),
        "flex_attention": _profile_gemma3_backend(sequence_length, "flex_attention"),
    }
    print(json.dumps(results, indent=2))

    for backend_result in (results["efficient_attention"], results["flex_attention"]):
        assert backend_result["status"] in {"completed", "oom"}
        if backend_result["status"] == "completed":
            assert backend_result["all_outputs_and_gradients_finite"]
