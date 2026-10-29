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

"""Gemma numerical oracles keep visibility, precision and weights explicit."""

import warnings
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import modeling_flash_attention_utils as hf_flash_utils
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM as HFTextLM

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.tiny_configs import tiny_gemma4_text_config
from tests.tools.gemma4_numerics import attention_module, dense_masks, fp32_attention, precision_config, run_kernel
from tests.tools.training_utils import make_eager_ops_config
from veomni.models.transformers.gemma4.generated import patched_modeling_gemma4_gpu as modeling
from veomni.ops import config as ops_config
from veomni.utils.device import IS_CUDA_AVAILABLE


_GPU = pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
_CUDA = pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Real Gemma attention kernels require CUDA")


@pytest.fixture
def fresh_compiler_cache():
    # Unrelated head/length variants can exhaust Dynamo's recompile limit and
    # silently substitute eager Flex math for the real Triton kernel under test.
    torch.compiler.reset()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="flex_attention called without torch.compile")
            yield
    finally:
        torch.compiler.reset()


def _require_backend(backend):
    if backend == "flash_attention_2":
        from transformers.utils import is_flash_attn_2_available

        if not is_flash_attn_2_available():
            pytest.skip("FA2 is not installed")
    else:
        pytest.importorskip("triton")


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("configured_window", [4, 32])
@pytest.mark.parametrize("direction", [-1, 1], ids=["left", "right"])
def test_bidirectional_window_endpoint_uses_backend_specific_reference(configured_window, direction):
    """A value only on the boundary distinguishes inclusive and native Flash windows."""
    config_dict = tiny_gemma4_text_config().to_dict()
    config_dict.update(use_bidirectional_attention="all", sliding_window=configured_window)
    config = modeling.Gemma4TextConfig.from_dict(config_dict)
    config._attn_implementation = "sdpa"
    window = config.sliding_window
    document_length = 2 * window + 3
    length = 2 * document_length
    cu = torch.tensor([0, document_length, length], dtype=torch.int32)
    mask = modeling._gemma4_create_mask(
        config=config,
        inputs_embeds=torch.zeros(1, length, config.hidden_size),
        attention_mask=None,
        past_key_values=None,
        cu_seq_lens_q=cu,
        sliding=True,
    )
    supports = dict.fromkeys(
        ("dropout_p", "deterministic", "softcap", "s_aux", "learnable_sink", "max_seqlen_q", "max_seqlen_k"),
        False,
    )
    supports["window_size"] = True
    native = hf_flash_utils._process_flash_attention_kwargs(
        query_length=length,
        key_length=length,
        is_causal=False,
        sliding_window=window,
        supports_mapping=supports,
    )
    flash_radius = window - 1
    assert native["window_size"] == (flash_radius, flash_radius)
    positions = torch.arange(length)
    same_document = positions[:, None] // document_length == positions[None, :] // document_length
    flash_visible = same_document & ((positions[:, None] - positions[None, :]).abs() <= flash_radius)
    q = torch.zeros(1, 1, length, 16)
    v = torch.zeros_like(q)
    query = window + 1
    boundary = query + direction * window
    v[:, :, boundary] = 1
    dense_output = F.scaled_dot_product_attention(q, q, v, attn_mask=mask, scale=1.0)
    flash_reference = F.scaled_dot_product_attention(q, q, v, attn_mask=flash_visible, scale=1.0)
    torch.testing.assert_close(dense_output[0, 0, query], torch.full((16,), 1 / (2 * window + 1)))
    assert torch.count_nonzero(flash_reference[0, 0, query]) == 0
    assert mask[0, 0, query, boundary]
    assert not flash_visible[query, boundary]
    assert not mask[0, 0, document_length, document_length - 1]


@pytest.fixture
def eager_loss(monkeypatch):
    """Keep model construction independent of other tests' installed op selection."""
    monkeypatch.setattr(ops_config, "_ops_config", make_eager_ops_config())


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("bidirectional", [False, True], ids=["causal", "bidirectional"])
def test_same_backend_matches_hf_with_shared_kv(eager_loss, dtype, backend, bidirectional):
    """Compare all gradients with HF using the same weights, precision and visibility."""
    torch.manual_seed(1242)
    config = precision_config(bidirectional)
    reference = HFTextLM._from_config(deepcopy(config), attn_implementation=backend, dtype=torch.float32)
    candidate = modeling.Gemma4ForCausalLM._from_config(
        deepcopy(config), attn_implementation="veomni_sdpa" if backend == "sdpa" else backend, dtype=torch.float32
    )
    candidate.load_state_dict(reference.state_dict())
    reference.to(dtype=dtype)
    candidate.to(dtype=dtype)
    reference_buffers = dict(reference.named_buffers())
    assert reference_buffers.keys() == dict(candidate.named_buffers()).keys()
    for name, value in candidate.named_buffers():
        torch.testing.assert_close(value, reference_buffers[name], rtol=0, atol=0)
    for model in (reference, candidate):
        assert model.lm_head.weight is model.model.embed_tokens.weight
        assert [layer.self_attn.is_kv_shared_layer for layer in model.model.layers] == [False, False, True, True]
    ids = torch.randint(3, 100, (1, 128), generator=torch.Generator().manual_seed(93))
    labels = ids.clone()
    labels[:, 64] = -100
    cu = torch.tensor([0, 64, 128], dtype=torch.int32)
    common = dict(input_ids=ids, labels=labels, position_ids=torch.arange(64).repeat(2).unsqueeze(0), use_cache=False)
    masks = dense_masks(config, 128, cu, backend=backend, dtype=dtype)
    with sdpa_kernel(SDPBackend.MATH):
        expected = reference(**common, attention_mask=masks)
        actual = candidate(**common, cu_seq_lens_q=cu)
        expected.loss.backward()
        actual.loss.backward()
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    reference_parameters = dict(reference.named_parameters())
    assert reference_parameters.keys() == dict(candidate.named_parameters()).keys()
    for name, parameter in candidate.named_parameters():
        expected_gradient = reference_parameters[name].grad
        assert (parameter.grad is None) == (expected_gradient is None), name
        if expected_gradient is not None:
            torch.testing.assert_close(
                parameter.grad, expected_gradient, rtol=0, atol=0, msg=lambda m, name=name: f"{name}: {m}"
            )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_fp32_attention_reference_preserves_math_precision(monkeypatch):
    """An outer MATH context cannot replace explicit promotion inside the oracle."""
    seen = []
    native = F.scaled_dot_product_attention

    def record(query, key, value, **kwargs):
        seen.append((query.dtype, key.dtype, value.dtype))
        assert torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        return native(query, key, value, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    generator = torch.Generator().manual_seed(1242)
    tensors = [torch.randn(1, heads, 8, 16, generator=generator).bfloat16().requires_grad_() for heads in (4, 2, 2)]
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool).tril()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        output, _ = fp32_attention(None, *tensors, mask, scaling=1.0)
        assert torch.backends.cuda.mem_efficient_sdp_enabled()
    assert seen == [(torch.float32,) * 3]
    assert output.dtype == torch.bfloat16
    output.float().square().sum().backward()
    for value in tensors:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


@_GPU
@_CUDA
@pytest.mark.parametrize("direction", [-1, 1], ids=["left", "right"])
def test_real_flash_bidirectional_endpoint_matches_native_reference(direction, fresh_compiler_cache):
    """Real FA2 excludes the effective-window endpoint that dense Gemma includes."""
    _require_backend("flash_attention_2")
    config = precision_config(bidirectional=True)
    window = config.sliding_window
    document_length = 2 * window + 3
    length = 2 * document_length
    cu = torch.tensor([0, document_length, length], device="cuda", dtype=torch.int32)
    q = torch.zeros(1, 2, length, 64, device="cuda", dtype=torch.bfloat16)
    k = q[:, :1].clone()
    v = torch.zeros_like(k)
    query = window + 1
    v[:, :, query + direction * window] = 1
    actual = run_kernel(attention_module(config, True), (q, k, v), None, cu, length, "flash_attention_2")
    native_mask = dense_masks(config, length, cu, flash=True, device="cuda")["sliding_attention"]
    expected = fp32_attention(None, q, k, v, native_mask, scaling=1.0)[0]
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[0, query]) == 0


@_GPU
@_CUDA
@pytest.mark.parametrize("backend", ["flex_attention", "flash_attention_2"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("bidirectional", [False, True], ids=["causal", "bidirectional"])
@pytest.mark.parametrize("length", [128, 512])
def test_real_attention_shared_kv_vjp_matches_fp32(backend, dtype, bidirectional, length, fresh_compiler_cache):
    """All four Q streams share two K/V leaves, matching Gemma's KV-sharing graph."""
    from tests.tools.gemma4_numerics import relative_l2

    _require_backend(backend)
    config = precision_config(bidirectional)
    config._attn_implementation = "veomni_" + backend
    generator = torch.Generator().manual_seed(1242)

    def sample(heads):
        value = torch.randn(1, heads, length, 256, generator=generator)
        return value.to(device="cuda", dtype=dtype).requires_grad_()

    leaves = [sample(4) for _ in range(4)] + [sample(2) for _ in range(4)]
    reference_leaves = [value.detach().float().requires_grad_() for value in leaves]
    cu = torch.tensor([0, length // 2, length], device="cuda", dtype=torch.int32)
    reference_masks = dense_masks(config, length, cu, device="cuda", flash=backend == "flash_attention_2")
    actual_outputs, reference_outputs, upstream = [], [], []
    for layer in range(4):
        sliding = layer % 2 == 0
        pair = layer % 2
        indices = (layer, 4 + pair, 6 + pair)
        qkv = [leaves[index] for index in indices]
        oracle_qkv = [reference_leaves[index] for index in indices]
        mask = None
        if backend == "flex_attention":
            mask = modeling._gemma4_create_mask(
                config=config,
                inputs_embeds=torch.zeros(1, length, 64, device="cuda", dtype=dtype),
                attention_mask=None,
                past_key_values=None,
                cu_seq_lens_q=cu,
                sliding=sliding,
            )
        actual = run_kernel(attention_module(config, sliding), qkv, mask, cu, length, backend)
        dense = reference_masks["sliding_attention" if sliding else "full_attention"]
        expected = fp32_attention(None, *oracle_qkv, dense, scaling=1.0)[0]
        torch.testing.assert_close(actual.float(), expected, rtol=3e-2, atol=3e-2)
        actual_outputs.append(actual)
        reference_outputs.append(expected)
        upstream.append(torch.randn(actual.shape, generator=generator).to(device="cuda", dtype=dtype))
    actual_gradients = torch.autograd.grad(actual_outputs, leaves, upstream)
    expected_gradients = torch.autograd.grad(
        reference_outputs, reference_leaves, [value.float() for value in upstream]
    )
    for index, (actual, expected) in enumerate(zip(actual_gradients, expected_gradients, strict=True)):
        assert torch.isfinite(actual).all()
        assert relative_l2(actual, expected) < 0.03, (index, relative_l2(actual, expected))


_MODEL_CASES = [
    pytest.param(backend, dtype, bidirectional, length, id=f"{dtype}-{backend}-{bidirectional}-{length}")
    for dtype in ("bf16", "fp16")
    for backend in ("flex_attention", "flash_attention_2")
    for bidirectional in (False, True)
    for length in (128, 512)
    if dtype == "bf16" or (backend == "flash_attention_2" and not bidirectional and length == 512)
]


@_GPU
@_CUDA
@pytest.mark.parametrize("backend,dtype_name,bidirectional,length", _MODEL_CASES)
def test_real_model_same_backend_matches_hf(
    eager_loss, backend, dtype_name, bidirectional, length, fresh_compiler_cache
):
    """The nine historical failures use native HF with the same backend as oracle."""
    _require_backend(backend)
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    torch.manual_seed(1242)
    config = precision_config(bidirectional)
    reference = HFTextLM._from_config(deepcopy(config), attn_implementation="eager", dtype=torch.float32)
    candidate = modeling.Gemma4ForCausalLM._from_config(
        deepcopy(config), attn_implementation="eager", dtype=torch.float32
    )
    candidate.load_state_dict(reference.state_dict())
    reference.to(device="cuda", dtype=dtype)
    candidate.to(device="cuda", dtype=dtype)
    reference.config._attn_implementation = backend
    candidate.config._attn_implementation = "veomni_" + backend
    reference_buffers = dict(reference.named_buffers())
    for name, value in candidate.named_buffers():
        torch.testing.assert_close(value, reference_buffers[name], rtol=0, atol=0)
    assert candidate.lm_head.weight is candidate.model.embed_tokens.weight
    assert reference.lm_head.weight is reference.model.embed_tokens.weight
    cu = torch.tensor([0, length // 2, length], device="cuda", dtype=torch.int32)
    masks = dict.fromkeys(("full_attention", "sliding_attention"))
    if backend == "flex_attention":
        for layer_type in masks:
            masks[layer_type] = modeling._gemma4_create_mask(
                config=candidate.config,
                inputs_embeds=torch.zeros(1, length, config.hidden_size, device="cuda", dtype=dtype),
                attention_mask=None,
                past_key_values=None,
                cu_seq_lens_q=cu,
                sliding=layer_type == "sliding_attention",
            )
    ids = torch.randint(3, 100, (1, length), generator=torch.Generator().manual_seed(93)).cuda()
    labels = ids.clone()
    labels[:, length // 2] = -100
    inputs = dict(
        input_ids=ids,
        labels=labels,
        position_ids=torch.arange(length // 2, device="cuda").repeat(2).unsqueeze(0),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=length // 2,
        max_length_k=length // 2,
        attention_mask=masks,
        use_cache=False,
        deterministic=True,
        kernel_options={"BACKEND": "TRITON", "num_stages": 1},
    )
    expected = reference(**inputs)
    actual = candidate(**inputs)
    expected.loss.backward()
    actual.loss.backward()
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.loss, expected.loss)
    reference_parameters = dict(reference.named_parameters())
    assert reference_parameters.keys() == dict(candidate.named_parameters()).keys()
    for name, parameter in candidate.named_parameters():
        gradient = reference_parameters[name].grad
        assert (parameter.grad is None) == (gradient is None), name
        if gradient is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, gradient, msg=lambda m, name=name: f"{name}: {m}")
