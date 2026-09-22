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

"""DeepSeek-V4 operator integration and Hugging Face parity tests.

Direct-import the generated class. Compare a toy CausalLM against Hugging Face.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM
from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb as hf_apply_rotary_pos_emb

from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    ops_config_scope,
)
from tests.models.tiny_configs import tiny_deepseek_v4_config as _tiny_config


def _dsv4_module():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_npu as gen
    else:
        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as gen
    return gen


def _dsv4_cls():
    return _dsv4_module().DeepseekV4ForCausalLM


def _build_ours(config: DeepseekV4Config, ops: SimpleNamespace | None = None):
    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return _dsv4_cls()(config)


def test_deepseek_v4_shared_mlp_passes_swiglu_limit():
    config = _tiny_config()
    model = _build_ours(config)
    shared = model.model.layers[0].mlp.shared_experts
    assert shared.limit == config.swiglu_limit
    captured: dict = {}

    def record(x, *args, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(x)

    shared.veomni_swiglu_mlp = record
    shared(torch.randn(2, 8, config.hidden_size))
    assert captured["swiglu_limit"] == config.swiglu_limit


def test_deepseek_v4_routers_use_fp32_projection_under_autocast():
    modeling = _dsv4_module()
    config = SimpleNamespace(
        num_experts_per_tok=2,
        num_local_experts=4,
        hidden_size=8,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        vocab_size=16,
    )
    topk_router = modeling.DeepseekV4TopKRouter(config).to(torch.bfloat16)
    hash_router = modeling.DeepseekV4HashRouter(config).to(torch.bfloat16)
    with torch.no_grad():
        topk_router.weight.copy_(torch.linspace(-0.5, 0.5, topk_router.weight.numel()).reshape_as(topk_router.weight))
        hash_router.weight.copy_(torch.linspace(0.5, -0.5, hash_router.weight.numel()).reshape_as(hash_router.weight))
    hidden_states = torch.linspace(-1.0, 1.0, 24, dtype=torch.bfloat16).reshape(1, 3, 8)
    input_ids = torch.tensor([[0, 1, 2]])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits, weights, indices = topk_router(hidden_states)
        hash_logits, _, _ = hash_router(hidden_states, input_ids)
    expected_logits = F.linear(hidden_states.reshape(-1, 8).float(), topk_router.weight.float())
    expected_hash_logits = F.linear(hidden_states.reshape(-1, 8).float(), hash_router.weight.float())
    expected_scores = expected_logits.sigmoid()
    expected_indices = torch.topk(expected_scores, 2, dim=-1, sorted=False).indices
    expected_weights = expected_scores.gather(1, expected_indices)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-20

    assert logits.dtype == torch.float32
    assert hash_logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(hash_logits, expected_hash_logits, rtol=0, atol=0)
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


def test_deepseek_v4_attention_preserves_q_norm_and_rope_dtype_modes():
    config = _tiny_config()
    model = _build_ours(config)
    attention = model.model.layers[0].self_attn.to(torch.bfloat16).eval()
    hidden_states = torch.randn(1, 7, config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0)
    rotary = model.model.rotary_emb.train()
    train_cos, train_sin = rotary(hidden_states, position_ids, layer_type="main")

    assert train_cos.dtype == hidden_states.dtype
    assert train_sin.dtype == hidden_states.dtype

    rotary.eval()
    cos, sin = rotary(hidden_states, position_ids, layer_type="main")
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32

    captured = {}

    def fake_attention(_module, query, _key, _value, _mask, **_kwargs):
        captured["query"] = query
        return torch.zeros_like(query.transpose(1, 2)), None

    attention.veomni_attn = fake_attention
    attention(
        hidden_states,
        position_embeddings={"main": (cos, sin), "compress": (cos, sin)},
        position_ids=position_ids,
        attention_mask=None,
    )

    q_residual = attention.q_a_norm(attention.q_a_proj(hidden_states))
    q_raw = attention.q_b_proj(q_residual).view(
        hidden_states.shape[0], hidden_states.shape[1], config.num_attention_heads, config.head_dim
    )
    rstd = torch.rsqrt(q_raw.float().square().mean(-1, keepdim=True) + config.rms_norm_eps)
    expected = q_raw * rstd.to(q_raw.dtype)
    expected = hf_apply_rotary_pos_emb(expected.transpose(1, 2), cos, sin)
    wrong_fp32_multiply = (q_raw.float() * rstd).to(q_raw.dtype)
    wrong_fp32_multiply = hf_apply_rotary_pos_emb(wrong_fp32_multiply.transpose(1, 2), cos, sin)

    torch.testing.assert_close(captured["query"], expected, rtol=0, atol=0)
    assert not torch.equal(captured["query"], wrong_fp32_multiply)


def test_deepseek_v4_experts_pass_merged_weights_dtype_and_swiglu_limit():
    config = _tiny_config()
    model = _build_ours(config)
    experts = model.model.layers[3].mlp.experts
    hidden_states = torch.linspace(-0.7, 0.8, steps=4 * config.hidden_size).reshape(4, config.hidden_size)
    selected_experts = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2]], dtype=torch.long)
    top_k_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.55, 0.45], [0.8, 0.2]],
        dtype=torch.float64,
    )
    captured = {}

    def record(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return torch.zeros_like(hidden_states)

    experts.veomni_moe = record
    actual = experts(hidden_states, selected_experts, top_k_weights)
    args = captured["args"]
    kwargs = captured["kwargs"]

    assert args[0] is hidden_states
    torch.testing.assert_close(args[1], top_k_weights.to(hidden_states.dtype), rtol=0, atol=0)
    assert args[2] is selected_experts
    assert args[3].numel() == 0
    assert args[4].numel() == 0
    assert args[5] is experts.down_proj
    assert args[6] is experts.gate_up_proj
    assert kwargs == {
        "num_experts": experts.num_experts,
        "swiglu_limit": config.swiglu_limit,
        "assume_distinct_experts": True,
    }
    torch.testing.assert_close(actual, torch.zeros_like(hidden_states), rtol=0, atol=0)


def test_deepseek_v4_sparse_moe_block_opts_topk_layers_into_tight_max_m_bound():
    config = _tiny_config()
    model = _build_ours(config)
    modeling = _dsv4_module()
    block = model.model.layers[3].mlp

    assert config.mlp_layer_types[3] == "moe"
    assert block.is_hash is False
    assert isinstance(block.gate, modeling.DeepseekV4TopKRouter)
    assert block.experts.assume_distinct_experts is True


def test_deepseek_v4_sparse_moe_block_keeps_hash_layers_conservative():
    config = _tiny_config()
    model = _build_ours(config)
    modeling = _dsv4_module()
    block = model.model.layers[0].mlp

    assert config.mlp_layer_types[0] == "hash_moe"
    assert block.is_hash is True
    assert isinstance(block.gate, modeling.DeepseekV4HashRouter)
    assert block.experts.assume_distinct_experts is False


@pytest.mark.parametrize("seq_len", [7, 19], ids=["before-hca-window", "compressed-topk"])
def test_deepseek_v4_eager_matches_hf(seq_len):
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFDeepseekV4ForCausalLM(config)
    ours = _build_ours(config)
    # Keep both attention and routing layouts exercised by the numerical comparison.
    assert config.layer_types == ["heavily_compressed_attention"] * 3 + ["compressed_sparse_attention"]
    assert config.mlp_layer_types == ["hash_moe"] * 3 + ["moe"]
    layer = ours.model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.variant == "deepseek_v4"
    assert layer.self_attn.q_b_norm.veomni_unweighted_rms_norm.variant == "unweighted"
    assert layer.veomni_mhc_post.variant == "post"
    assert layer.self_attn.veomni_dsa_attention.variant == "deepseek_v4"
    assert ours.model.layers[3].self_attn.compressor.indexer.veomni_dsa_indexer.variant == "deepseek_v4"
    assert ours.model.hc_head.veomni_mhc_head.variant == "head"
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, seq_len))
    # DSA + mHC ULP (~2e-7); not bitwise vs Hugging Face.
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)

    if seq_len >= config.compress_rates["heavily_compressed_attention"]:
        # Every attention layer must train its compressor, not just match an unused branch.
        for layer in ours.model.layers:
            grad = layer.self_attn.compressor.kv_proj.weight.grad
            assert grad is not None
            assert torch.isfinite(grad).all() and grad.count_nonzero() > 0


def test_deepseek_v4_eager_matches_hf_aux_loss():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFDeepseekV4ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 19))
    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    assert ours_out.aux_loss is not None
    assert hf_out.aux_loss is not None
    torch.testing.assert_close(ours_out.aux_loss, hf_out.aux_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=1e-6, rtol=1e-6)


class _FusedDsv4AttentionSpy:
    impl = "tilelang"

    def __call__(self, *args, **kwargs):
        raise AssertionError("fused DeepSeek-V4 attention should have raised before the kernel call")


def test_deepseek_v4_eager_dropout_train_eval_seed_and_grads():
    config = _tiny_config()
    config.attention_dropout = 0.5
    model = _build_ours(config)
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    model.train()
    torch.manual_seed(1)
    train_a = model(input_ids=input_ids, use_cache=False).logits
    torch.manual_seed(2)
    train_b = model(input_ids=input_ids, use_cache=False).logits
    assert not torch.allclose(train_a, train_b)
    model.eval()
    with torch.no_grad():
        torch.manual_seed(1)
        eval_a = model(input_ids=input_ids, use_cache=False).logits
        torch.manual_seed(2)
        eval_b = model(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(eval_a, eval_b)
    model.train()
    torch.manual_seed(3)
    logits = model(input_ids=input_ids, use_cache=False).logits
    grads = torch.autograd.grad(
        logits.sum(),
        [param for param in model.parameters() if param.requires_grad],
        allow_unused=True,
    )
    assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)


def test_deepseek_v4_eager_output_attentions_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFDeepseekV4ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    hf_out = hf(input_ids=input_ids, use_cache=False, output_attentions=True)
    ours_out = ours(input_ids=input_ids, use_cache=False, output_attentions=True)
    assert ours_out.attentions is not None and ours_out.attentions != ()
    assert len(ours_out.attentions) == config.num_hidden_layers
    assert len(hf_out.attentions) == config.num_hidden_layers
    for ours_weights, hf_weights in zip(ours_out.attentions, hf_out.attentions, strict=True):
        assert ours_weights.shape == hf_weights.shape
        torch.testing.assert_close(ours_weights, hf_weights, atol=1e-4, rtol=1e-4)


def test_deepseek_v4_fused_rejects_output_attentions():
    config = _tiny_config()
    model = _build_ours(config)
    model.model.layers[0].self_attn.veomni_dsa_attention = _FusedDsv4AttentionSpy()
    with pytest.raises(ValueError, match="output_attentions=True"):
        model(input_ids=torch.randint(3, config.vocab_size, (2, 8)), use_cache=False, output_attentions=True)


def test_deepseek_v4_packed_public_entry_validates_cu_seqlens():
    config = _tiny_config()
    model = _build_ours(config).eval()
    input_ids = torch.randint(3, config.vocab_size, (1, 8))
    with pytest.raises(ValueError, match="must span the full sequence"):
        model(input_ids=input_ids, cu_seq_lens_q=torch.tensor([0, 4], dtype=torch.int32), use_cache=False)


def test_deepseek_v4_packed_forward_uses_host_slices(monkeypatch):
    from veomni.models.transformers.deepseek_v4 import packed_utils as packed_utils_mod

    calls = {"from_cu": 0}
    real = packed_utils_mod.packed_sequence_slices_from_cu_seqlens

    def counting(cu_seqlens):
        calls["from_cu"] += 1
        return real(cu_seqlens)

    monkeypatch.setattr(packed_utils_mod, "packed_sequence_slices_from_cu_seqlens", counting)
    config = _tiny_config()
    model = _build_ours(config).eval()
    input_ids = torch.randint(3, config.vocab_size, (1, 8))
    position_ids = torch.cat([torch.arange(4), torch.arange(4)]).view(1, 8)
    with torch.no_grad():
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            packed_sequence_slices=((0, 4), (4, 8)),
            attention_mask_is_all_ones=True,
            cu_seq_lens_q=torch.tensor([0, 8], dtype=torch.int32),
            use_cache=False,
        )
    assert calls["from_cu"] == 0
