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

"""DeepSeek-V4 models_kernel consume tests.

Direct-import the generated class. Compare a toy CausalLM against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM

from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
)
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


def _tiny_config() -> DeepseekV4Config:
    """Official DeepseekV4Config fields, sized down for a toy.

    Omit schedule / router / mHC / RoPE / window fields so ``__post_init__``
    keeps the official defaults: 2× HCA bootstrap then CSA/HCA interleave,
    3× ``hash_moe`` then ``moe``, ``scoring_func="sqrtsoftplus"``,
    ``hc_mult=4``, ``sliding_window=128``, CSA=4 / HCA=128.
    Four layers therefore include a CSA indexer layer and one routed MoE layer.
    """
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=16,
        num_experts_per_tok=2,
        n_routed_experts=4,
        max_position_embeddings=64,
        o_groups=8,
        o_lora_rank=16,
        index_n_heads=4,
        index_head_dim=16,
        attn_implementation="eager",
        experts_implementation="eager",
    )


def _dsv4_module():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models_kernel.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_npu as gen
    else:
        from veomni.models_kernel.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as gen
    return gen


def _dsv4_cls():
    return _dsv4_module().DeepseekV4ForCausalLM


def _build_ours(config: DeepseekV4Config, ops: SimpleNamespace | None = None):
    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        return _dsv4_cls()(config)
    finally:
        set_ops_config(previous)


def test_deepseek_v4_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniOp)
    assert model.veomni_ce.impl == "eager"
    assert isinstance(model.veomni_lb, VeomniOp)
    assert model.config.layer_types == [
        "heavily_compressed_attention",
        "heavily_compressed_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    ]
    assert model.config.mlp_layer_types == ["hash_moe", "hash_moe", "hash_moe", "moe"]
    layer = model.model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer.input_layernorm.veomni_rms_norm.variant == "deepseek_v4"
    assert layer.attn_hc.veomni_mhc_pre.op == "mhc"
    assert layer.veomni_mhc_post.variant == "post"
    assert layer.self_attn.veomni_dsa_attention.op == "dsa_attention"
    assert layer.self_attn.veomni_dsa_attention.variant == "deepseek_v4"
    csa = model.model.layers[3].self_attn.compressor
    assert csa.indexer.veomni_dsa_indexer.op == "dsa_indexer"
    assert csa.indexer.veomni_dsa_indexer.variant == "deepseek_v4"
    assert layer.mlp.experts.veomni_moe.op == "moe_experts"
    assert layer.mlp.shared_experts.veomni_swiglu_mlp.op == "swiglu_mlp"
    assert layer.mlp.shared_experts.limit == model.config.swiglu_limit
    assert model.model.hc_head.veomni_mhc_head.variant == "head"


def test_deepseek_v4_shared_mlp_passes_swiglu_limit():
    config = _tiny_config()
    model = _build_ours(config)
    shared = model.model.layers[0].mlp.shared_experts
    captured: dict = {}

    def record(x, *args, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(x)

    shared.veomni_swiglu_mlp = record
    shared(torch.randn(2, 8, config.hidden_size))
    assert captured["swiglu_limit"] == config.swiglu_limit


def test_deepseek_v4_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_ops_config())
    chunk_cfg = eager_ops_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_ops_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"
    assert eager.model.layers[0].self_attn.veomni_dsa_attention.impl == "eager"


def test_deepseek_v4_rms_norms_use_selected_liger_impl(available_nvidia_ops):
    ops = eager_ops_config()
    ops.rms_norm_implementation = "liger_kernel"
    model = _build_ours(_tiny_config(), ops)
    layer = model.model.layers[0]

    assert layer.input_layernorm.veomni_rms_norm.variant == "deepseek_v4"
    assert layer.input_layernorm.veomni_rms_norm.impl == "liger_kernel"
    assert layer.self_attn.q_b_norm.veomni_unweighted_rms_norm.variant == "unweighted"
    assert layer.self_attn.q_b_norm.veomni_unweighted_rms_norm.impl == "liger_kernel"


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


def test_deepseek_v4_attention_preserves_q_norm_and_rope_dtype_modes(monkeypatch):
    modeling = _dsv4_module()
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

    monkeypatch.setattr(
        modeling, "ALL_ATTENTION_FUNCTIONS", SimpleNamespace(get_interface=lambda *_args: fake_attention)
    )
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
    expected = modeling.apply_rotary_pos_emb(expected.transpose(1, 2), cos, sin)
    wrong_fp32_multiply = (q_raw.float() * rstd).to(q_raw.dtype)
    wrong_fp32_multiply = modeling.apply_rotary_pos_emb(wrong_fp32_multiply.transpose(1, 2), cos, sin)

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
    assert kwargs == {"num_experts": experts.num_experts, "swiglu_limit": config.swiglu_limit}
    torch.testing.assert_close(actual, torch.zeros_like(hidden_states), rtol=0, atol=0)


def test_deepseek_v4_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFDeepseekV4ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_deepseek_v4_eager_matches_hf_aux_loss():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFDeepseekV4ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    assert ours_out.aux_loss is not None
    assert hf_out.aux_loss is not None
    torch.testing.assert_close(ours_out.aux_loss, hf_out.aux_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=1e-6, rtol=1e-6)
