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

"""DeepSeek-V4 operator integration and Hugging Face parity tests.

Direct-import the generated class. Compare a toy CausalLM against Hugging Face.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM
from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb as hf_apply_rotary_pos_emb

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
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


@pytest.mark.parametrize("impl", ["liger_kernel", "npu"])
def test_deepseek_v4_rope_rejects_impls_without_a_v4_kernel(impl):
    ops = eager_ops_config()
    ops.rotary_pos_emb_implementation = impl
    with pytest.raises(KeyError, match=rf"'rope' variant='deepseek_v4' impl='{impl}'"):
        _build_ours(_tiny_config(), ops)


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


@pytest.mark.parametrize("impl", ["eager", "fused_triton", "deepgemm_fp8"])
def test_deepseek_v4_experts_pass_merged_weights_dtype_and_swiglu_limit(impl):
    """DeepSeek-V4 experts forward passes merged weights, raw routing weights, and ``swiglu_limit``."""
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
        """Record positional and keyword arguments passed to ``veomni_moe``."""
        captured["args"] = args
        captured["kwargs"] = kwargs
        return torch.zeros_like(hidden_states)

    experts.veomni_moe = Mock(impl=impl, side_effect=record)
    actual = experts(hidden_states, selected_experts, top_k_weights)
    args = captured["args"]
    kwargs = captured["kwargs"]

    assert args[0] is hidden_states
    assert args[1] is top_k_weights
    torch.testing.assert_close(args[1], top_k_weights, rtol=0, atol=0)
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


def test_deepseek_v4_experts_deepgemm_fp8_forward_backward():
    """Select FP8 through the public config and execute the real expert module."""
    from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device

    if not IS_CUDA_AVAILABLE or torch.version.hip is not None:
        pytest.skip("NVIDIA Hopper or Blackwell is required.")
    if get_torch_device().get_device_capability() not in ((9, 0), (10, 0)):
        pytest.skip("Requires SM90 or SM100.")
    pytest.importorskip("deep_gemm")
    from deep_gemm.testing import calc_diff

    from tests.ops.moe_experts.reference import standard_ep_reference
    from veomni.arguments import OpsImplementationConfig

    config = SimpleNamespace(
        num_local_experts=4,
        hidden_size=256,
        intermediate_size=128,
        hidden_act="silu",
        swiglu_limit=10.0,
    )
    device = get_device_type()
    ops = OpsImplementationConfig(moe_implementation="deepgemm_fp8", qat_implementation="none")
    with ops_config_scope(ops):
        experts = _dsv4_module().DeepseekV4Experts(config).to(device=device, dtype=torch.bfloat16)
    assert experts.veomni_moe.impl == "deepgemm_fp8"
    torch.manual_seed(91)
    with torch.no_grad():
        experts.gate_up_proj.normal_(std=0.03)
        experts.down_proj.normal_(std=0.03)
    x = torch.randn(131, config.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    routing = torch.rand(131, 2, device=device, requires_grad=True)
    indices = torch.randint(0, config.num_local_experts - 1, (131, 2), device=device)
    grad = torch.randn_like(x) * 0.1
    y = experts(x, indices, routing)
    actual = (y, *torch.autograd.grad(y, (x, routing, experts.gate_up_proj, experts.down_proj), grad))
    xr, rr, w1, w2 = (
        t.detach().clone().requires_grad_() for t in (x, routing, experts.gate_up_proj, experts.down_proj)
    )
    gate, up = w1.chunk(2, dim=1)
    yr = standard_ep_reference(xr, rr, indices, gate, up, w2, num_experts=config.num_local_experts, swiglu_limit=10.0)
    expected = (yr, *torch.autograd.grad(yr, (xr, rr, w1, w2), grad))
    for result, reference in zip(actual, expected):
        assert torch.isfinite(result).all()
        assert calc_diff(result, reference) < 0.004
    assert torch.count_nonzero(actual[-1][-1]) == 0
    assert torch.count_nonzero(actual[-2][-1]) == 0


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
    assert_eager_matches_hf(hf, ours, input_ids=input_ids, logits_equal=True)

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


class _StopForward(Exception):
    pass


class _Dsv4AttentionRequestSpy:
    """Stand-in ``dsa_attention`` op that records the request and stops the forward."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        raise _StopForward


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


def test_deepseek_v4_cached_decode_matches_hf():
    # A decode query sits past the cached rows, so this is what pins the
    # attention's ``query_offset`` and the compressors' decode candidates. The
    # prefill is longer than the sliding window and every compression rate.
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFDeepseekV4ForCausalLM(config).eval()
    ours = _build_ours(config).eval()
    ours.load_state_dict(hf.state_dict())
    input_ids = torch.randint(3, config.vocab_size, (2, 22))
    prefill, steps = input_ids[:, :19], input_ids[:, 19:]
    with torch.no_grad():
        hf_cache = hf(input_ids=prefill, use_cache=True).past_key_values
        ours_cache = ours(input_ids=prefill, use_cache=True).past_key_values
        for step in range(steps.shape[1]):
            token = steps[:, step : step + 1]
            hf_out = hf(input_ids=token, past_key_values=hf_cache, use_cache=True)
            ours_out = ours(input_ids=token, past_key_values=ours_cache, use_cache=True)
            torch.testing.assert_close(ours_out.logits, hf_out.logits, atol=1e-5, rtol=1e-5)
            hf_cache, ours_cache = hf_out.past_key_values, ours_out.past_key_values


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


def test_deepseek_v4_output_attentions_reaches_the_op():
    # Each implementation decides whether it can return weights; modeling only
    # has to pass the request through.
    config = _tiny_config()
    model = _build_ours(config)
    spy = _Dsv4AttentionRequestSpy()
    model.model.layers[0].self_attn.veomni_dsa_attention = spy
    with pytest.raises(_StopForward):
        model(input_ids=torch.randint(3, config.vocab_size, (2, 8)), use_cache=False, output_attentions=True)
    assert spy.kwargs["return_attn_weights"] is True


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


def test_deepseek_v4_duplicate_hash_routes_match_ep_dispatch_splits(monkeypatch):
    import veomni.distributed.moe.moe_layer as moe

    selected = torch.tensor([[0, 0], [0, 2], [3, 3]])
    mask = F.one_hot(selected, num_classes=4).permute(2, 1, 0)
    group = SimpleNamespace(size=lambda: 2)
    monkeypatch.setattr(moe.dist, "get_rank", lambda _group: 0)

    def gather(output, counts, group):
        output.copy_(counts.expand(2, -1))

    monkeypatch.setattr(moe.dist, "all_gather_into_tensor", gather)
    inputs, outputs, counts, totals = moe.preprocess(mask, 4, group)
    dispatched, _ = moe.permute(torch.ones(3, 2), mask.sum(dim=1))
    assert inputs == [2, 2]
    assert outputs == [2, 2]
    assert sum(inputs) == dispatched.shape[0] == 4
    torch.testing.assert_close(counts, torch.tensor([[2, 0], [2, 0]]))
    torch.testing.assert_close(totals, torch.tensor([4, 0]))


def _deepseek_v4_experts_reference(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    swiglu_limit: float | None,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, gate_up_proj[idx]).chunk(2, dim=-1)
        if swiglu_limit is not None:
            gate = gate.clamp(min=None, max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        current_hidden_states = F.silu(gate) * up
        current_hidden_states = F.linear(current_hidden_states, down_proj[idx])
        current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, current_hidden_states.to(output.dtype))
    return output


def _run_deepseek_v4_ep_fixed_route_vjp(rank, world_size, init_file, seed):
    import torch.distributed as dist

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.distributed.parallel_state import ParallelState, _init_parallel_state, use_parallel_state
    from veomni.models import build_foundation_model
    from veomni.ops.kernels.moe_experts.standard.triton import group_gemm_fused_moe_forward
    from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device

    device_type = get_device_type()
    get_torch_device().set_device(rank)
    dist.init_process_group(
        get_dist_comm_backend(), init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("highest")
    with use_parallel_state(ParallelState(dp_size=world_size, dp_shard_size=world_size)):
        model = build_foundation_model(
            config_path="tests/toy_config/deepseek_v4_toy",
            weights_path=None,
            torch_dtype="bfloat16",
            init_device=device_type,
            ops_implementation=make_eager_ops_config(moe_implementation="fused_triton"),
        ).train()
        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, "tid2eid"):
                    table = module.tid2eid
                    table.copy_(
                        (
                            torch.arange(table.shape[0], device=table.device)[:, None]
                            + torch.arange(table.shape[1], device=table.device)[None, :]
                        )
                        % module.num_experts
                    )
        expert = model.model.layers[3].mlp.experts
        captured = {}

        def capture(_module, args, kwargs, output):
            for i, name in enumerate(("hidden_states", "top_k_index", "top_k_weights")):
                captured[name] = (args[i] if len(args) > i else kwargs[name]).detach().clone()
            output.register_hook(lambda grad: captured.update(cotangent=grad.detach().clone()))

        handle = expert.register_forward_hook(capture, with_kwargs=True)
        ids = torch.arange(64, device=device_type).view(1, 64)
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        handle.remove()
    full_gate_up = expert.gate_up_proj.detach().clone()
    full_down = expert.down_proj.detach().clone()
    limit = expert.limit
    del model
    for value in (*captured.values(), full_gate_up, full_down):
        dist.broadcast(value, src=0)
    state = _init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        device_type=device_type,
        extra_parallel_sizes=(world_size,),
        name=None,
    )
    local_experts = full_gate_up.shape[0] // world_size
    expert_slice = slice(state.ep_rank * local_experts, (state.ep_rank + 1) * local_experts)
    offsets = (0, 9, 24, 41, 64)
    token_slice = slice(offsets[rank], offsets[rank + 1])
    for duplicate_slots in (False, True):
        indices = captured["top_k_index"].clone()
        if duplicate_slots:
            indices[::8, 1] = indices[::8, 0]
        route_bits = captured["top_k_weights"].to(torch.bfloat16)
        hidden = captured["hidden_states"][token_slice].detach().clone().requires_grad_()
        route = route_bits[token_slice].detach().clone().requires_grad_()
        gate_up = full_gate_up[expert_slice].detach().clone().requires_grad_()
        down = full_down[expert_slice].detach().clone().requires_grad_()
        with use_parallel_state(state):
            output = group_gemm_fused_moe_forward(
                num_experts=full_gate_up.shape[0],
                routing_weights=route,
                selected_experts=indices[token_slice],
                hidden_states=hidden,
                fc1_1_weight=None,
                fc1_2_weight=None,
                fc1_1_2_weight=gate_up,
                fc2_weight=down,
                swiglu_limit=limit,
            )
            output.backward(captured["cotangent"][token_slice])
        ref_hidden = captured["hidden_states"].float().detach().requires_grad_()
        ref_route = route_bits.float().detach().requires_grad_()
        ref_gate_up = full_gate_up.float().detach().requires_grad_()
        ref_down = full_down.float().detach().requires_grad_()
        ref_output = _deepseek_v4_experts_reference(
            num_experts=full_gate_up.shape[0],
            routing_weights=ref_route,
            selected_experts=indices,
            hidden_states=ref_hidden,
            gate_up_proj=ref_gate_up,
            down_proj=ref_down,
            swiglu_limit=limit,
        )
        ref_output.backward(captured["cotangent"].float())
        pairs = {
            "output": (output, ref_output[token_slice]),
            "hidden_grad": (hidden.grad, ref_hidden.grad[token_slice]),
            "route_grad": (route.grad, ref_route.grad[token_slice]),
            # Global reference VJP sums cotangents from EVERY source rank.
            # Direct EP has no FSDP gradient averaging to undo here.
            "gate_up_grad": (gate_up.grad, ref_gate_up.grad[expert_slice]),
            "down_grad": (down.grad, ref_down.grad[expert_slice]),
        }
        for name, (actual, expected) in pairs.items():
            context = (seed, rank, duplicate_slots, name)
            assert actual is not None and expected is not None, context
            assert actual.shape == expected.shape, context
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), context
            delta = actual.double() - expected.double()
            relative_l2 = delta.norm() / expected.double().norm().clamp_min(1e-30)
            # Same multi-stage BF16 rounding budget as the non-EP oracle;
            # discovery seed0 is excluded from these held-out cases.
            eps = torch.finfo(torch.bfloat16).eps
            assert relative_l2 <= 2 * eps, (context, relative_l2.item())
            assert delta.abs().max() <= 4 * eps * expected.double().abs().max(), context
        dist.barrier()
    dist.destroy_process_group()


_EP_ORACLE_LAUNCH = LaunchSpec(nproc=4)


def _has_four_nvidia_gpus() -> bool:
    from veomni.utils.device import IS_CUDA_AVAILABLE

    return bool(IS_CUDA_AVAILABLE and torch.version.hip is None and runner_device_count() >= _EP_ORACLE_LAUNCH.nproc)


@pytest.mark.skipif(
    not _has_four_nvidia_gpus(),
    reason="DeepSeek V4 EP expert oracle requires four NVIDIA CUDA devices",
)
@pytest.mark.launch_plan(provider=_EP_ORACLE_LAUNCH)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_deepseek_v4_expert_parallel_fixed_route_vjp_matches_fp32_reference(seed, tmp_path):
    """Check all-to-all output and four VJPs, unequal splits and duplicate slots."""
    torch.multiprocessing.spawn(
        _run_deepseek_v4_ep_fixed_route_vjp,
        args=(_EP_ORACLE_LAUNCH.nproc, str(tmp_path / "rendezvous"), seed),
        nprocs=_EP_ORACLE_LAUNCH.nproc,
        join=True,
    )


@pytest.mark.parametrize("num_mtp_layers", [1, 3])
def test_deepseek_v4_dspark_mtp_forward_backward_and_packed_isolation(num_mtp_layers: int):
    torch.manual_seed(0)
    config = _tiny_config()
    config.dspark_block_size = 5
    config.num_nextn_predict_layers = num_mtp_layers
    config.dspark_target_layer_ids = [1, 2, 3]
    config.dspark_markov_rank = 8
    config.dspark_n_routed_experts = 2
    config.dspark_num_experts_per_tok = 1
    config.mtp_loss_coef = 0.5

    model = _build_ours(config).train()
    assert len(model.model.mtp) == num_mtp_layers
    assert "DeepseekV4DSparkBlock" in model._no_split_modules
    assert not model._keys_to_ignore_on_load_unexpected
    assert hasattr(model.model.mtp[0], "main_proj")
    assert hasattr(model.model.mtp[-1], "markov_head")
    assert hasattr(model.model.mtp[-1], "hc_head")
    assert not model.model.mtp[-1].confidence_head.proj.weight.requires_grad
    assert model.model.mtp[0].mlp.experts.num_experts == 2

    from veomni.optim.muon import split_muon_adamw_params

    _, _, muon_names, adamw_names = split_muon_adamw_params(model)
    assert any("markov_head.embed.weight" in n for n in adamw_names)
    assert any("markov_head.head.weight" in n for n in adamw_names)
    assert not any("markov_head" in n for n in muon_names)

    # Without return_mtp_outputs=True, DeepseekV4Model.forward skips running MTP stages.
    raw_model_out = model.model(input_ids=torch.randint(3, config.vocab_size, (1, 4)), use_cache=False)
    assert raw_model_out.mtp_hidden_states is None
    assert raw_model_out.mtp_markov_embed is None
    assert raw_model_out.mtp_markov_head_weight is None
    assert raw_model_out.mtp_markov_bias is None

    # With return_mtp_outputs=True, DeepseekV4Model.forward returns low-rank mtp_markov_embed
    # and mtp_markov_head_weight without materializing a full [B, S, V] mtp_markov_bias.
    with torch.no_grad():
        mtp_model_out = model.model(
            input_ids=torch.randint(3, config.vocab_size, (1, 4)),
            return_mtp_outputs=True,
            use_cache=False,
        )
    assert mtp_model_out.mtp_markov_embed is not None
    assert mtp_model_out.mtp_markov_embed.shape == (1, 4, config.dspark_markov_rank)
    assert mtp_model_out.mtp_markov_head_weight is not None
    assert mtp_model_out.mtp_markov_head_weight.shape == (config.vocab_size, config.dspark_markov_rank)
    assert mtp_model_out.mtp_markov_bias is None

    # Two packed samples of length 4 each: position_ids restarts at 0 at index 4.
    input_ids = torch.randint(3, config.vocab_size, (1, 8))
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)
    labels = input_ids.clone()
    labels[0, 4] = -100  # packed sample boundary

    out = model(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        packed_sequence_slices=((0, 4), (4, 8)),
        attention_mask_is_all_ones=True,
        use_cache=False,
    )
    assert out.loss is not None
    assert out.aux_metrics is not None and "mtp_loss" in out.aux_metrics
    mtp_loss = out.aux_metrics["mtp_loss"]
    assert torch.isfinite(mtp_loss) and mtp_loss.item() > 0

    # Compare against coef=0.0 baseline to verify loss == lm_loss + mtp_loss_coef * mtp_loss.
    model.config.mtp_loss_coef = 0.0
    with torch.no_grad():
        base_out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            packed_sequence_slices=((0, 4), (4, 8)),
            attention_mask_is_all_ones=True,
            use_cache=False,
        )
    model.config.mtp_loss_coef = 0.5
    torch.testing.assert_close(out.loss.detach(), base_out.loss + 0.5 * mtp_loss, atol=1e-5, rtol=1e-5)

    out.loss.backward()
    for name, param in [
        ("mtp.0.main_proj", model.model.mtp[0].main_proj.weight),
        ("mtp.0.self_attn.kv_proj", model.model.mtp[0].self_attn.kv_proj.weight),
        ("mtp.0.mlp.experts.down_proj", model.model.mtp[0].mlp.experts.down_proj),
        ("mtp.-1.hc_head.hc_fn", model.model.mtp[-1].hc_head.hc_fn),
        ("mtp.-1.markov_head.embed", model.model.mtp[-1].markov_head.embed.weight),
        ("mtp.-1.markov_head.head", model.model.mtp[-1].markov_head.head.weight),
        ("lm_head", model.lm_head.weight),
    ]:
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all() and param.grad.count_nonzero() > 0, f"Zero/non-finite grad for {name}"


def test_deepseek_v4_freeze_mtp_disables_mtp_grads_and_loss():
    torch.manual_seed(0)
    config = _tiny_config()
    config.dspark_block_size = 5
    config.num_nextn_predict_layers = 1
    config.freeze_mtp = True

    model = _build_ours(config).train()
    assert len(model.model.mtp) == 1
    assert all(not p.requires_grad for p in model.model.mtp.parameters())

    input_ids = torch.randint(3, config.vocab_size, (1, 6))
    out = model(input_ids=input_ids, labels=input_ids.clone(), use_cache=False)
    assert out.aux_metrics is None or "mtp_loss" not in out.aux_metrics


def test_deepseek_v4_dspark_packed_samples_do_not_attend_across_boundary():
    torch.manual_seed(0)
    config = _tiny_config()
    config.dspark_block_size = 5
    config.num_nextn_predict_layers = 2
    config.dspark_target_layer_ids = [1, 2, 3]
    config.dspark_markov_rank = 8

    model = _build_ours(config).eval()
    sample_a1 = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    sample_a2 = torch.tensor([[20, 21, 22, 23]], dtype=torch.long)
    sample_b = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)

    with torch.no_grad():
        out_1 = model.model(
            input_ids=torch.cat([sample_a1, sample_b], dim=1),
            position_ids=position_ids,
            packed_sequence_slices=((0, 4), (4, 8)),
            attention_mask_is_all_ones=True,
            return_mtp_outputs=True,
            use_cache=False,
        )
        out_2 = model.model(
            input_ids=torch.cat([sample_a2, sample_b], dim=1),
            position_ids=position_ids,
            packed_sequence_slices=((0, 4), (4, 8)),
            attention_mask_is_all_ones=True,
            return_mtp_outputs=True,
            use_cache=False,
        )
    assert out_1.mtp_hidden_states is not None and out_2.mtp_hidden_states is not None
    assert not torch.allclose(out_1.mtp_hidden_states[:, :4], out_2.mtp_hidden_states[:, :4])
    torch.testing.assert_close(out_1.mtp_hidden_states[:, 4:], out_2.mtp_hidden_states[:, 4:], atol=1e-6, rtol=1e-6)


def test_deepseek_v4_dspark_batch_gt_1_and_output_router_logits():
    torch.manual_seed(0)
    config = _tiny_config()
    config.dspark_block_size = 5
    config.num_nextn_predict_layers = 2
    config.dspark_target_layer_ids = [1, 2, 3]
    config.dspark_markov_rank = 8
    config.n_routed_experts = 8
    config.dspark_n_routed_experts = 4
    config.dspark_num_experts_per_tok = 1

    model = _build_ours(config).train()
    input_ids = torch.randint(3, config.vocab_size, (2, 6))
    labels = input_ids.clone()

    # position_ids=None with batch_size=2 exercises [1, S] broadcast position_ids in _packed_draft_attention,
    # and output_router_logits=True exercises mixed expert counts between backbone (8) and MTP (4).
    out = model(
        input_ids=input_ids,
        labels=labels,
        output_router_logits=True,
        use_cache=False,
    )
    assert out.loss is not None
    assert out.aux_loss is not None and torch.isfinite(out.aux_loss)
    assert out.aux_metrics is not None and "mtp_loss" in out.aux_metrics
    out.loss.backward()
    assert model.model.mtp[0].main_proj.weight.grad is not None


def test_deepseek_v4_dspark_left_padding_mask_excludes_masked_tokens():
    torch.manual_seed(0)
    config = _tiny_config()
    config.compress_rates = {"compressed_sparse_attention": 8, "heavily_compressed_attention": 16}
    config.dspark_block_size = 5
    config.num_nextn_predict_layers = 2
    config.dspark_target_layer_ids = [1, 2, 3]
    config.dspark_markov_rank = 8

    model = _build_ours(config).train()
    attention_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
    input_ids_a = torch.tensor([[4, 10, 11, 12]], dtype=torch.long)
    input_ids_b = torch.tensor([[99, 10, 11, 12]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 11, 12]], dtype=torch.long)

    with torch.no_grad():
        out_a = model(
            input_ids=input_ids_a,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        out_b = model(
            input_ids=input_ids_b,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
    assert out_a.aux_metrics is not None and out_b.aux_metrics is not None
    torch.testing.assert_close(out_a.aux_metrics["mtp_loss"], out_b.aux_metrics["mtp_loss"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_a.loss, out_b.loss, atol=1e-6, rtol=1e-6)


def test_deepseek_v4_dspark_packed_unpacked_parity_beyond_sliding_window():
    torch.manual_seed(0)
    config = _tiny_config()
    config.sliding_window = 4
    config.dspark_block_size = 5
    config.num_nextn_predict_layers = 2
    config.dspark_target_layer_ids = [1, 2, 3]
    config.dspark_markov_rank = 8

    model = _build_ours(config).eval()
    input_ids = torch.randint(3, config.vocab_size, (1, 10))
    position_ids = torch.arange(10, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()

    with torch.no_grad():
        unpacked_out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            return_mtp_outputs=True,
            use_cache=False,
        )
        packed_out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            packed_sequence_slices=((0, 10),),
            attention_mask_is_all_ones=True,
            return_mtp_outputs=True,
            use_cache=False,
        )
    assert unpacked_out.aux_metrics is not None and packed_out.aux_metrics is not None
    torch.testing.assert_close(
        unpacked_out.aux_metrics["mtp_loss"],
        packed_out.aux_metrics["mtp_loss"],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(unpacked_out.loss, packed_out.loss, atol=1e-5, rtol=1e-5)


def test_deepseek_v4_dspark_presliced_shift_labels_with_logits_to_keep():
    torch.manual_seed(0)
    config = _tiny_config()
    config.dspark_block_size = 5
    config.num_nextn_predict_layers = 1
    config.dspark_target_layer_ids = [1, 2, 3]
    config.dspark_markov_rank = 8

    model = _build_ours(config).train()
    input_ids = torch.randint(3, config.vocab_size, (1, 6))
    labels = input_ids.clone()
    logits_to_keep = torch.tensor([1, 3, 4], dtype=torch.long)
    shift_labels = torch.tensor([[5, 6, 7]], dtype=torch.long)

    out = model(
        input_ids=input_ids,
        labels=labels,
        shift_labels=shift_labels,
        logits_to_keep=logits_to_keep,
        use_cache=False,
    )
    assert out.loss is not None
    assert out.aux_metrics is not None and "mtp_loss" in out.aux_metrics
    assert torch.isfinite(out.aux_metrics["mtp_loss"])

    # With logits_to_keep=1, a valid 1-token shift_labels, and an all-ones 2D attention_mask,
    # the final row keeps its supervision and matches the attention_mask=None run.
    model.zero_grad(set_to_none=True)
    one_tok_shift = torch.tensor([[3]], dtype=torch.long)
    all_ones_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        no_mask_out = model(
            input_ids=input_ids,
            labels=labels,
            shift_labels=one_tok_shift,
            logits_to_keep=1,
            use_cache=False,
        )
    masked_out = model(
        input_ids=input_ids,
        attention_mask=all_ones_mask,
        labels=labels,
        shift_labels=one_tok_shift,
        logits_to_keep=1,
        use_cache=False,
    )
    assert masked_out.aux_metrics is not None and no_mask_out.aux_metrics is not None
    assert masked_out.aux_metrics["mtp_loss"].item() > 0
    torch.testing.assert_close(
        masked_out.aux_metrics["mtp_loss"],
        no_mask_out.aux_metrics["mtp_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    masked_out.loss.backward()
    assert model.model.mtp[0].main_proj.weight.grad is not None
    assert model.model.mtp[0].main_proj.weight.grad.count_nonzero() > 0


def test_deepseek_v4_dspark_mtp_loss_coef_zero_ddp_two_iterations(monkeypatch):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from tests.tools.launch_utils import find_free_port

    already_init = dist.is_available() and dist.is_initialized()
    if not already_init:
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(find_free_port()))
        dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        # Case 1: mtp_loss_coef=0.0 configured at model construction time freezes self.mtp
        # so DDP (find_unused_parameters=False) runs multiple iterations cleanly.
        config_init_zero = _tiny_config()
        config_init_zero.layer_types = ["sliding_attention"] * config_init_zero.num_hidden_layers
        config_init_zero.dspark_block_size = 5
        config_init_zero.num_nextn_predict_layers = 1
        config_init_zero.dspark_target_layer_ids = [1, 2, 3]
        config_init_zero.dspark_markov_rank = 8
        config_init_zero.mtp_loss_coef = 0.0

        model_init_zero = _build_ours(config_init_zero).train()
        assert len(model_init_zero.model.mtp) == 1
        assert all(not p.requires_grad for p in model_init_zero.model.mtp.parameters())

        ddp_init_zero = DDP(model_init_zero, find_unused_parameters=False)
        input_ids = torch.randint(3, config_init_zero.vocab_size, (1, 6))
        labels = input_ids.clone()
        for _ in range(2):
            ddp_init_zero.zero_grad(set_to_none=True)
            out = ddp_init_zero(input_ids=input_ids, labels=labels, use_cache=False)
            assert out.loss is not None
            assert out.aux_metrics is None or "mtp_loss" not in out.aux_metrics
            out.loss.backward()

        # Case 2: mtp_loss_coef mutated to 0.0 after DDP wrap still completes two DDP iterations.
        config_runtime_zero = _tiny_config()
        config_runtime_zero.layer_types = ["sliding_attention"] * config_runtime_zero.num_hidden_layers
        config_runtime_zero.dspark_block_size = 5
        config_runtime_zero.num_nextn_predict_layers = 1
        config_runtime_zero.dspark_target_layer_ids = [1, 2, 3]
        config_runtime_zero.dspark_markov_rank = 8
        config_runtime_zero.mtp_loss_coef = 1.0

        model_runtime_zero = _build_ours(config_runtime_zero).train()
        ddp_runtime_zero = DDP(model_runtime_zero, find_unused_parameters=False)
        model_runtime_zero.config.mtp_loss_coef = 0.0
        for _ in range(2):
            ddp_runtime_zero.zero_grad(set_to_none=True)
            out = ddp_runtime_zero(input_ids=input_ids, labels=labels, use_cache=False)
            assert out.loss is not None
            assert out.aux_metrics is None or "mtp_loss" not in out.aux_metrics
            out.loss.backward()

        # Case 3: Under FSDP2+EP, MTP dense and expert parameters live as DTensors on
        # different meshes; skipping MTP with return_mtp_outputs=False must not sum
        # DTensors across meshes.
        from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard

        mesh_1d = DeviceMesh("cpu", [0], mesh_dim_names=("dp_shard",))
        mesh_2d = DeviceMesh("cpu", [[0]], mesh_dim_names=("ep_fsdp", "ep"))
        dtensor_model = _build_ours(config_runtime_zero).train()
        dtensor_model.config.mtp_loss_coef = 1.0
        dtensor_model.model.mtp[0].main_proj.weight = torch.nn.Parameter(
            DTensor.from_local(dtensor_model.model.mtp[0].main_proj.weight.detach(), mesh_1d, [Shard(0)])
        )
        dtensor_model.model.mtp[0].mlp.experts.gate_up_proj = torch.nn.Parameter(
            DTensor.from_local(
                dtensor_model.model.mtp[0].mlp.experts.gate_up_proj.detach(),
                mesh_2d,
                [Replicate(), Shard(0)],
            )
        )
        out_dtensor = dtensor_model(
            input_ids=input_ids,
            labels=labels,
            return_mtp_outputs=False,
            use_cache=False,
        )
        assert out_dtensor.loss is not None
        assert not isinstance(out_dtensor.loss, DTensor)
        out_dtensor.loss.backward()
    finally:
        if not already_init and dist.is_initialized():
            dist.destroy_process_group()
