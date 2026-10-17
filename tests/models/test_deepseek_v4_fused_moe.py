from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    MOE_TRITON_DEVICE_TYPES,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


def _deepseek_v4_experts_reference(
    *,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        gate_up = F.linear(hidden_states[token_idx], gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        current = F.linear(F.silu(gate) * up, down_proj[expert_idx])
        current = current * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, current.to(output.dtype))

    return output


@pytest.mark.parametrize(
    ("device_type", "expected_moe"),
    [
        (MOE_TRITON_DEVICE_TYPES[0], "fused_triton"),
        ("npu", "fused_npu"),
    ],
)
def test_deepseek_v4_test_overrides_follow_active_device(monkeypatch, device_type, expected_moe):
    from tests.tools import training_utils

    monkeypatch.setattr(training_utils, "get_device_type", lambda: device_type)
    overrides = training_utils.resolve_ops_overrides("deepseek_v4")

    assert "--model.ops_implementation.attn_implementation=eager" in overrides
    assert f"--model.ops_implementation.moe_implementation={expected_moe}" in overrides


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


def _run_deepseek_v4_ep_fixed_route_vjp(rank, world_size, init_file, seed):
    from tests.tools.training_utils import make_eager_ops_config
    from veomni.distributed.parallel_state import ParallelState, _init_parallel_state, use_parallel_state
    from veomni.models import build_foundation_model
    from veomni.ops.kernels.moe.group_gemm import group_gemm_fused_moe_forward

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


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or torch.version.hip is not None or get_torch_device().device_count() < 4,
    reason="DeepSeek V4 EP expert oracle requires four NVIDIA CUDA devices",
)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_deepseek_v4_expert_parallel_fixed_route_vjp_matches_fp32_reference(seed, tmp_path):
    """Check all-to-all output and four VJPs, unequal splits and duplicate slots."""
    torch.multiprocessing.spawn(
        _run_deepseek_v4_ep_fixed_route_vjp,
        args=(4, str(tmp_path / "rendezvous"), seed),
        nprocs=4,
        join=True,
    )


def test_deepseek_v4_routers_use_fp32_projection_under_autocast():
    config = SimpleNamespace(
        num_experts_per_tok=2,
        num_local_experts=4,
        hidden_size=8,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        vocab_size=16,
    )
    topk_router = dsv4.DeepseekV4TopKRouter(config).to(torch.bfloat16)
    hash_router = dsv4.DeepseekV4HashRouter(config).to(torch.bfloat16)
    # Standalone routers allocate empty weights; only the enclosing model
    # initializes them. Do not let allocator contents decide this comparison.
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for router in (topk_router, hash_router):
            router.weight.copy_(torch.randn(router.weight.shape, generator=generator, dtype=torch.bfloat16))
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


def test_deepseek_v4_weighted_rms_norm_matches_official_fp32_multiply_order():
    norm = dsv4.DeepseekV4RMSNorm(32).to(torch.bfloat16)
    generator = torch.Generator().manual_seed(42)
    hidden_states = torch.randn(2, 3, 32, generator=generator, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32, generator=generator, dtype=torch.bfloat16))

    normalized = hidden_states.float()
    variance = normalized.square().mean(-1, keepdim=True)
    normalized *= torch.rsqrt(variance + norm.variance_epsilon)
    expected = (norm.weight.float() * normalized).to(hidden_states.dtype)
    old_cast_order = norm.weight * normalized.to(hidden_states.dtype)

    actual = norm(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, old_cast_order)


def test_deepseek_v4_shared_expert_matches_official_clamped_fp32_swiglu():
    config = SimpleNamespace(
        hidden_size=2,
        intermediate_size=2,
        mlp_bias=False,
        hidden_act="silu",
        swiglu_limit=10.0,
    )
    mlp = dsv4.DeepseekV4MLP(config).to(torch.bfloat16)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.eye(2, dtype=torch.bfloat16).mul_(20))
        mlp.up_proj.weight.copy_(torch.eye(2, dtype=torch.bfloat16).mul_(20))
        mlp.down_proj.weight.copy_(torch.eye(2, dtype=torch.bfloat16))

    hidden_states = torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16)
    gate = mlp.gate_proj(hidden_states).float().clamp(max=config.swiglu_limit)
    up = (
        mlp.up_proj(hidden_states)
        .float()
        .clamp(
            min=-config.swiglu_limit,
            max=config.swiglu_limit,
        )
    )
    expected = mlp.down_proj((F.silu(gate) * up).to(hidden_states.dtype))
    unclamped = mlp.down_proj(
        (F.silu(mlp.gate_proj(hidden_states).float()) * mlp.up_proj(hidden_states).float()).to(hidden_states.dtype)
    )

    actual = mlp(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, unclamped)


def test_deepseek_v4_attention_matches_official_q_norm_and_rope_dtype_modes(monkeypatch):
    config = dsv4.DeepseekV4Config.from_pretrained("tests/toy_config/deepseek_v4_toy")
    torch.manual_seed(42)
    attention = dsv4.DeepseekV4Attention(config, layer_idx=0).to(torch.bfloat16).eval()
    hidden_states = torch.randn(1, 7, config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0)
    rotary = dsv4.DeepseekV4RotaryEmbedding(config).train()
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
        dsv4,
        "ALL_ATTENTION_FUNCTIONS",
        SimpleNamespace(get_interface=lambda *_args: fake_attention),
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
    expected = q_raw * torch.rsqrt(q_raw.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    expected = dsv4.apply_rotary_pos_emb(expected.transpose(1, 2), cos, sin)
    old_fp32_norm = attention.q_b_norm(q_raw.transpose(1, 2))
    old_fp32_norm = dsv4.apply_rotary_pos_emb(old_fp32_norm, cos, sin)

    torch.testing.assert_close(captured["query"], expected, rtol=0, atol=0)
    assert not torch.equal(captured["query"], old_fp32_norm)

    rope_dim = cos.shape[-1] * 2
    expected_rope = torch.view_as_real(
        torch.view_as_complex(expected[..., -rope_dim:].float().unflatten(-1, (-1, 2)))
        * torch.complex(cos, sin).unsqueeze(1)
    ).flatten(-2)
    actual_twice_rotated = dsv4.apply_rotary_pos_emb(expected, cos, sin)
    torch.testing.assert_close(
        actual_twice_rotated[..., -rope_dim:], expected_rope.to(torch.bfloat16), rtol=0, atol=2**-13
    )


def test_deepseek_v4_fused_moe_receives_merged_weights_and_swiglu_limit(monkeypatch):
    config = SimpleNamespace(
        num_local_experts=3,
        hidden_size=5,
        intermediate_size=7,
        hidden_act="silu",
        swiglu_limit=6.5,
    )
    experts = dsv4.DeepseekV4Experts(config)

    gate = torch.arange(
        config.num_local_experts * config.intermediate_size * config.hidden_size,
        dtype=torch.float32,
    ).reshape(config.num_local_experts, config.intermediate_size, config.hidden_size)
    gate = gate.mul_(0.01).add_(0.1)
    up = torch.arange(gate.numel(), dtype=torch.float32).reshape_as(gate).mul_(0.02).sub_(0.2)
    down = torch.arange(
        config.num_local_experts * config.hidden_size * config.intermediate_size,
        dtype=torch.float32,
    ).reshape(config.num_local_experts, config.hidden_size, config.intermediate_size)
    down = down.mul_(0.015).sub_(0.05)

    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.cat([gate, up], dim=1))
        experts.down_proj.copy_(down)

    hidden_states = torch.linspace(-0.7, 0.8, steps=4 * config.hidden_size).reshape(4, config.hidden_size)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [0, 2],
        ],
        dtype=torch.long,
    )
    top_k_weights = torch.tensor(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.55, 0.45],
            [0.8, 0.2],
        ],
        dtype=torch.float64,
    )

    class _FusedSlot:
        use_non_eager_impl = True

    captured = {}

    def fake_fused_moe_forward(
        *,
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        fc1_1_2_weight=None,
        swiglu_limit=None,
        assume_distinct_experts=False,
    ):
        captured.update(
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            fc1_1_weight=fc1_1_weight,
            fc1_2_weight=fc1_2_weight,
            fc2_weight=fc2_weight,
            fc1_1_2_weight=fc1_1_2_weight,
            swiglu_limit=swiglu_limit,
            assume_distinct_experts=assume_distinct_experts,
        )
        return _deepseek_v4_experts_reference(
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            gate_up_proj=fc1_1_2_weight,
            down_proj=fc2_weight,
            swiglu_limit=swiglu_limit,
        )

    monkeypatch.setattr(dsv4, "veomni_moe_experts_forward", _FusedSlot())
    monkeypatch.setattr(dsv4, "fused_moe_forward", fake_fused_moe_forward)

    actual = experts(hidden_states, selected_experts, top_k_weights)
    expected = _deepseek_v4_experts_reference(
        num_experts=config.num_local_experts,
        routing_weights=top_k_weights.to(hidden_states.dtype),
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        gate_up_proj=experts.gate_up_proj,
        down_proj=experts.down_proj,
        swiglu_limit=config.swiglu_limit,
    )

    assert captured["num_experts"] == config.num_local_experts
    assert captured["fc1_1_weight"] is None
    assert captured["fc1_2_weight"] is None
    assert captured["fc2_weight"] is experts.down_proj
    assert captured["fc1_1_2_weight"] is experts.gate_up_proj
    assert captured["swiglu_limit"] == config.swiglu_limit
    # A bare ``DeepseekV4Experts`` keeps the conservative default; only
    # ``DeepseekV4SparseMoeBlock`` opts the learned top-k layers into ``T``.
    assert captured["assume_distinct_experts"] is False
    assert captured["routing_weights"].dtype == hidden_states.dtype
    torch.testing.assert_close(captured["routing_weights"], top_k_weights.to(hidden_states.dtype), rtol=0, atol=0)
    torch.testing.assert_close(captured["fc1_1_2_weight"][:, : config.intermediate_size], gate, rtol=0, atol=0)
    torch.testing.assert_close(captured["fc1_1_2_weight"][:, config.intermediate_size :], up, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _sparse_moe_block_config(mlp_layer_types):
    return SimpleNamespace(
        mlp_layer_types=mlp_layer_types,
        num_experts_per_tok=2,
        num_local_experts=4,
        hidden_size=5,
        intermediate_size=7,
        swiglu_limit=6.5,
        hidden_act="silu",
        mlp_bias=False,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        vocab_size=16,
    )


def test_deepseek_v4_sparse_moe_block_opts_topk_layers_into_tight_max_m_bound():
    # Learned top-k routing (``torch.topk``) guarantees distinct experts per
    # token, so the block must opt its experts into the tight ``max_M = T``
    # grouped-GEMM bound (``assume_distinct_experts=True``).
    config = _sparse_moe_block_config(["moe", "hash_moe"])
    block = dsv4.DeepseekV4SparseMoeBlock(config, layer_idx=0)

    assert block.is_hash is False
    assert isinstance(block.gate, dsv4.DeepseekV4TopKRouter)
    assert block.experts.assume_distinct_experts is True


def test_deepseek_v4_sparse_moe_block_keeps_hash_layers_conservative():
    # Hash routing reads a frozen ``tid2eid`` table that may repeat an expert
    # within a token's slots, so the tight ``T`` bound is unsafe — the block
    # must leave the conservative ``T * top_k`` default in place.
    config = _sparse_moe_block_config(["moe", "hash_moe"])
    block = dsv4.DeepseekV4SparseMoeBlock(config, layer_idx=1)

    assert block.is_hash is True
    assert isinstance(block.gate, dsv4.DeepseekV4HashRouter)
    assert block.experts.assume_distinct_experts is False
