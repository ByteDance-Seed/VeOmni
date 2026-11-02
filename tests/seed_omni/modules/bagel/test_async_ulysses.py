"""Distributed numerical guards for BAGEL MoT async Ulysses and Flex-FA4."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import torchrun
from veomni.distributed.parallel_state import _init_parallel_state, use_parallel_state
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated import accelerated
from veomni.models.seed_omni.modules.bagel.qwen2_mot.configuration import BagelQwen2MoTConfig
from veomni.models.seed_omni.modules.bagel.qwen2_mot.masking import (
    build_mot_attention_metadata,
    pad_mot_attention_metadata,
)
from veomni.ops import VeomniOp
from veomni.ops.kernels.attention.standard import flex as flex_backend
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device
from veomni.utils.helper import enable_high_precision_for_bf16


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(nproc=4))


def _rms_norm_forward(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute reference FP32-accumulated RMSNorm."""
    invvar = torch.rsqrt(input_tensor.float().square().mean(dim=-1, keepdim=True) + eps)
    output = (input_tensor.float() * invvar * weight.float()).to(input_tensor.dtype)
    return output, invvar


def _apply_rotary(
    input_tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Compute reference rotate-half RoPE on ``[tokens, heads, head_dim]``."""
    first, second = input_tensor.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return input_tensor * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


def _reference_qkv_projection(
    hidden_states: torch.Tensor,
    route_0_indices: torch.Tensor,
    route_1_indices: torch.Tensor,
    position_cos: torch.Tensor,
    position_sin: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    *,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute full-sequence sequential two-route Q/K/V projection, RMSNorm, and RoPE."""

    def project(name: str, route: int, indices: torch.Tensor, num_heads: int) -> torch.Tensor:
        projected = F.linear(
            hidden_states.index_select(0, indices),
            parameters[f"{name}_weight_{route}"],
            parameters[f"{name}_bias_{route}"],
        )
        return projected.reshape(-1, num_heads, head_dim)

    def combine(name: str, num_heads: int, *, normalize: bool) -> torch.Tensor:
        output = hidden_states.new_zeros(hidden_states.shape[0], num_heads, head_dim)
        for route, indices in ((0, route_0_indices), (1, route_1_indices)):
            projected = project(name, route, indices, num_heads)
            if normalize:
                projected, _ = _rms_norm_forward(projected, parameters[f"{name}_norm_weight_{route}"], eps)
            output = output.index_copy(0, indices, projected)
        return output

    query = _apply_rotary(combine("q", num_query_heads, normalize=True), position_cos, position_sin)
    key = _apply_rotary(combine("k", num_key_value_heads, normalize=True), position_cos, position_sin)
    value = combine("v", num_key_value_heads, normalize=False)
    return query, key, value


def _reference_output_projection(
    hidden_states: torch.Tensor,
    route_0_indices: torch.Tensor,
    route_1_indices: torch.Tensor,
    parameters: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute full-sequence sequential two-route output projection."""
    flattened_states = hidden_states.flatten(1)
    output = flattened_states.new_zeros(flattened_states.shape)
    for route, indices in ((0, route_0_indices), (1, route_1_indices)):
        projected = F.linear(
            flattened_states.index_select(0, indices),
            parameters[f"proj_weight_{route}"],
            parameters[f"proj_bias_{route}"],
        )
        output = output.index_copy(0, indices, projected)
    return output


def _make_parameters(
    device: torch.device,
    *,
    hidden_size: int,
    query_size: int,
    key_value_size: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Allocate random two-route QKV, QK-norm, and output projection parameters."""
    shapes = {
        "q_weight_0": (query_size, hidden_size),
        "q_bias_0": (query_size,),
        "k_weight_0": (key_value_size, hidden_size),
        "k_bias_0": (key_value_size,),
        "v_weight_0": (key_value_size, hidden_size),
        "v_bias_0": (key_value_size,),
        "q_norm_weight_0": (head_dim,),
        "k_norm_weight_0": (head_dim,),
        "q_weight_1": (query_size, hidden_size),
        "q_bias_1": (query_size,),
        "k_weight_1": (key_value_size, hidden_size),
        "k_bias_1": (key_value_size,),
        "v_weight_1": (key_value_size, hidden_size),
        "v_bias_1": (key_value_size,),
        "q_norm_weight_1": (head_dim,),
        "k_norm_weight_1": (head_dim,),
        "proj_weight_0": (hidden_size, query_size),
        "proj_bias_0": (hidden_size,),
        "proj_weight_1": (hidden_size, query_size),
        "proj_bias_1": (hidden_size,),
    }
    parameters = {}
    for name, shape in shapes.items():
        value = torch.randn(shape, device=device, dtype=torch.float32)
        if "norm_weight" in name:
            value = value * 0.1 + 1
        parameters[name] = value.requires_grad_()
    return parameters


def _clone_parameters(parameters: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach and clone ``parameters`` into independent leaf tensors."""
    return {name: parameter.detach().clone().requires_grad_() for name, parameter in parameters.items()}


def _async_ulysses_mot_worker(num_key_value_heads: int, route_case: str) -> None:
    """Compare 4-rank ``async_ulysses_qkv`` / ``async_ulysses_o`` ``bagel`` ops against sequential reference."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4
    device = torch.device(f"{get_device_type()}:{rank}")
    group = dist.group.WORLD
    enable_high_precision_for_bf16()
    torch.manual_seed(4201)

    sequence_length = 8
    local_sequence_length = sequence_length // world_size
    hidden_size = 32
    num_query_heads = 8
    head_dim = hidden_size // num_query_heads
    query_size = num_query_heads * head_dim
    key_value_size = num_key_value_heads * head_dim
    eps = 1e-6

    full_input = torch.randn(sequence_length, hidden_size, device=device, dtype=torch.float32)
    dist.broadcast(full_input, src=0)
    angles = torch.randn(sequence_length, head_dim // 2, device=device, dtype=torch.float32)
    dist.broadcast(angles, src=0)
    position_cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    position_sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    full_output_gradient = torch.randn_like(full_input)
    dist.broadcast(full_output_gradient, src=0)

    parameter_seed = _make_parameters(
        device,
        hidden_size=hidden_size,
        query_size=query_size,
        key_value_size=key_value_size,
        head_dim=head_dim,
    )
    for parameter in parameter_seed.values():
        dist.broadcast(parameter, src=0)
    async_parameters = _clone_parameters(parameter_seed)
    reference_parameters = _clone_parameters(parameter_seed)

    start = rank * local_sequence_length
    stop = start + local_sequence_length
    local_input = full_input[start:stop].detach().clone().requires_grad_()
    reference_input = full_input.detach().clone().requires_grad_()
    token_types = torch.arange(sequence_length, device=device) % 2
    if route_case == "empty_gen":
        token_types.zero_()
    elif route_case == "rank_routes":
        token_types = torch.arange(sequence_length, device=device) // local_sequence_length % 2
    elif route_case == "padding":
        token_types[-1] = -1
    full_route_0_indices = torch.where(token_types == 0)[0]
    full_route_1_indices = torch.where(token_types == 1)[0]
    local_route_0_indices = torch.where(token_types[start:stop] == 0)[0]
    local_route_1_indices = torch.where(token_types[start:stop] == 1)[0]

    qkv_op = VeomniOp("async_ulysses_qkv", "bagel")
    o_op = VeomniOp("async_ulysses_o", "bagel")
    rms_op = VeomniOp("rms_norm", "standard", "eager")
    rope_op = VeomniOp("rope", "full", "eager")

    async_query, async_key, async_value = qkv_op(
        hidden_states=local_input,
        route_0_indices=local_route_0_indices,
        route_1_indices=local_route_1_indices,
        position_cos=position_cos[start:stop],
        position_sin=position_sin[start:stop],
        q_weight_0=async_parameters["q_weight_0"],
        q_bias_0=async_parameters["q_bias_0"],
        k_weight_0=async_parameters["k_weight_0"],
        k_bias_0=async_parameters["k_bias_0"],
        v_weight_0=async_parameters["v_weight_0"],
        v_bias_0=async_parameters["v_bias_0"],
        q_norm_weight_0=async_parameters["q_norm_weight_0"],
        k_norm_weight_0=async_parameters["k_norm_weight_0"],
        q_weight_1=async_parameters["q_weight_1"],
        q_bias_1=async_parameters["q_bias_1"],
        k_weight_1=async_parameters["k_weight_1"],
        k_bias_1=async_parameters["k_bias_1"],
        v_weight_1=async_parameters["v_weight_1"],
        v_bias_1=async_parameters["v_bias_1"],
        q_norm_weight_1=async_parameters["q_norm_weight_1"],
        k_norm_weight_1=async_parameters["k_norm_weight_1"],
        head_dim=head_dim,
        rms_norm_eps=eps,
        unpadded_dim_size=sequence_length,
        group=group,
        rms_norm=rms_op,
        rope=rope_op,
    )

    reference_query, reference_key, reference_value = _reference_qkv_projection(
        reference_input,
        full_route_0_indices,
        full_route_1_indices,
        position_cos,
        position_sin,
        reference_parameters,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        eps=eps,
    )
    local_query_heads = num_query_heads // world_size
    exchanged_key_value_heads = max(num_key_value_heads, world_size)
    local_key_value_heads = exchanged_key_value_heads // world_size
    torch.testing.assert_close(
        async_query,
        reference_query[:, rank * local_query_heads : (rank + 1) * local_query_heads],
    )
    repeated_reference_key = torch.repeat_interleave(
        reference_key,
        repeats=exchanged_key_value_heads // num_key_value_heads,
        dim=1,
    )
    repeated_reference_value = torch.repeat_interleave(
        reference_value,
        repeats=exchanged_key_value_heads // num_key_value_heads,
        dim=1,
    )
    torch.testing.assert_close(
        async_key,
        repeated_reference_key[:, rank * local_key_value_heads : (rank + 1) * local_key_value_heads],
    )
    torch.testing.assert_close(
        async_value,
        repeated_reference_value[:, rank * local_key_value_heads : (rank + 1) * local_key_value_heads],
    )

    local_gqa_repeats = local_query_heads // local_key_value_heads
    async_attention_output = async_query + torch.repeat_interleave(
        async_key + async_value,
        repeats=local_gqa_repeats,
        dim=1,
    )
    reference_attention_output = reference_query + torch.repeat_interleave(
        reference_key + reference_value,
        repeats=num_query_heads // num_key_value_heads,
        dim=1,
    )
    async_output = o_op(
        hidden_states=async_attention_output,
        route_0_indices=local_route_0_indices,
        route_1_indices=local_route_1_indices,
        proj_weight_0=async_parameters["proj_weight_0"],
        proj_bias_0=async_parameters["proj_bias_0"],
        proj_weight_1=async_parameters["proj_weight_1"],
        proj_bias_1=async_parameters["proj_bias_1"],
        group=group,
    )
    reference_output = _reference_output_projection(
        reference_attention_output,
        full_route_0_indices,
        full_route_1_indices,
        reference_parameters,
    )
    torch.testing.assert_close(async_output, reference_output[start:stop], atol=2e-5, rtol=2e-5)

    async_output.backward(full_output_gradient[start:stop])
    reference_output.backward(full_output_gradient)

    gathered_input_grads = [torch.empty_like(local_input.grad) for _ in range(world_size)]
    dist.all_gather(gathered_input_grads, local_input.grad)
    torch.testing.assert_close(torch.cat(gathered_input_grads), reference_input.grad, atol=3e-5, rtol=3e-5)
    for name, async_parameter in async_parameters.items():
        assert async_parameter.grad is not None, f"missing async gradient for {name}"
        assert reference_parameters[name].grad is not None, f"missing reference gradient for {name}"
        async_gradient = async_parameter.grad.detach().clone()
        dist.all_reduce(async_gradient, group=group)
        torch.testing.assert_close(
            async_gradient,
            reference_parameters[name].grad,
            atol=5e-5,
            rtol=5e-5,
            msg=f"gradient mismatch for {name}",
        )


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
@pytest.mark.parametrize(
    ("num_key_value_heads", "route_case"),
    [(2, "mixed"), (4, "empty_gen"), (4, "rank_routes"), (4, "padding")],
)
def test_async_ulysses_mot_matches_eager_two_route_projection_forward_backward(
    num_key_value_heads: int,
    route_case: str,
) -> None:
    """Verify two-route BAGEL async Ulysses forward and backward across 4 ranks."""
    torchrun(_async_ulysses_mot_worker, world_size=4, num_key_value_heads=num_key_value_heads, route_case=route_case)


def _bagel_fa4_async_worker(ops_impl: str, checkpointing: bool, fsdp: bool, backend: str, dtype: torch.dtype) -> None:
    """Compare synchronous and asynchronous Ulysses in ``BagelQwen2MoTAccelerated`` over 2 steps."""
    rank = dist.get_rank()
    device = torch.device(f"{get_device_type()}:{rank}")
    torch.manual_seed(6321)
    enable_high_precision_for_bf16()
    state = _init_parallel_state(dp_size=1, ulysses_size=4, name="bagel_sync_test")
    async_state = replace(state, async_enabled=True)

    # SP4 leaves seven Q heads and one KV head per rank, matching BAGEL's
    # non-power-of-two GQA geometry. Keep weights small while exercising FA4.
    config = BagelQwen2MoTConfig(
        vocab_size=128,
        hidden_size=896,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        bos_token_id=1,
        eos_token_id=2,
        ops_implementation={
            "attn_implementation": "flex_attention",
            "rms_norm_implementation": ops_impl,
            "rotary_pos_emb_implementation": ops_impl,
            "swiglu_mlp_implementation": "eager",
        },
    )
    reference = accelerated.BagelQwen2MoTAccelerated(config).to(device=device, dtype=dtype).train()
    candidate = copy.deepcopy(reference)
    if checkpointing:
        for model in (reference, candidate):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if fsdp:
        # FP32 parameter/gradient storage with BF16 forward matches the mixed
        # precision contract used by the production FSDP2 wrapper.
        for model in (reference, candidate):
            model.float()
            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            for layer in model.model.layers:
                fully_shard(layer, mesh=state.fsdp_mesh, mp_policy=policy)
            fully_shard(model.model, mesh=state.fsdp_mesh, mp_policy=policy)

    selected_backends = []
    original_resolve = flex_backend.resolve_flex_attention

    def capture_backend(*args, **kwargs):
        if backend == "TRITON":
            args[1]["BACKEND"] = "TRITON"
        interface = original_resolve(*args, **kwargs)
        selected_backends.append(interface)
        return interface

    flex_backend.resolve_flex_attention = capture_backend
    # Two independent documents, a full span, and two mutually invisible noise
    # spans exercise real MoT mask rules. The final SP shard includes padding.
    sequence_length, padded_length = 259, 260
    metadata = build_mot_attention_metadata(
        [[65, 64, 64], [33, 33]],
        [["causal", "noise", "noise"], ["full", "causal"]],
        device=device,
    )
    metadata = pad_mot_attention_metadata(metadata, padded_length=padded_length)
    hidden = torch.randn(padded_length, config.hidden_size, device=device, dtype=dtype)
    token_types = torch.zeros(padded_length, device=device, dtype=torch.long)
    token_types[65:193] = 1
    token_types[sequence_length:] = -1
    hidden[sequence_length:] = 0
    output_gradient = torch.randn_like(hidden)
    output_gradient[sequence_length:] = 0
    positions = torch.arange(padded_length, device=device)
    start, stop = rank * 65, (rank + 1) * 65
    local_und = torch.where(token_types[start:stop] == 0)[0]
    local_gen = torch.where(token_types[start:stop] == 1)[0]
    inputs = {
        "packed_position_ids": positions[start:stop],
        "packed_attention_metadata": metadata,
        "packed_und_token_indexes": local_und,
        "packed_gen_token_indexes": local_gen,
    }
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-3) for model in (reference, candidate)]
    try:
        # A second update catches weight-view and saved-tensor lifetime errors
        # that may be hidden by the first FSDP unshard/recompute pass.
        for _ in range(2):
            outputs, input_gradients = [], []
            for model, model_state in ((reference, state), (candidate, async_state)):
                local_input = hidden[start:stop].detach().clone().requires_grad_()
                with use_parallel_state(model_state):
                    output = model.model(packed_sequence=local_input, **inputs).packed_query_sequence
                    output.backward(output_gradient[start:stop])
                outputs.append(output.detach())
                input_gradients.append(local_input.grad)
            tolerance = 2e-4 if dtype == torch.float32 else 3e-2
            torch.testing.assert_close(outputs[1], outputs[0], atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(input_gradients[1], input_gradients[0], atol=tolerance, rtol=tolerance)
            for (name, expected), (actual_name, actual) in zip(
                reference.named_parameters(), candidate.named_parameters(), strict=True
            ):
                assert name == actual_name
                assert expected.grad is not None and actual.grad is not None, name
                expected_grad, actual_grad = expected.grad, actual.grad
                if fsdp:
                    expected_grad, actual_grad = expected_grad.to_local(), actual_grad.to_local()
                assert torch.isfinite(actual_grad).all(), name
                if dtype == torch.float32:
                    torch.testing.assert_close(
                        actual_grad,
                        expected_grad,
                        atol=2e-4,
                        rtol=2e-4,
                        msg=lambda message, name=name: f"{name}: {message}",
                    )
                else:
                    gradient_pairs = [(name, actual_grad, expected_grad)]
                    if "qkv_proj" in name and not fsdp:
                        split_sizes = reference.model.layers[0].self_attn.qkv_split_sizes
                        gradient_pairs = [
                            (f"{name}.{part}", actual_part, expected_part)
                            for part, actual_part, expected_part in zip(
                                ("q", "k", "v"),
                                actual_grad.split(split_sizes),
                                expected_grad.split(split_sizes),
                                strict=True,
                            )
                        ]
                    # Split and fused schedules can differ at BF16 precision.
                    # Check each Q/K/V block separately so a large Q gradient
                    # cannot hide an incorrect K/V gradient. The FP32 control
                    # above retains strict pointwise derivative checks.
                    for gradient_name, actual_part, expected_part in gradient_pairs:
                        error = (actual_part.float() - expected_part.float()).norm()
                        scale = expected_part.float().norm()
                        assert error <= 0.01 * scale + 1e-7, (
                            f"{gradient_name}: relative gradient L2 error {error / scale.clamp_min(1e-12)}"
                        )

            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        expected_backend = (
            flex_backend._flex_attention_fa4 if backend == "FLASH" else flex_backend._flex_attention_triton
        )
        assert selected_backends and all(selected is expected_backend for selected in selected_backends)
    finally:
        flex_backend.resolve_flex_attention = original_resolve


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or get_torch_device().device_count() < 4,
    reason="BAGEL FA4 + async Ulysses integration requires four SM90+ GPUs",
)
@pytest.mark.parametrize(
    ("ops_impl", "checkpointing", "fsdp", "backend", "dtype"),
    [
        pytest.param("eager", False, False, "TRITON", torch.float32, id="fp32-triton-control"),
        pytest.param("eager", False, False, "FLASH", torch.bfloat16, id="bf16-fa4-eager"),
        pytest.param("liger_kernel", True, False, "FLASH", torch.bfloat16, id="bf16-fa4-liger-checkpoint"),
        pytest.param("liger_kernel", True, True, "FLASH", torch.bfloat16, id="bf16-fa4-liger-checkpoint-fsdp2"),
    ],
)
def test_bagel_fa4_async_matches_sync_forward_backward_and_updates(
    ops_impl: str,
    checkpointing: bool,
    fsdp: bool,
    backend: str,
    dtype: torch.dtype,
) -> None:
    """Verify BAGEL async Ulysses matches synchronous Ulysses under Flex-FA4, checkpointing, and FSDP2."""
    device = torch.device(get_device_type())
    if backend == "FLASH" and (
        flex_backend.get_gpu_compute_capability(device) < 90 or not flex_backend._fa4._flash_attn_cute_available()
    ):
        pytest.skip("Flex-FA4 integration requires SM90+ and flash-attn-4")
    torchrun(
        _bagel_fa4_async_worker,
        world_size=4,
        ops_impl=ops_impl,
        checkpointing=checkpointing,
        fsdp=fsdp,
        backend=backend,
        dtype=dtype,
    )
