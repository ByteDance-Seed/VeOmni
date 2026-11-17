"""Accelerated BAGEL Qwen2-MoT alignment: single-GPU eager parity and 4-GPU FSDP2 + Async Ulysses."""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.seed_omni.modules.bagel.helpers import (
    ALIGN_ATOL,
    ALIGN_GRAD_ATOL,
    ALIGN_GRAD_RTOL,
    ALIGN_RTOL,
    build_toy_conversation,
    clone_conversation,
    config_cls,
    conversation_ce_loss,
    conversation_mse_loss,
    native_model_cls,
    run_accelerated_mot,
    run_eager_mot,
    tiny_align_qwen2_cfg,
)
from tests.tools.launch_utils import torchrun
from veomni.distributed.parallel_state import _init_parallel_state, use_parallel_state
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated import accelerated
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated.accelerated import BagelQwen2MoTAccelerated
from veomni.models.seed_omni.modules.bagel.qwen2_mot.configuration import BagelQwen2MoTConfig
from veomni.models.seed_omni.modules.bagel.qwen2_mot.masking import (
    build_mot_attention_metadata,
    pad_mot_attention_metadata,
)
from veomni.ops.kernels.attention.standard import flex as flex_backend
from veomni.ops.kernels.attention.standard.magi import _kernel as magi_kernel
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device
from veomni.utils.helper import enable_high_precision_for_bf16


_PATCH_LATENT_DIM = 4
_MAGI_IMPLEMENTATION = "veomni_magi_attention"
_FLEX_IMPLEMENTATION = "veomni_flex_attention"


def _is_magi_ffa_available() -> bool:
    """Return whether a runnable MagiAttention CUTE DSL/JIT backend is installed on this GPU."""
    if not IS_CUDA_AVAILABLE or importlib.util.find_spec("magi_attention") is None:
        return False

    device = torch.device(get_device_type())
    return magi_kernel.get_kernel_mode(device) == magi_kernel.KERNEL_CUTE_JIT and (
        importlib.util.find_spec("flash_attn_cute") is not None
    )


_MAGI_FFA_AVAILABLE = _is_magi_ffa_available()
_MAGI_FFA_REASON = "MagiAttention alignment requires an NVIDIA SM100+ GPU with the CUTE DSL/JIT backend"


def _build_aligned_models(device: torch.device, dtype: torch.dtype, *, attn_implementation: str):
    torch.manual_seed(29)
    cfg = tiny_align_qwen2_cfg()
    cfg.pop("ops_implementation", None)
    eager_cfg = config_cls("bagel_qwen2_mot")(**cfg, attn_implementation="sdpa")
    accel_cfg = config_cls("bagel_qwen2_mot")(**cfg, attn_implementation=attn_implementation)
    eager = native_model_cls("bagel_qwen2_mot")(eager_cfg).to(device=device, dtype=dtype).train()
    accel = BagelQwen2MoTAccelerated(accel_cfg).to(device=device, dtype=dtype).train()
    accel.load_state_dict(copy.deepcopy(eager.state_dict()))
    return eager, accel


def _assert_accelerated_matches_eager(attn_implementation: str) -> None:
    """Verify ``BagelQwen2MoTAccelerated`` matches eager SDPA outputs, CE/MSE losses, and gradients on mixed data."""
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    eager, accel = _build_aligned_models(device, dtype, attn_implementation=attn_implementation)
    hidden_size = int(eager.config.hidden_size)
    vocab_size = int(eager.config.vocab_size)
    conversation = build_toy_conversation(
        "mixed",
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        patch_latent_dim=_PATCH_LATENT_DIM,
        device=device,
        dtype=dtype,
    )
    eager_conversation = clone_conversation(conversation)
    accel_conversation = clone_conversation(conversation)
    lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False).to(device=device, dtype=torch.float32)
    llm2vae = torch.nn.Linear(hidden_size, _PATCH_LATENT_DIM, bias=False).to(device=device, dtype=torch.float32)

    eager_hidden = run_eager_mot(eager, eager_conversation)
    accel_hidden = run_accelerated_mot(accel, accel_conversation)
    assert torch.isfinite(eager_hidden).all() and torch.isfinite(accel_hidden).all()
    assert eager_hidden.shape == accel_hidden.shape

    eager_ce = conversation_ce_loss(eager_conversation, lm_head)
    accel_ce = conversation_ce_loss(accel_conversation, lm_head)
    eager_mse = conversation_mse_loss(eager_conversation, llm2vae)
    accel_mse = conversation_mse_loss(accel_conversation, llm2vae)
    assert eager_ce is not None and accel_ce is not None
    assert eager_mse is not None and accel_mse is not None
    torch.testing.assert_close(accel_ce, eager_ce, atol=ALIGN_ATOL, rtol=ALIGN_RTOL)
    torch.testing.assert_close(accel_mse, eager_mse, atol=ALIGN_ATOL, rtol=ALIGN_RTOL)

    eager_loss = eager_hidden.float().square().mean() + eager_ce + eager_mse
    accel_loss = accel_hidden.float().square().mean() + accel_ce + accel_mse
    eager_loss.backward()
    accel_loss.backward()

    for name in (
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.norm.weight",
    ):
        eager_grad = dict(eager.named_parameters())[name].grad
        accel_grad = dict(accel.named_parameters())[name].grad
        assert eager_grad is not None and accel_grad is not None
        torch.testing.assert_close(
            accel_grad,
            eager_grad,
            atol=ALIGN_GRAD_ATOL,
            rtol=ALIGN_GRAD_RTOL,
            msg=f"gradient mismatch for {name}",
        )


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=1))
@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FlexAttention alignment requires CUDA")
def test_accelerated_ce_mse_matches_eager_on_toy_data() -> None:
    _assert_accelerated_matches_eager(_FLEX_IMPLEMENTATION)


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=1))
@pytest.mark.skipif(not _MAGI_FFA_AVAILABLE, reason=_MAGI_FFA_REASON)
def test_magi_ce_mse_matches_eager_on_toy_data() -> None:
    _assert_accelerated_matches_eager(_MAGI_IMPLEMENTATION)


def _bagel_fa4_async_worker() -> None:
    """Compare synchronous and asynchronous Ulysses in ``BagelQwen2MoTAccelerated`` under FSDP2 + Flex-FA4 over 2 steps."""
    rank = dist.get_rank()
    device = torch.device(f"{get_device_type()}:{rank}")
    dtype = torch.bfloat16
    torch.manual_seed(6321)
    enable_high_precision_for_bf16()
    state = _init_parallel_state(dp_size=1, ulysses_size=4, name="bagel_sync_test")
    async_state = replace(state, async_enabled=True)

    # SP4 leaves seven Q heads and one KV head per rank, matching BAGEL's
    # non-power-of-two GQA geometry while exercising FA4 + Liger + checkpointing + FSDP2.
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
            "rms_norm_implementation": "liger_kernel",
            "rotary_pos_emb_implementation": "liger_kernel",
            "swiglu_mlp_implementation": "eager",
        },
    )
    reference = accelerated.BagelQwen2MoTAccelerated(config).to(device=device, dtype=dtype).train()
    candidate = copy.deepcopy(reference)
    for model in (reference, candidate):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.float()
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.model.layers:
            fully_shard(layer, mesh=state.fsdp_mesh, mp_policy=policy)
        fully_shard(model.model, mesh=state.fsdp_mesh, mp_policy=policy)

    selected_backends = []
    original_resolve = flex_backend.resolve_flex_attention

    def capture_backend(*args, **kwargs):
        interface = original_resolve(*args, **kwargs)
        selected_backends.append(interface)
        return interface

    flex_backend.resolve_flex_attention = capture_backend
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
    inputs = {
        "packed_position_ids": positions[start:stop],
        "packed_attention_metadata": metadata,
        "packed_und_token_indexes": torch.where(token_types[start:stop] == 0)[0],
        "packed_gen_token_indexes": torch.where(token_types[start:stop] == 1)[0],
    }
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-3) for model in (reference, candidate)]
    try:
        for _ in range(2):
            outputs, input_gradients = [], []
            for model, model_state in ((reference, state), (candidate, async_state)):
                local_input = hidden[start:stop].detach().clone().requires_grad_()
                with use_parallel_state(model_state):
                    output = model.model(packed_sequence=local_input, **inputs).packed_query_sequence
                    output.backward(output_gradient[start:stop])
                outputs.append(output.detach())
                input_gradients.append(local_input.grad)
            torch.testing.assert_close(outputs[1], outputs[0], atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(input_gradients[1], input_gradients[0], atol=3e-2, rtol=3e-2)
            for (name, expected), (actual_name, actual) in zip(
                reference.named_parameters(), candidate.named_parameters(), strict=True
            ):
                assert name == actual_name
                assert expected.grad is not None and actual.grad is not None, name
                expected_grad, actual_grad = expected.grad.to_local(), actual.grad.to_local()
                assert torch.isfinite(actual_grad).all(), name
                error = (actual_grad.float() - expected_grad.float()).norm()
                scale = expected_grad.float().norm()
                assert error <= 0.01 * scale + 1e-7, (
                    f"{name}: relative gradient L2 error {error / scale.clamp_min(1e-12)}"
                )

            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        assert selected_backends and all(
            selected is flex_backend._flex_attention_fa4 for selected in selected_backends
        )
    finally:
        flex_backend.resolve_flex_attention = original_resolve


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4))
@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or get_torch_device().device_count() < 4,
    reason="BAGEL FA4 + async Ulysses integration requires four SM90+ GPUs",
)
def test_bagel_fa4_async_matches_sync_forward_backward_and_updates() -> None:
    """Verify BAGEL async Ulysses matches synchronous Ulysses under Flex-FA4, Liger, checkpointing, and FSDP2."""
    device = torch.device(get_device_type())
    if flex_backend.get_gpu_compute_capability(device) < 90 or not flex_backend._fa4._flash_attn_cute_available():
        pytest.skip("Flex-FA4 integration requires SM90+ and flash-attn-4")
    torchrun(_bagel_fa4_async_worker, world_size=4)
