"""Distributed parity coverage for BAGEL VAE batch-dimension SP."""

from __future__ import annotations

from types import MethodType

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls
from tests.tools.launch_utils import torchrun
from veomni.distributed.parallel_state import init_parallel_state, use_parallel_state
from veomni.models.seed_omni.accelerator.dispatch import call_graph_endpoint
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.modules.bagel.vae.accelerated import BAGEL_VAE_PIXEL_SHAPE
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.utils.device import get_device_type, get_torch_device


def _replicated_conversation(
    device: torch.device,
) -> tuple[list[list[ConversationItem]], list[torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(6117)
    conversation: list[list[ConversationItem]] = []
    inputs: list[torch.Tensor] = []
    for height, width in ((8, 8), (8, 4), (4, 8)):
        value = torch.zeros(3, 8, 8, device=device, dtype=torch.bfloat16)
        real_pixels = torch.randn(
            3,
            height,
            width,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        value[:, :height, :width] = real_pixels
        value.requires_grad_()
        inputs.append(value)
        conversation.append(
            [
                ConversationItem(
                    type="image",
                    value=value,
                    role="assistant",
                    source=BAGEL_VAE_CONTEXT,
                    meta={BAGEL_VAE_PIXEL_SHAPE: torch.tensor([height, width], device=device)},
                )
            ]
        )
    return conversation, inputs


def _deterministic_latents(self, posterior: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mean, _ = posterior
    return self.config.scale_factor * (mean - self.config.shift_factor)


def _forward_encode(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
) -> tuple[list[torch.Tensor], tuple[int, ...]]:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(device_type=get_device_type(), dtype=torch.bfloat16),
    ):
        inputs = model.encode_pre(conversation_list=conversation)
        local_shape = tuple(inputs["pixel_values"].shape)
        outputs = call_graph_endpoint(model, model, method="encode", kwargs=inputs)
        result = model.encode_post(**outputs)
    return [sample[0].value for sample in result["conversation_list"]], local_shape


def _forward_offline_encode(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
) -> list[torch.Tensor]:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(device_type=get_device_type(), dtype=torch.bfloat16),
    ):
        inputs = model.encode_pre(conversation_list=conversation)
        outputs = call_graph_endpoint(model, model, method="offline_encode", kwargs=inputs)
        result = model.offline_encode_post(**outputs)
    return [sample[0].value for sample in result["conversation_list"]]


def _vae_sp_worker() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4
    device = torch.device(f"{get_device_type()}:{rank}")

    non_sp_state = init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
    )
    sp_state = init_parallel_state(
        ulysses_size=world_size,
        dp_mode="fsdp2",
    )

    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    config_kwargs = {
        "resolution": 8,
        "in_channels": 3,
        "downsample": 2,
        "ch": 32,
        "ch_mult": [1, 1],
        "num_res_blocks": 1,
        "z_channels": 2,
        "max_image_size": 8,
        "min_image_size": 4,
        "image_stride": 4,
        "max_pixels": 64,
        "freeze": False,
    }
    torch.manual_seed(2271)
    reference = BagelVAE(BagelVAEConfig(**config_kwargs)).to(device=device, dtype=torch.bfloat16).train()
    torch.manual_seed(2271)
    sequence_parallel = (
        BagelVAE(BagelVAEConfig(**config_kwargs))
        .to(
            device=device,
            dtype=torch.bfloat16,
        )
        .train()
    )
    sequence_parallel.load_state_dict(reference.state_dict())
    reference._sample_scaled_latents = MethodType(_deterministic_latents, reference)
    sequence_parallel._sample_scaled_latents = MethodType(_deterministic_latents, sequence_parallel)

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(sequence_parallel, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)

    reference_conversation, reference_inputs = _replicated_conversation(device)
    sp_conversation, sp_inputs = _replicated_conversation(device)
    reference_latents, reference_local_shape = _forward_encode(reference, reference_conversation, non_sp_state)
    sp_latents, sp_local_shape = _forward_encode(sequence_parallel, sp_conversation, sp_state)

    assert reference_local_shape == (3, 3, 8, 8)
    assert sp_local_shape == (1, 3, 8, 8)
    assert sequence_parallel._metric_full_seqlens["encode"] == [16, 8, 8]
    assert [tuple(latent.shape) for latent in sp_latents] == [(2, 4, 4), (2, 4, 2), (2, 2, 4)]
    for reference_latent, sp_latent in zip(reference_latents, sp_latents, strict=True):
        assert torch.isfinite(reference_latent).all()
        assert torch.isfinite(sp_latent).all()
        torch.testing.assert_close(sp_latent, reference_latent, rtol=2e-2, atol=2e-2)

    reference_loss = sum(latent.float().square().mean() for latent in reference_latents)
    sp_loss = sum(latent.float().square().mean() for latent in sp_latents)
    reference_loss.backward()
    sp_loss.backward()

    for reference_input, sp_input in zip(reference_inputs, sp_inputs, strict=True):
        assert reference_input.grad is not None
        assert sp_input.grad is not None
        assert torch.isfinite(reference_input.grad).all()
        assert torch.isfinite(sp_input.grad).all()
        sp_grad = sp_input.grad.detach().clone()
        dist.all_reduce(sp_grad, op=dist.ReduceOp.SUM, group=sp_state.sp_group)
        sp_grad /= world_size
        torch.testing.assert_close(sp_grad, reference_input.grad, rtol=3e-2, atol=3e-2)

    reference_grads: dict[str, torch.Tensor] = {}
    for name, parameter in reference.named_parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all()
        grad = parameter.grad.detach().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        reference_grads[name] = grad / world_size

    for name, parameter in sequence_parallel.named_parameters():
        if name not in reference_grads:
            assert parameter.grad is None
            continue
        assert parameter.grad is not None, f"SP parameter {name} did not participate in backward"
        local_grad = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        assert torch.isfinite(local_grad).all()
        sp_grad = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
        torch.testing.assert_close(
            sp_grad,
            reference_grads[name],
            rtol=5e-2,
            atol=5e-2,
            msg=f"SP gradient mismatch for {name}",
        )

    reference_offline_conversation, _ = _replicated_conversation(device)
    sp_offline_conversation, _ = _replicated_conversation(device)
    reference_cache = _forward_offline_encode(reference, reference_offline_conversation, non_sp_state)
    sp_cache = _forward_offline_encode(sequence_parallel, sp_offline_conversation, sp_state)
    assert [tuple(cache.shape) for cache in sp_cache] == [(2, 2, 4, 4), (2, 2, 4, 2), (2, 2, 2, 4)]
    for reference_item_cache, sp_item_cache in zip(reference_cache, sp_cache, strict=True):
        assert torch.isfinite(reference_item_cache).all()
        assert torch.isfinite(sp_item_cache).all()
        torch.testing.assert_close(sp_item_cache, reference_item_cache, rtol=2e-2, atol=2e-2)

    dist.barrier()


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
def test_vae_sp4_matches_non_sp_for_encode_and_offline_encode() -> None:
    torchrun(_vae_sp_worker, world_size=4)
