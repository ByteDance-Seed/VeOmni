"""Distributed parity coverage for BAGEL flow-connector sequence parallelism."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls
from tests.tools.launch_utils import torchrun
from veomni.distributed.parallel_state import init_parallel_state, use_parallel_state
from veomni.models.seed_omni.graphs.dispatch import run_sp_looped_endpoint
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_FLOW_HIDDEN, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem
from veomni.utils.device import get_device_type, get_torch_device


_TOKEN_COUNTS = (4, 2, 1, 3)
_LATENT_SHAPES = ((4, 4), (2, 4), (2, 2), (2, 6))


def _embed_conversation(
    rank: int,
    device: torch.device,
) -> tuple[list[list[ConversationItem]], torch.Tensor]:
    height, width = _LATENT_SHAPES[rank]
    latent = torch.arange(height * width, device=device, dtype=torch.bfloat16).reshape(1, height, width)
    latent = (latent + rank).requires_grad_(rank == 2)
    item = ConversationItem(
        type="image",
        value=latent,
        role="dummy" if rank == 2 else "assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={} if rank == 2 else {_IMG_TAG_KEY: "gen" if rank == 1 else "edit"},
    )
    return [[item]], latent


def _decode_conversation(
    rank: int,
    device: torch.device,
    hidden_size: int,
    patch_latent_dim: int,
) -> tuple[list[list[ConversationItem]], torch.Tensor, bool]:
    if rank == 2:
        hidden = torch.arange(3 * hidden_size, device=device, dtype=torch.bfloat16).reshape(3, hidden_size)
        hidden = hidden.requires_grad_()
        item = ConversationItem(type="text", value=hidden, role="user")
        return [[item]], hidden, False

    token_count = _TOKEN_COUNTS[rank]
    hidden = torch.arange(token_count * hidden_size, device=device, dtype=torch.bfloat16)
    hidden = (hidden.reshape(token_count, hidden_size) / 16 + rank).requires_grad_()
    target = torch.arange(token_count * patch_latent_dim, device=device, dtype=torch.bfloat16)
    target = target.reshape(token_count, patch_latent_dim) / 8 - rank
    item = ConversationItem(
        type="image",
        value=hidden,
        role="assistant",
        source=BAGEL_FLOW_HIDDEN,
        meta={"flow_velocity_target": target},
    )
    return [[item]], hidden, True


def _flow_connector_sp_worker() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"{get_device_type()}:{rank}")
    outer_state = init_parallel_state(
        dp_size=world_size,
        dp_mode="ddp",
    )
    module_state = init_parallel_state(
        ulysses_size=world_size,
        dp_mode="ddp",
    )

    BagelFlowConnector = model_cls("bagel_flow_connector")
    BagelFlowConnectorConfig = config_cls("bagel_flow_connector")
    config = BagelFlowConnectorConfig(
        hidden_size=8,
        z_channels=1,
        latent_patch_size=2,
        max_latent_size=8,
        timestep_frequency_embedding_size=4,
    )
    torch.manual_seed(9201)
    reference = BagelFlowConnector(config).to(device=device, dtype=torch.bfloat16).train()
    with torch.no_grad():
        reference.llm2vae.weight.normal_(std=0.02)
        reference.llm2vae.bias.normal_(std=0.02)
    torch.manual_seed(9202)
    sequence_parallel = BagelFlowConnector(config).to(device=device, dtype=torch.bfloat16).train()
    sequence_parallel.load_state_dict(reference.state_dict())

    reference_embed_conversation, reference_latent = _embed_conversation(rank, device)
    sp_embed_conversation, sp_latent = _embed_conversation(rank, device)

    torch.manual_seed(9300 + rank)
    with use_parallel_state(outer_state):
        reference_embed_inputs = reference.embed_latent_pre(conversation_list=reference_embed_conversation)
        reference_embed_output = reference.embed_latent(**reference_embed_inputs)
        reference_embed_result = reference.embed_latent_post(**reference_embed_output)

    torch.manual_seed(9300 + rank)
    with use_parallel_state(module_state):
        sp_embed_inputs = sequence_parallel.embed_latent_pre(conversation_list=sp_embed_conversation)
        assert sequence_parallel.supports_sp("embed_latent")
        assert sp_embed_inputs["latents"].shape[0] == _TOKEN_COUNTS[rank]
        sp_embed_output = run_sp_looped_endpoint(
            sequence_parallel,
            sequence_parallel,
            method="embed_latent",
            kwargs=sp_embed_inputs,
        )
        sp_embed_result = sequence_parallel.embed_latent_post(**sp_embed_output)

    reference_embed = reference_embed_result["conversation_list"][0][0].value
    sp_embed = sp_embed_result["conversation_list"][0][0].value
    torch.testing.assert_close(sp_embed, reference_embed, rtol=2e-2, atol=2e-2)

    hidden_size = int(config.hidden_size)
    patch_latent_dim = int(config.patch_latent_dim)
    reference_decode_conversation, reference_hidden, has_target = _decode_conversation(
        rank,
        device,
        hidden_size,
        patch_latent_dim,
    )
    sp_decode_conversation, sp_hidden, _ = _decode_conversation(rank, device, hidden_size, patch_latent_dim)

    with use_parallel_state(outer_state):
        reference_decode_inputs = reference.decode_velocity_pre(conversation_list=reference_decode_conversation)
        reference_decode_output = reference.decode_velocity(**reference_decode_inputs)
        reference_decode_result = reference.decode_velocity_post(**reference_decode_output)

    with use_parallel_state(module_state):
        sp_decode_inputs = sequence_parallel.decode_velocity_pre(conversation_list=sp_decode_conversation)
        assert sequence_parallel.supports_sp("decode_velocity")
        assert sp_decode_inputs["hidden_states"].shape[0] == _TOKEN_COUNTS[rank]
        sp_decode_output = run_sp_looped_endpoint(
            sequence_parallel,
            sequence_parallel,
            method="decode_velocity",
            kwargs=sp_decode_inputs,
        )
        sp_decode_result = sequence_parallel.decode_velocity_post(**sp_decode_output)

    reference_decode_loss = reference_decode_result["_loss"]
    sp_decode_loss = sp_decode_result["_loss"]
    torch.testing.assert_close(sp_decode_loss, reference_decode_loss, rtol=2e-2, atol=2e-2)
    if has_target:
        reference_velocity = reference_decode_result["conversation_list"][0][0].value
        sp_velocity = sp_decode_result["conversation_list"][0][0].value
        torch.testing.assert_close(sp_velocity, reference_velocity, rtol=2e-2, atol=2e-2)

    embed_scale = 0.0 if rank == 2 else 1.0
    reference_loss = reference_embed.float().square().mean() * embed_scale + reference_decode_loss.float()
    sp_loss = sp_embed.float().square().mean() * embed_scale + sp_decode_loss.float()
    reference_loss.backward()
    sp_loss.backward()

    assert reference_hidden.grad is not None
    assert sp_hidden.grad is not None
    torch.testing.assert_close(sp_hidden.grad, reference_hidden.grad, rtol=3e-2, atol=3e-2)
    if rank == 2:
        assert reference_latent.grad is not None
        assert sp_latent.grad is not None
        torch.testing.assert_close(sp_latent.grad, reference_latent.grad, rtol=3e-2, atol=3e-2)

    reference_grads: dict[str, torch.Tensor] = {}
    for name, parameter in reference.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"reference parameter {name} did not participate in backward"
        grad = parameter.grad.detach().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        reference_grads[name] = grad / world_size

    for name, parameter in sequence_parallel.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"SP parameter {name} did not participate in backward"
        sp_grad = parameter.grad.detach().clone()
        dist.all_reduce(sp_grad, op=dist.ReduceOp.SUM)
        sp_grad /= world_size
        torch.testing.assert_close(
            sp_grad,
            reference_grads[name],
            rtol=5e-2,
            atol=5e-2,
            msg=f"SP gradient mismatch for {name}",
        )

    dist.barrier()


def _flow_connector_fsdp2_worker() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4
    device = torch.device(f"{get_device_type()}:{rank}")
    outer_state = init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
    )
    module_state = init_parallel_state(ulysses_size=world_size, dp_mode="fsdp2")

    BagelFlowConnector = model_cls("bagel_flow_connector")
    BagelFlowConnectorConfig = config_cls("bagel_flow_connector")
    config = BagelFlowConnectorConfig(
        hidden_size=8,
        z_channels=1,
        latent_patch_size=2,
        max_latent_size=8,
        timestep_frequency_embedding_size=4,
    )
    torch.manual_seed(9401)
    reference = BagelFlowConnector(config).to(device=device, dtype=torch.bfloat16).train()
    with torch.no_grad():
        reference.llm2vae.weight.normal_(std=0.02)
        reference.llm2vae.bias.normal_(std=0.02)
    sequence_parallel = BagelFlowConnector(config).to(device=device, dtype=torch.bfloat16).train()
    sequence_parallel.load_state_dict(reference.state_dict())
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(sequence_parallel, mesh=module_state.fsdp_mesh, mp_policy=mp_policy)

    reference_conversation, reference_hidden, has_target = _decode_conversation(
        rank,
        device,
        int(config.hidden_size),
        int(config.patch_latent_dim),
    )
    sp_conversation, sp_hidden, _ = _decode_conversation(
        rank,
        device,
        int(config.hidden_size),
        int(config.patch_latent_dim),
    )

    with use_parallel_state(outer_state):
        reference_inputs = reference.decode_velocity_pre(conversation_list=reference_conversation)
        reference_output = reference.decode_velocity(**reference_inputs)
        reference_result = reference.decode_velocity_post(**reference_output)

    with use_parallel_state(module_state):
        sp_inputs = sequence_parallel.decode_velocity_pre(conversation_list=sp_conversation)
        assert sequence_parallel.supports_sp("decode_velocity")
        assert sp_inputs["hidden_states"].shape[0] == _TOKEN_COUNTS[rank]
        sp_output = run_sp_looped_endpoint(
            sequence_parallel,
            sequence_parallel,
            method="decode_velocity",
            kwargs=sp_inputs,
        )
        sp_result = sequence_parallel.decode_velocity_post(**sp_output)

    torch.testing.assert_close(sp_result["_loss"], reference_result["_loss"], rtol=2e-2, atol=2e-2)
    if has_target:
        torch.testing.assert_close(
            sp_result["conversation_list"][0][0].value,
            reference_result["conversation_list"][0][0].value,
            rtol=2e-2,
            atol=2e-2,
        )

    reference_result["_loss"].float().backward()
    sp_result["_loss"].float().backward()
    assert reference_hidden.grad is not None
    assert sp_hidden.grad is not None
    torch.testing.assert_close(sp_hidden.grad, reference_hidden.grad, rtol=3e-2, atol=3e-2)

    for name, parameter in sequence_parallel.named_parameters():
        if not parameter.requires_grad or not name.startswith("llm2vae."):
            continue
        assert parameter.grad is not None, f"SP parameter {name} did not participate in backward"
        local_grad = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        assert torch.isfinite(local_grad).all(), f"SP parameter {name} has non-finite gradients"

    dist.barrier()


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
def test_flow_connector_sp2_matches_non_sp() -> None:
    torchrun(_flow_connector_sp_worker, world_size=2)


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
def test_flow_connector_sp4_matches_non_sp_with_fsdp2_and_mixed_dummy_inputs() -> None:
    torchrun(_flow_connector_fsdp2_worker, world_size=4)
