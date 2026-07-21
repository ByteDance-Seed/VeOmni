"""Distributed parity coverage for BAGEL flow-connector token SP."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls
from tests.tools.launch_utils import torchrun
from veomni.distributed.parallel_state import init_parallel_state, use_parallel_state
from veomni.models.seed_omni.graphs.dispatch import call_graph_endpoint
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_FLOW_HIDDEN, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem
from veomni.utils.device import get_device_type, get_torch_device


def _replicated_embed_conversation(device: torch.device) -> list[list[ConversationItem]]:
    generator = torch.Generator(device=device).manual_seed(9182)
    conversation: list[list[ConversationItem]] = []
    for height, width in ((2, 2), (1, 2), (1, 3)):
        conversation.append(
            [
                ConversationItem(
                    type="image",
                    value=torch.randn(
                        1,
                        height,
                        width,
                        generator=generator,
                        device=device,
                        dtype=torch.bfloat16,
                    ),
                    role="assistant",
                    source=BAGEL_VAE_CONTEXT,
                    meta={_IMG_TAG_KEY: "edit"},
                )
            ]
        )
    return conversation


def _replicated_decode_conversation(
    device: torch.device,
    *,
    hidden_size: int,
    patch_latent_dim: int,
) -> tuple[list[list[ConversationItem]], list[torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(1743)
    conversation: list[list[ConversationItem]] = []
    inputs: list[torch.Tensor] = []
    for token_count in (4, 2, 3):
        hidden = torch.randn(
            token_count,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        target = torch.randn(
            token_count,
            patch_latent_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        inputs.append(hidden)
        conversation.append(
            [
                ConversationItem(
                    type="image",
                    value=hidden,
                    role="assistant",
                    source=BAGEL_FLOW_HIDDEN,
                    meta={"flow_velocity_target": target},
                )
            ]
        )
    return conversation, inputs


def _forward_embed(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(device_type=get_device_type(), dtype=torch.bfloat16),
    ):
        inputs = model.embed_latent_pre(conversation_list=conversation)
        local_shape = tuple(inputs["latents"].shape)
        outputs = call_graph_endpoint(model, model, method="embed_latent", kwargs=inputs)
        result = model.embed_latent_post(**outputs)
    embeds = torch.cat([sample[0].value for sample in result["conversation_list"]], dim=0)
    return embeds, local_shape


def _forward_decode(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(device_type=get_device_type(), dtype=torch.bfloat16),
    ):
        inputs = model.decode_velocity_pre(conversation_list=conversation)
        local_shape = tuple(inputs["hidden_states"].shape)
        outputs = call_graph_endpoint(model, model, method="decode_velocity", kwargs=inputs)
        result = model.decode_velocity_post(**outputs)
    velocity = torch.cat([sample[0].value for sample in result["conversation_list"]], dim=0)
    return velocity, result["_loss"], local_shape


def _flow_connector_sp_worker() -> None:
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

    BagelFlowConnector = model_cls("bagel_flow_connector")
    BagelFlowConnectorConfig = config_cls("bagel_flow_connector")
    config = BagelFlowConnectorConfig(
        hidden_size=8,
        z_channels=1,
        latent_patch_size=1,
        patch_latent_dim=1,
        max_latent_size=4,
        timestep_frequency_embedding_size=4,
        timestep_shift=1.0,
    )
    torch.manual_seed(6029)
    reference = BagelFlowConnector(config).to(device=device, dtype=torch.bfloat16).train()
    with torch.no_grad():
        reference.llm2vae.weight.normal_(mean=0.0, std=0.1)
        reference.llm2vae.bias.normal_(mean=0.0, std=0.1)
    torch.manual_seed(6029)
    sequence_parallel = BagelFlowConnector(config).to(device=device, dtype=torch.bfloat16).train()
    sequence_parallel.load_state_dict(reference.state_dict())

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(sequence_parallel, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)

    reference_embed_conversation = _replicated_embed_conversation(device)
    sp_embed_conversation = _replicated_embed_conversation(device)
    reference_embeds, reference_embed_shape = _forward_embed(
        reference,
        reference_embed_conversation,
        non_sp_state,
    )
    sp_embeds, sp_embed_shape = _forward_embed(sequence_parallel, sp_embed_conversation, sp_state)

    assert reference_embed_shape == (9, 1)
    assert sp_embed_shape == (3, 1)
    assert sequence_parallel._metric_full_seqlens["embed_latent"] == [4, 2, 3]
    assert torch.isfinite(reference_embeds).all()
    assert torch.isfinite(sp_embeds).all()
    torch.testing.assert_close(sp_embeds, reference_embeds, rtol=2e-2, atol=2e-2)

    reference_decode_conversation, reference_inputs = _replicated_decode_conversation(
        device,
        hidden_size=int(config.hidden_size),
        patch_latent_dim=int(config.patch_latent_dim),
    )
    sp_decode_conversation, sp_inputs = _replicated_decode_conversation(
        device,
        hidden_size=int(config.hidden_size),
        patch_latent_dim=int(config.patch_latent_dim),
    )
    reference_velocity, reference_decode_loss, reference_decode_shape = _forward_decode(
        reference,
        reference_decode_conversation,
        non_sp_state,
    )
    sp_velocity, sp_decode_loss, sp_decode_shape = _forward_decode(
        sequence_parallel,
        sp_decode_conversation,
        sp_state,
    )

    assert reference_decode_shape == (9, int(config.hidden_size))
    assert sp_decode_shape == (3, int(config.hidden_size))
    assert sequence_parallel._metric_full_seqlens["decode_velocity"] == [4, 2, 3]
    assert torch.isfinite(reference_velocity).all()
    assert torch.isfinite(sp_velocity).all()
    assert torch.isfinite(reference_decode_loss)
    assert torch.isfinite(sp_decode_loss)
    torch.testing.assert_close(sp_velocity, reference_velocity, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(sp_decode_loss, reference_decode_loss, rtol=2e-2, atol=2e-2)

    reference_loss = reference_embeds.float().square().mean() + reference_decode_loss.float()
    sp_loss = sp_embeds.float().square().mean() + sp_decode_loss.float()
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
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"reference parameter {name} did not participate in backward"
        assert torch.isfinite(parameter.grad).all()
        grad = parameter.grad.detach().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        reference_grads[name] = grad / world_size

    for name, parameter in sequence_parallel.named_parameters():
        if not parameter.requires_grad:
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

    dist.barrier()


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
def test_flow_connector_sp4_matches_non_sp_for_embed_and_decode() -> None:
    torchrun(_flow_connector_sp_worker, world_size=4)
