"""Distributed parity coverage for BAGEL SigLIP NaViT batch-dimension SP."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls
from tests.tools.launch_utils import torchrun
from veomni.distributed.parallel_state import init_parallel_state, use_parallel_state
from veomni.models.seed_omni.modules.bagel.siglip_navit.modulemixin import (
    _OMNI_POSITION_IDS,
    _OMNI_TOKEN_LEN,
)
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_SIGLIP_CONTEXT
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.utils.device import get_device_type, get_torch_device


def _replicated_conversation(
    device: torch.device,
    patch_dim: int,
) -> tuple[list[list[ConversationItem]], list[torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(8124)
    conversation: list[list[ConversationItem]] = []
    inputs: list[torch.Tensor] = []
    for token_len in (2, 3, 5):
        value = torch.randn(
            token_len,
            patch_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        inputs.append(value)
        conversation.append(
            [
                ConversationItem(
                    type="image",
                    value=value,
                    role="user",
                    source=BAGEL_SIGLIP_CONTEXT,
                    meta={
                        _OMNI_POSITION_IDS: torch.arange(token_len, device=device),
                        _OMNI_TOKEN_LEN: token_len,
                    },
                )
            ]
        )
    return conversation, inputs


def _carrier_embeds(conversation: list[list[ConversationItem]]) -> torch.Tensor:
    return torch.cat([sample[0].value for sample in conversation], dim=0)


def _forward_carrier(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
) -> tuple[torch.Tensor, list[int]]:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(device_type=get_device_type(), dtype=torch.bfloat16),
    ):
        inputs = model.forward_pre(conversation_list=conversation)
        local_token_lens = inputs["token_lens"].detach().cpu().tolist()
        outputs = model(**inputs)
        result = model.forward_post(**outputs)
    return _carrier_embeds(result["conversation_list"]), local_token_lens


def _siglip_navit_sp_worker() -> None:
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

    BagelSiglip = model_cls("bagel_siglip_navit")
    BagelSiglipConfig = config_cls("bagel_siglip_navit")
    config = BagelSiglipConfig(
        hidden_size=16,
        output_size=16,
        image_size=8,
        min_image_size=2,
        max_pixels=64,
        intermediate_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_channels=1,
        patch_size=2,
        vit_max_num_patch_per_side=4,
    )
    torch.manual_seed(4901)
    reference = BagelSiglip(config).to(device=device, dtype=torch.bfloat16).train()
    torch.manual_seed(4901)
    sequence_parallel = BagelSiglip(config).to(device=device, dtype=torch.bfloat16).train()
    sequence_parallel.load_state_dict(reference.state_dict())

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in sequence_parallel.vision_model.encoder.layers:
        fully_shard(layer, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)
    fully_shard(sequence_parallel, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)

    patch_dim = int(config.num_channels * config.patch_size * config.patch_size)
    reference_conversation, reference_inputs = _replicated_conversation(device, patch_dim)
    sp_conversation, sp_inputs = _replicated_conversation(device, patch_dim)

    reference_embeds, reference_token_lens = _forward_carrier(reference, reference_conversation, non_sp_state)
    sp_embeds, local_token_lens = _forward_carrier(sequence_parallel, sp_conversation, sp_state)

    assert reference_token_lens == [2, 3, 5]
    assert local_token_lens == [[2], [3], [5], [1]][rank]
    assert sequence_parallel._metric_full_seqlens["forward"] == [2, 3, 5]
    assert torch.isfinite(reference_embeds).all()
    assert torch.isfinite(sp_embeds).all()
    torch.testing.assert_close(sp_embeds, reference_embeds, rtol=2e-2, atol=2e-2)

    reference_embeds.float().square().mean().backward()
    sp_embeds.float().square().mean().backward()

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
def test_siglip_navit_sp4_matches_non_sp_with_variable_image_lengths() -> None:
    torchrun(_siglip_navit_sp_worker, world_size=4)
