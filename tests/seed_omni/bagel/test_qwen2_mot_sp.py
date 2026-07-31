"""Distributed parity coverage for BAGEL Qwen2-MoT sequence parallelism."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls, tiny_bagel_qwen2_cfg
from tests.tools.launch_utils import torchrun
from veomni.distributed.parallel_state import init_parallel_state, use_parallel_state
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem
from veomni.utils.device import get_device_type, get_torch_device


def _sample_items(
    sample_index: int,
    device: torch.device,
    hidden_size: int,
) -> tuple[list[ConversationItem], list[torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(7300 + sample_index)

    def make_tensor(length: int) -> torch.Tensor:
        return torch.randn(
            length,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    if sample_index == 0:
        values = [make_tensor(2), make_tensor(3)]
        items = [
            ConversationItem(type="text", value=values[0], role="user"),
            ConversationItem(
                type="image",
                value=values[1],
                role="user",
                source=BAGEL_SIGLIP_CONTEXT,
            ),
        ]
    elif sample_index == 1:
        values = [make_tensor(2), make_tensor(3), make_tensor(1)]
        items = [
            ConversationItem(type="text", value=values[0], role="user"),
            ConversationItem(
                type="image",
                value=values[1],
                role="assistant",
                source=BAGEL_VAE_CONTEXT,
                meta={_IMG_TAG_KEY: "gen"},
            ),
            ConversationItem(type="text", value=values[2], role="assistant"),
        ]
    elif sample_index == 2:
        values = [make_tensor(2), make_tensor(5)]
        items = [
            ConversationItem(
                type="image",
                value=values[0],
                role="user",
                source=BAGEL_SIGLIP_CONTEXT,
            ),
            ConversationItem(type="text", value=values[1], role="assistant"),
        ]
    else:
        values = [make_tensor(1), make_tensor(2), make_tensor(2), make_tensor(3)]
        items = [
            ConversationItem(type="text", value=values[0], role="user"),
            ConversationItem(
                type="image",
                value=values[1],
                role="user",
                source=BAGEL_SIGLIP_CONTEXT,
            ),
            ConversationItem(
                type="image",
                value=values[2],
                role="assistant",
                source=BAGEL_VAE_CONTEXT,
                meta={_IMG_TAG_KEY: "gen"},
            ),
            ConversationItem(
                type="image",
                value=values[3],
                role="assistant",
                source=BAGEL_VAE_CONTEXT,
                meta={_IMG_TAG_KEY: "gen"},
            ),
        ]

    dummy = make_tensor(1)
    values.append(dummy)
    items.append(
        ConversationItem(
            type="image",
            value=dummy,
            role="dummy",
            source=BAGEL_SIGLIP_CONTEXT,
        )
    )
    return items, values


def _replicated_batch(
    device: torch.device,
    hidden_size: int,
) -> tuple[list[list[ConversationItem]], list[torch.Tensor]]:
    conversation: list[list[ConversationItem]] = []
    inputs: list[torch.Tensor] = []
    for sample_index in range(4):
        items, sample_inputs = _sample_items(sample_index, device, hidden_size)
        conversation.append(items)
        inputs.extend(sample_inputs)
    return conversation, inputs


def _carrier_hidden(conversation: list[list[ConversationItem]]) -> torch.Tensor:
    real_hidden = [
        item.value
        for sample in conversation
        for item in sample
        if item.role != "dummy" and torch.is_tensor(item.value)
    ]
    assert real_hidden
    return torch.cat(real_hidden, dim=0)


def _forward_carrier(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
    input_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
) -> torch.Tensor:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(
            device_type=get_device_type(),
            dtype=torch.bfloat16,
        ),
    ):
        inputs = model.forward_pre(conversation_list=conversation)
        if input_shapes is not None:
            input_shapes.append(
                (
                    tuple(inputs["packed_sequence"].shape),
                    tuple(inputs["packed_attention_metadata"].shape),
                )
            )
        outputs = model(**inputs)
        result = model.forward_post(**outputs)
    return _carrier_hidden(result["conversation_list"])


def _enable_scoped_gradient_checkpointing(model: torch.nn.Module, parallel_state) -> None:
    def context_fn():
        return nullcontext(), use_parallel_state(parallel_state)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": context_fn,
        }
    )


def _qwen2_mot_sp_worker() -> None:
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

    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    config_kwargs = {
        **tiny_bagel_qwen2_cfg(),
        # FlexAttention's Triton kernel requires head_dim >= 16.
        "hidden_size": 448,
        "intermediate_size": 896,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "attn_implementation": "veomni_flex_attention_with_sp",
    }
    torch.manual_seed(9102)
    reference = (
        BagelQwen2MoT(BagelQwen2MoTConfig(**config_kwargs))
        .to(
            device=device,
            dtype=torch.bfloat16,
        )
        .train()
    )
    torch.manual_seed(9102)
    sequence_parallel = (
        BagelQwen2MoT(BagelQwen2MoTConfig(**config_kwargs))
        .to(
            device=device,
            dtype=torch.bfloat16,
        )
        .train()
    )
    sequence_parallel.load_state_dict(reference.state_dict())

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in sequence_parallel.model.layers:
        fully_shard(layer, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)
    fully_shard(sequence_parallel, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)
    _enable_scoped_gradient_checkpointing(sequence_parallel, sp_state)

    hidden_size = int(reference.config.hidden_size)
    reference_conversation, reference_inputs = _replicated_batch(device, hidden_size)
    sp_conversation, sp_inputs = _replicated_batch(device, hidden_size)
    expected_sample_lengths = [5, 6, 7, 8]
    input_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    reference_hidden = _forward_carrier(reference, reference_conversation, non_sp_state)
    sp_hidden = _forward_carrier(sequence_parallel, sp_conversation, sp_state, input_shapes)
    assert input_shapes == [((7, hidden_size), (3, 28))]
    assert sequence_parallel._metric_full_seqlens["forward"] == expected_sample_lengths
    assert torch.isfinite(reference_hidden).all()
    assert torch.isfinite(sp_hidden).all()
    torch.testing.assert_close(sp_hidden, reference_hidden, rtol=2e-2, atol=2e-2)

    reference_loss = reference_hidden.float().square().mean()
    sp_loss = sp_hidden.float().square().mean()
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
        assert parameter.grad is not None, f"reference parameter {name} did not participate in backward"
        assert torch.isfinite(parameter.grad).all()
        grad = parameter.grad.detach().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        reference_grads[name] = grad / world_size

    for name, parameter in sequence_parallel.named_parameters():
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
def test_qwen2_mot_sp4_matches_non_sp_with_fsdp2_and_gradient_checkpointing() -> None:
    torchrun(_qwen2_mot_sp_worker, world_size=4)
