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


def _rank_items(
    rank: int, device: torch.device, hidden_size: int
) -> tuple[list[ConversationItem], list[torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(7300 + rank)

    def make_tensor(length: int) -> torch.Tensor:
        return torch.randn(
            length,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    if rank == 0:
        values = [make_tensor(3)]
        items = [
            ConversationItem(
                type="image",
                value=values[0],
                role="user",
                source=BAGEL_SIGLIP_CONTEXT,
            )
        ]
    elif rank == 1:
        values = [make_tensor(2), make_tensor(3)]
        items = [
            ConversationItem(type="text", value=values[0], role="user"),
            ConversationItem(
                type="image",
                value=values[1],
                role="assistant",
                source=BAGEL_VAE_CONTEXT,
                meta={_IMG_TAG_KEY: "gen"},
            ),
        ]
    elif rank == 2:
        values = [make_tensor(7)]
        items = [ConversationItem(type="text", value=values[0], role="assistant")]
    else:
        values = [make_tensor(1)]
        items = [
            ConversationItem(
                type="image",
                value=values[0],
                role="dummy",
                source=BAGEL_SIGLIP_CONTEXT,
            )
        ]
    return items, values


def _carrier_hidden(conversation: list[list[ConversationItem]]) -> torch.Tensor:
    sample = conversation[0]
    real_hidden = [item.value for item in sample if item.role != "dummy" and torch.is_tensor(item.value)]
    if real_hidden:
        return torch.cat(real_hidden, dim=0)

    dummy_output = sample[-1]
    assert dummy_output.role == "dummy"
    assert dummy_output.source == "bagel_qwen2_mot"
    hidden = dummy_output.value
    return hidden.unsqueeze(0) if hidden.dim() == 1 else hidden


def _forward_carrier(
    model: torch.nn.Module,
    items: list[ConversationItem],
    parallel_state,
) -> torch.Tensor:
    conversation = [items]
    with (
        use_parallel_state(parallel_state),
        torch.autocast(
            device_type=get_device_type(),
            dtype=torch.bfloat16,
        ),
    ):
        inputs = model.forward_pre(conversation_list=conversation)
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

    outer_state = init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
    )
    module_state = init_parallel_state(
        ulysses_size=world_size,
        dp_mode="fsdp2",
    )

    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    config_kwargs = {
        **tiny_bagel_qwen2_cfg(),
        "hidden_size": 112,
        "intermediate_size": 224,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "attn_implementation": "veomni_flash_attention_2_with_sp",
    }
    torch.manual_seed(9102)
    reference = BagelQwen2MoT(BagelQwen2MoTConfig(**config_kwargs)).to(device=device, dtype=torch.bfloat16).train()
    torch.manual_seed(9102)
    sequence_parallel = (
        BagelQwen2MoT(BagelQwen2MoTConfig(**config_kwargs)).to(device=device, dtype=torch.bfloat16).train()
    )
    sequence_parallel.load_state_dict(reference.state_dict())

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in sequence_parallel.model.layers:
        fully_shard(layer, mesh=module_state.fsdp_mesh, mp_policy=mp_policy)
    fully_shard(sequence_parallel, mesh=module_state.fsdp_mesh, mp_policy=mp_policy)
    _enable_scoped_gradient_checkpointing(sequence_parallel, module_state)

    hidden_size = int(reference.config.hidden_size)
    reference_items, reference_inputs = _rank_items(rank, device, hidden_size)
    sp_items, sp_inputs = _rank_items(rank, device, hidden_size)

    reference_hidden = _forward_carrier(reference, reference_items, outer_state)
    sp_hidden = _forward_carrier(sequence_parallel, sp_items, module_state)
    torch.testing.assert_close(sp_hidden, reference_hidden, rtol=2e-2, atol=2e-2)

    reference_loss = reference_hidden.float().square().mean()
    sp_loss = sp_hidden.float().square().mean()
    reference_loss.backward()
    sp_loss.backward()

    for reference_input, sp_input in zip(reference_inputs, sp_inputs, strict=True):
        assert reference_input.grad is not None
        assert sp_input.grad is not None
        assert torch.isfinite(sp_input.grad).all()
        torch.testing.assert_close(sp_input.grad, reference_input.grad, rtol=3e-2, atol=3e-2)

    reference_grads: dict[str, torch.Tensor] = {}
    for name, parameter in reference.named_parameters():
        assert parameter.grad is not None, f"reference parameter {name} did not participate in backward"
        grad = parameter.grad.detach().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        reference_grads[name] = grad / world_size

    for name, parameter in sequence_parallel.named_parameters():
        assert parameter.grad is not None, f"SP parameter {name} did not participate in backward"
        assert torch.isfinite(
            parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        ).all()
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
