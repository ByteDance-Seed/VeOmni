"""Distributed coverage for BAGEL text-encoder sequence parallelism."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls
from tests.tools.launch_utils import torchrun
from veomni.distributed.parallel_state import init_parallel_state, use_parallel_state
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.ops.kernels.cross_entropy import install_loss_mapping
from veomni.utils.device import get_device_type, get_torch_device


def _text_encoder_sp_worker() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    device = torch.device(f"{get_device_type()}:{rank}")
    module_state = init_parallel_state(ulysses_size=world_size, dp_mode="ddp")
    install_loss_mapping("eager")

    BagelTextEncoder = model_cls("bagel_text_encoder")
    BagelTextEncoderConfig = config_cls("bagel_text_encoder")
    with use_parallel_state(module_state):
        torch.manual_seed(7301)
        model = BagelTextEncoder(
            BagelTextEncoderConfig(
                vocab_size=32,
                hidden_size=8,
                tie_word_embeddings=False,
            )
        ).to(device=device, dtype=torch.float32)

    # Uniform SP replicates the same DP sample across every SP rank. Use an odd
    # sequence length so encode_pre must pad to six tokens before slicing it into
    # two equal local shards.
    input_ids = torch.arange(5, device=device, dtype=torch.long)
    item = ConversationItem(
        type="text",
        value=input_ids,
        role="assistant",
        meta={
            "_omni_tokenized": True,
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        },
    )
    conversation = [[item]]

    with use_parallel_state(module_state):
        encode_inputs = model.encode_pre(conversation_list=conversation)
        assert encode_inputs["input_ids"].shape == (3,)
        encoded = model.encode(**encode_inputs)
        model.encode_post(**encoded)

    expected_embeds = model.embed_tokens(input_ids)
    torch.testing.assert_close(item.value, expected_embeds)
    item.value.retain_grad()

    with use_parallel_state(module_state):
        decode_inputs = model.decode_pre(conversation_list=conversation)
        assert decode_inputs["hidden_states"].shape == (5, model.config.hidden_size)
        decoded = model.decode(**decode_inputs)
        outputs = model.decode_post(**decoded)

    loss = outputs["_loss"]
    assert torch.isfinite(loss)
    gathered_losses = [torch.empty_like(loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, loss.detach())
    for gathered_loss in gathered_losses:
        torch.testing.assert_close(gathered_loss, loss.detach())

    loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
    assert item.value.grad is not None
    assert torch.isfinite(item.value.grad).all()

    dist.barrier()


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
def test_bagel_text_encoder_sp2_handles_replicated_padded_sequence() -> None:
    torchrun(_text_encoder_sp_worker, world_size=2)
