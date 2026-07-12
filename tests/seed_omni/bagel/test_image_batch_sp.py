"""Distributed coverage for BAGEL VAE/SigLIP image-batch redistribution."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls
from tests.tools.launch_utils import torchrun
from veomni.distributed.parallel_state import init_parallel_state, use_parallel_state
from veomni.models.seed_omni.modules.bagel.siglip_navit.modulemixin import (
    _OMNI_POSITION_IDS,
    _OMNI_TOKEN_LEN,
)
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.modules.bagel.vae.modulemixin import BAGEL_VAE_PIXEL_SHAPE
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.utils.device import get_device_type, get_torch_device


def _vae_items(rank: int, device: torch.device) -> tuple[list[ConversationItem], list[torch.Tensor]]:
    height, width = (2, 4) if rank == 0 else (4, 2)
    values = [
        torch.full(
            (3, height, width),
            fill_value=rank * 10 + index + 1,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        for index in range(rank + 1)
    ]
    items = [
        ConversationItem(
            type="image",
            value=value,
            role="assistant",
            source=BAGEL_VAE_CONTEXT,
            meta={BAGEL_VAE_PIXEL_SHAPE: torch.tensor([height, width])},
        )
        for value in values
    ]
    return items, values


def _siglip_items(rank: int, device: torch.device) -> tuple[list[ConversationItem], list[torch.Tensor]]:
    token_lens = [2] if rank == 0 else [1, 4]
    values = [
        torch.full(
            (token_len, 4),
            fill_value=rank * 10 + index + 1,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        for index, token_len in enumerate(token_lens)
    ]
    items = [
        ConversationItem(
            type="image",
            value=value,
            role="user",
            source=BAGEL_SIGLIP_CONTEXT,
            meta={
                _OMNI_POSITION_IDS: torch.arange(value.shape[0], device=device),
                _OMNI_TOKEN_LEN: int(value.shape[0]),
            },
        )
        for value in values
    ]
    return items, values


def _image_batch_sp_worker() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    device = torch.device(f"{get_device_type()}:{rank}")
    module_state = init_parallel_state(ulysses_size=world_size, dp_mode="ddp")

    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    vae = BagelVAE(
        BagelVAEConfig(
            resolution=4,
            downsample=1,
            ch=32,
            ch_mult=[1],
            num_res_blocks=0,
            z_channels=1,
            freeze=False,
        )
    ).to(device=device, dtype=torch.float32)
    vae_items, vae_values = _vae_items(rank, device)
    with use_parallel_state(module_state):
        vae_inputs = vae.encode_pre(conversation_list=[vae_items])
        assert vae_inputs["pixel_values"].shape == (2, 3, 4, 4)
        vae_outputs = vae.encode_post(vae_inputs["pixel_values"][:, :1])

    assert vae_outputs["conversation_list"] == [vae_items]
    for item, original in zip(vae_items, vae_values, strict=True):
        torch.testing.assert_close(item.value, original[:1])
    sum(item.value.sum() for item in vae_items).backward()
    for value in vae_values:
        assert value.grad is not None
        torch.testing.assert_close(value.grad[0], torch.ones_like(value.grad[0]))
        torch.testing.assert_close(value.grad[1:], torch.zeros_like(value.grad[1:]))

    BagelSiglip = model_cls("bagel_siglip_navit")
    BagelSiglipConfig = config_cls("bagel_siglip_navit")
    siglip = BagelSiglip(
        BagelSiglipConfig(
            hidden_size=8,
            output_size=4,
            image_size=4,
            min_image_size=1,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_channels=1,
            patch_size=2,
            vit_max_num_patch_per_side=2,
        )
    ).to(device=device, dtype=torch.float32)
    siglip_items, siglip_values = _siglip_items(rank, device)
    with use_parallel_state(module_state):
        siglip_inputs = siglip.forward_pre(conversation_list=[siglip_items])
        expected_local_images = 2 if rank == 0 else 1
        assert int(siglip_inputs["token_lens"].numel()) == expected_local_images
        image_embeds = siglip_inputs["patchified_pixel_values"][:, :1].repeat(1, 4)
        siglip_outputs = siglip.forward_post(image_embeds, siglip_inputs["token_lens"])

    assert siglip_outputs["conversation_list"] == [siglip_items]
    for item, original in zip(siglip_items, siglip_values, strict=True):
        torch.testing.assert_close(item.value, original[:, :1].repeat(1, 4))
    sum(item.value.sum() for item in siglip_items).backward()
    for value in siglip_values:
        assert value.grad is not None
        torch.testing.assert_close(value.grad[:, :1], torch.full_like(value.grad[:, :1], 4.0))
        torch.testing.assert_close(value.grad[:, 1:], torch.zeros_like(value.grad[:, 1:]))

    dist.barrier()


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
def test_bagel_vae_and_siglip_redistribute_whole_images_across_sp2() -> None:
    torchrun(_image_batch_sp_worker, world_size=2)
