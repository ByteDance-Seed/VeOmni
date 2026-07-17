"""Distributed coverage for BAGEL image-module sequence parallelism."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.tools.launch_utils import torchrun
from veomni.distributed.parallel_state import init_parallel_state, use_parallel_state
from veomni.models.seed_omni.graphs.dispatch import run_sp_looped_endpoint
from veomni.models.seed_omni.modules.bagel.siglip_navit.modulemixin import (
    _OMNI_POSITION_IDS,
    _OMNI_TOKEN_LEN,
    BagelSiglipNavitModuleMixin,
)
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.modules.bagel.vae.modulemixin import BAGEL_VAE_PIXEL_SHAPE, BagelVAEModuleMixin
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.utils.device import get_device_type, get_torch_device


def _vae_items(rank: int, device: torch.device) -> tuple[list[ConversationItem], list[torch.Tensor]]:
    canvas = (2, 4) if rank == 0 else (4, 4)
    pixel_shapes = [(2, 4)] if rank == 0 else [(4, 2), (2, 4)]
    values: list[torch.Tensor] = []
    for index, (height, width) in enumerate(pixel_shapes):
        value = torch.zeros((3, *canvas), device=device, dtype=torch.float32)
        value[:, :height, :width] = rank * 10 + index + 1
        values.append(value.requires_grad_())
    items = [
        ConversationItem(
            type="image",
            value=value,
            role="assistant",
            source=BAGEL_VAE_CONTEXT,
            meta={BAGEL_VAE_PIXEL_SHAPE: torch.tensor(pixel_shape)},
        )
        for value, pixel_shape in zip(values, pixel_shapes, strict=True)
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


class _BagelSiglipLoopHarness(BagelSiglipNavitModuleMixin, torch.nn.Module):
    """Exercise BAGEL's carrier + SP hooks without loading the real vision weights."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._device = device
        self.forward_token_lens: list[list[int]] = []

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def forward(
        self,
        patchified_pixel_values: torch.Tensor,
        patchified_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        token_lens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del patchified_position_ids
        expected_cu = torch.nn.functional.pad(torch.cumsum(token_lens, dim=0), (1, 0)).to(torch.int32)
        torch.testing.assert_close(cu_seqlens, expected_cu)
        assert max_seqlen == int(token_lens.max().item())
        self.forward_token_lens.append(token_lens.detach().cpu().tolist())
        return {
            "image_embeds": patchified_pixel_values[:, :1].repeat(1, 4),
            "token_lens": token_lens,
        }


class _BagelVAELoopHarness(BagelVAEModuleMixin, torch.nn.Module):
    """Exercise BAGEL VAE carrier + SP hooks without loading codec weights."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.config = SimpleNamespace(downsample=1, z_channels=1)
        self._device = device
        self.encode_batch_shapes: list[tuple[int, ...]] = []
        self.offline_encode_batch_shapes: list[tuple[int, ...]] = []

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def encode(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        self.encode_batch_shapes.append(tuple(pixel_values.shape))
        return {"latents": pixel_values[:, :1]}

    def offline_encode(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        self.offline_encode_batch_shapes.append(tuple(pixel_values.shape))
        return {"encoded_cache": pixel_values[:, :2]}

    def online_process(self, **kwargs: object) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def _image_batch_sp_worker() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    device = torch.device(f"{get_device_type()}:{rank}")
    module_state = init_parallel_state(ulysses_size=world_size, dp_mode="ddp")

    vae = _BagelVAELoopHarness(device)
    vae_items, vae_values = _vae_items(rank, device)
    with use_parallel_state(module_state):
        vae_inputs = vae.encode_pre(conversation_list=[vae_items])
        assert vae.supports_sp("encode") is True
        assert vae.supports_sp("offline_encode") is True
        assert vae._metric_full_seqlens["encode"] == ([8] if rank == 0 else [8, 8])
        vae_outputs = run_sp_looped_endpoint(vae, vae, method="encode", kwargs=vae_inputs)
        vae_outputs = vae.encode_post(**vae_outputs)

    # Every VAE shard receives complete zero-padded image canvases. The second
    # sample deliberately contains two different real H/W values in one 4x4
    # canvas so post-forward cropping is exercised after looped all-gather.
    expected_vae_batch_shapes = [(1, 3, 2, 4), (1, 3, 4, 4)]
    assert vae.encode_batch_shapes == expected_vae_batch_shapes

    assert vae_outputs["conversation_list"] == [vae_items]
    for item, original in zip(vae_items, vae_values, strict=True):
        height, width = item.meta[BAGEL_VAE_PIXEL_SHAPE].tolist()
        torch.testing.assert_close(item.value, original[:1, :height, :width])
    sum(item.value.sum() for item in vae_items).backward()
    for item, value in zip(vae_items, vae_values, strict=True):
        assert value.grad is not None
        height, width = item.meta[BAGEL_VAE_PIXEL_SHAPE].tolist()
        expected_grad = torch.zeros_like(value)
        expected_grad[0, :height, :width] = 1
        torch.testing.assert_close(value.grad, expected_grad)

    offline_items, offline_values = _vae_items(rank, device)
    offline_pixel_shapes = [tuple(item.meta[BAGEL_VAE_PIXEL_SHAPE].tolist()) for item in offline_items]
    with use_parallel_state(module_state):
        offline_inputs = vae.encode_pre(conversation_list=[offline_items])
        offline_outputs = run_sp_looped_endpoint(vae, vae, method="offline_encode", kwargs=offline_inputs)
        offline_outputs = vae.offline_encode_post(**offline_outputs)

    assert vae.offline_encode_batch_shapes == expected_vae_batch_shapes
    assert offline_outputs["conversation_list"] == [offline_items]
    for item, original, (height, width) in zip(
        offline_items,
        offline_values,
        offline_pixel_shapes,
        strict=True,
    ):
        torch.testing.assert_close(
            item.value,
            original[:2, :height, :width].detach().reshape(2, 1, height, width),
        )

    siglip = _BagelSiglipLoopHarness(device)
    siglip_items, siglip_values = _siglip_items(rank, device)
    with use_parallel_state(module_state):
        siglip_inputs = siglip.forward_pre(conversation_list=[siglip_items])
        assert siglip.supports_sp("forward") is True
        assert siglip._metric_full_seqlens["forward"] == ([2] if rank == 0 else [5])
        siglip_outputs = run_sp_looped_endpoint(siglip, siglip, method="forward", kwargs=siglip_inputs)
        siglip_outputs = siglip.forward_post(**siglip_outputs)

    expected_forward_token_lens = [[2], [1]] if rank == 0 else [[1], [4]]
    assert siglip.forward_token_lens == expected_forward_token_lens

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
def test_bagel_vae_and_siglip_sample_loops_across_sp2() -> None:
    torchrun(_image_batch_sp_worker, world_size=2)
