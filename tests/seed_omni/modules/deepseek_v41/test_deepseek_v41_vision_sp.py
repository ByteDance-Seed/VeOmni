# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4.1 vision tower under Ulysses sequence parallelism against the unsharded forward.

Gloo on CPU with the SDPA varlen path. The packed batch holds two images with grids
``(3, 4)`` and ``(5, 3)`` (``12 + 15 = 27`` patches), so both ``world_size=2`` and
``world_size=4`` pad the patch stream, a shard boundary falls inside an image, and
``downsample_ratio=2`` exercises the aligner's edge padding on partial spatial cells.

Each rank's loss is the replicated downstream loss on the gathered ``image_embeds``;
FSDP averages gradients over ``dp_shard_sp``, so the rank-summed gradient divided by
``world_size`` must equal the unsharded gradient for every ViT, aligner, and framing
delimiter parameter.
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.distributed.sequence_parallel import sp_pad_and_slice
from veomni.models.seed_omni.modules.deepseek_v41.vision.accelerated.accelerated import (
    DeepseekV41VisionEncoderAccelerated,
)
from veomni.models.seed_omni.modules.deepseek_v41.vision.configuration import DeepseekV41VisionEncoderConfig
from veomni.models.seed_omni.modules.deepseek_v41.vision.modeling import (
    DeepseekV41VisionEncoder,
    build_deepseek_v41_vit_metadata,
)
from veomni.models.seed_omni.modules.deepseek_v41.vision.processing import (
    _OMNI_TOKEN_TYPES,
    _OMNI_VIT_GRID,
    image_token_types,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem


GRIDS = [[3, 4], [5, 3]]
OUT_HIDDEN = 32


def tiny_vision_config() -> DeepseekV41VisionEncoderConfig:
    return DeepseekV41VisionEncoderConfig(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        patch_size=2,
        downsample_ratio=2,
        max_image_tokens=64,
        min_pixels=16,
        out_hidden_size=OUT_HIDDEN,
    )


def _token_types_for_grids(grids: list[list[int]], ratio: int) -> list[torch.Tensor]:
    return [image_token_types(math.ceil(h / ratio), math.ceil(w / ratio)) for h, w in grids]


def _inputs(config: DeepseekV41VisionEncoderConfig):
    generator = torch.Generator().manual_seed(7)
    total_patches = sum(h * w for h, w in GRIDS)
    pixel_values = torch.randn(
        total_patches,
        3,
        config.patch_size,
        config.patch_size,
        generator=generator,
    )
    token_types = _token_types_for_grids(GRIDS, config.downsample_ratio)
    total_span = sum(int(t.numel()) for t in token_types)
    weights = torch.randn(total_span, config.out_hidden_size, generator=generator)
    return pixel_values, GRIDS, token_types, weights


def _worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        from veomni.models.seed_omni.modules.deepseek_v41.vision import modeling as vision_modeling

        vision_modeling.USE_FLASH_ATTN_VARLEN = False

        config = tiny_vision_config()
        torch.manual_seed(0)
        model = DeepseekV41VisionEncoder(config).float().train()
        with torch.no_grad():
            model.image_start.copy_(torch.linspace(-0.2, 0.2, OUT_HIDDEN))
            model.image_end.copy_(torch.linspace(0.1, -0.3, OUT_HIDDEN))
            model.image_newline.copy_(torch.linspace(-0.1, 0.3, OUT_HIDDEN))

        pixel_values, grids, token_types, weights = _inputs(config)

        _init_parallel_state(dp_size=world_size, ulysses_size=1, device_type="cpu", name=None)
        ref_pixels = pixel_values.clone().requires_grad_(True)
        ref_meta = build_deepseek_v41_vit_metadata(grids, config.rope_dim, config.rope_theta, ref_pixels.device)
        ref_out = model(
            pixel_values=ref_pixels,
            grids=grids,
            token_types=token_types,
            vit_metadata=ref_meta,
        )["image_embeds"]
        ref_loss = (ref_out * weights).sum()
        ref_loss.backward()
        ref_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        assert len(ref_grads) == len(list(model.named_parameters()))
        model.zero_grad(set_to_none=True)
        clear_parallel_state()

        _init_parallel_state(dp_size=1, ulysses_size=world_size, device_type="cpu", name=None)
        sp_pixels = pixel_values.clone().requires_grad_(True)
        sp_meta = build_deepseek_v41_vit_metadata(grids, config.rope_dim, config.rope_theta, sp_pixels.device)
        local_pixels = sp_pad_and_slice(sp_pixels, dim=0, pad_value=0)
        sp_out = model(
            pixel_values=local_pixels,
            grids=grids,
            token_types=token_types,
            vit_metadata=sp_meta,
        )["image_embeds"]
        sp_loss = (sp_out * weights).sum()
        sp_loss.backward()

        torch.testing.assert_close(sp_out, ref_out, rtol=1e-5, atol=1e-5)

        mean_loss = sp_loss.detach().clone()
        dist.all_reduce(mean_loss)
        torch.testing.assert_close(mean_loss / world_size, ref_loss.detach(), rtol=1e-5, atol=1e-5)

        input_grad = sp_pixels.grad.clone()
        dist.all_reduce(input_grad)
        torch.testing.assert_close(input_grad / world_size, ref_pixels.grad, rtol=1e-4, atol=1e-5)

        sp_grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        assert sp_grads.keys() == ref_grads.keys()
        for name, grad in sp_grads.items():
            dist.all_reduce(grad)
            torch.testing.assert_close(
                grad / world_size,
                ref_grads[name],
                rtol=1e-4,
                atol=1e-5,
                msg=lambda msg, name=name: f"{name}: {msg}",
            )
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo required")
@pytest.mark.parametrize("world_size", [2, 4])
def test_sharded_vision_forward_and_gradients_match_unsharded(world_size: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        mp.spawn(_worker, args=(world_size, init_file), nprocs=world_size, join=True)


class TestVisionSequenceParallelHooks:
    def test_forward_pre_pads_and_slices_patches_and_isolates_padding_in_cu_seqlens(self, monkeypatch):
        import veomni.distributed.sequence_parallel.data as sp_data
        import veomni.models.seed_omni.modules.deepseek_v41.vision.accelerated.accelerated as accelerated
        import veomni.models.seed_omni.modules.deepseek_v41.vision.modeling as vision_modeling

        class FakeState:
            sp_enabled = True
            sp_size = 4
            sp_rank = 3

        monkeypatch.setattr(accelerated, "get_parallel_state", lambda: FakeState())
        monkeypatch.setattr(vision_modeling, "get_parallel_state", lambda: FakeState())
        monkeypatch.setattr(sp_data, "get_parallel_state", lambda: FakeState())
        config = tiny_vision_config()
        model = DeepseekV41VisionEncoderAccelerated(config)

        pixel_values, grids, token_types, _ = _inputs(config)
        # 12 + 15 = 27 patches -> padded to 28 across 4 ranks (7 per rank).
        p0, p1 = pixel_values[:12], pixel_values[12:]
        conversation = [
            [
                ConversationItem(
                    type="image",
                    value=p0,
                    role="user",
                    meta={_OMNI_VIT_GRID: grids[0], _OMNI_TOKEN_TYPES: token_types[0]},
                ),
                ConversationItem(
                    type="image",
                    value=p1,
                    role="user",
                    meta={_OMNI_VIT_GRID: grids[1], _OMNI_TOKEN_TYPES: token_types[1]},
                ),
            ]
        ]

        out = model.forward_pre(conversation_list=conversation)
        meta = out["vit_metadata"]

        assert out["pixel_values"].shape == (7, 3, config.patch_size, config.patch_size)
        # Rank 3 holds patches [21..27) plus 1 zero-padded patch at index 27.
        torch.testing.assert_close(out["pixel_values"][:6], pixel_values[21:27])
        assert out["pixel_values"][6:].abs().sum() == 0
        assert meta["cu_seqlens"].tolist() == [0, 12, 27, 28]
        assert meta["cos"].shape[0] == 7
        assert meta["sin"].shape[0] == 7
        assert meta["cos"][6:].abs().sum() == 0
        assert meta["sin"][6:].abs().sum() == 0
