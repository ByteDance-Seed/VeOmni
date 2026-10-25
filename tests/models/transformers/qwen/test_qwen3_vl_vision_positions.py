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

"""On-device Qwen3-VL vision position embeddings against per-grid references.

The patched ``fast_pos_embed_interpolate`` and ``rot_pos_emb`` build the whole
batch on the accelerator from a host grid list. This checks them on real device
tensors with many large, mixed image and video grids, against straightforward
per-grid implementations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.models.compare import eager_ops_config, ops_config_scope
from veomni.models import build_foundation_model
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type


pytestmark = pytest.mark.skipif(not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE), reason="needs a CUDA GPU or Ascend NPU")

_TOY_CONFIG = Path(__file__).resolve().parents[3] / "toy_config" / "qwen3vl_toy"


def _random_grid_thw(
    batch: int,
    spatial_merge_size: int,
    device: str,
    min_t: int = 1,
    max_t: int = 4,
    min_hw: int = 14,
    max_hw: int = 56,
) -> torch.Tensor:
    t = torch.randint(min_t, max_t, (batch,))
    h = torch.randint(min_hw, max_hw, (batch,)) * spatial_merge_size
    w = torch.randint(min_hw, max_hw, (batch,)) * spatial_merge_size
    return torch.stack((t, h, w), dim=1).to(device)


def _fast_pos_embed_interpolate_reference(visual: torch.nn.Module, grid_thw: torch.Tensor) -> torch.Tensor:
    grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
    num_grid_per_side = visual.num_grid_per_side
    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]
    for h, w in zip(grid_hs, grid_ws, strict=True):
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h, dtype=torch.float64)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w, dtype=torch.float64)
        h_floor, w_floor = h_idxs.int(), w_idxs.int()
        h_ceil = (h_floor + 1).clip(max=num_grid_per_side - 1)
        w_ceil = (w_floor + 1).clip(max=num_grid_per_side - 1)
        dh, dw = h_idxs - h_floor, w_idxs - w_floor
        base_h, base_h_ceil = h_floor * num_grid_per_side, h_ceil * num_grid_per_side
        indices = [
            (base_h[None].T + w_floor[None]).flatten(),
            (base_h[None].T + w_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_floor[None]).flatten(),
            (base_h_ceil[None].T + w_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]
        for corner in range(4):
            idx_list[corner].extend(indices[corner].tolist())
            weight_list[corner].extend(weights[corner].tolist())

    weight = visual.pos_embed.weight
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=weight.device)
    weight_tensor = torch.tensor(weight_list, dtype=weight.dtype, device=weight.device)
    pos_embeds = visual.pos_embed(idx_tensor) * weight_tensor[:, :, None]
    patch_pos_embeds = (pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]).split(
        [int(h * w) for h, w in zip(grid_hs, grid_ws)]
    )

    merge_size = visual.spatial_merge_size
    permuted = []
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        permuted.append(pos_embed)
    return torch.cat(permuted)


def _rot_pos_emb_reference(visual: torch.nn.Module, grid_thw: torch.Tensor) -> torch.Tensor:
    merge_size = visual.spatial_merge_size
    device = visual.rotary_pos_emb.inv_freq.device
    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
    offset = 0
    for num_frames, height, width in grid_thw:
        merged_h, merged_w = height // merge_size, width // merge_size
        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra = torch.arange(merge_size, device=device)
        row_idx = block_rows[:, None, None, None] * merge_size + intra[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra[None, None, None, :]
        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)
        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)
        pos_ids[offset : offset + coords.shape[0]] = coords
        offset += coords.shape[0]
    return visual.rotary_pos_emb(pos_ids)


@pytest.fixture(scope="module")
def visual() -> torch.nn.Module:
    device = get_device_type()
    with ops_config_scope(eager_ops_config()):
        model = build_foundation_model(config_path=str(_TOY_CONFIG), init_device=device)
    visual = model.model.visual
    torch.manual_seed(0)
    visual.pos_embed.weight = torch.nn.Parameter(torch.randn_like(visual.pos_embed.weight))
    return visual.to(device)


@pytest.mark.parametrize("batch", (10, 20, 30))
@torch.no_grad()
def test_fast_pos_embed_interpolate_matches_per_grid_reference_on_device(visual: torch.nn.Module, batch: int):
    torch.manual_seed(batch)
    grid_thw = _random_grid_thw(batch, visual.spatial_merge_size, device=get_device_type())

    actual = visual.fast_pos_embed_interpolate(grid_thw)
    expected = _fast_pos_embed_interpolate_reference(visual, grid_thw)

    assert actual.device.type == get_device_type()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch", (10, 20, 30))
@torch.no_grad()
def test_rot_pos_emb_matches_per_grid_reference_on_device(visual: torch.nn.Module, batch: int):
    torch.manual_seed(batch)
    grid_thw = _random_grid_thw(batch, visual.spatial_merge_size, device=get_device_type())

    actual = visual.rot_pos_emb(grid_thw)
    expected = _rot_pos_emb_reference(visual, grid_thw)

    assert actual.device.type == get_device_type()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
