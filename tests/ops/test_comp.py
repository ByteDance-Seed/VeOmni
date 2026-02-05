import argparse
import os
import time

import pytest
import torch

from veomni.models import build_foundation_model
from veomni.utils.device import get_device_type, synchronize


def generate_grid_thw(batch: int, spatial_merge_size: int, min_t=1, max_t=4, min_hw=14, max_hw=56, device="cpu"):
    grid_thw = torch.zeros((batch, 3), dtype=torch.long, device=device)

    for i in range(batch):
        t = torch.randint(min_t, max_t, (1,)).item()
        h = torch.randint(min_hw, max_hw, (1,)).item()
        w = torch.randint(min_hw, max_hw, (1,)).item()
        grid_thw[i, 0] = t
        grid_thw[i, 1] = h * spatial_merge_size
        grid_thw[i, 2] = w * spatial_merge_size
    return grid_thw


def fast_pos_embed_interpolate_ref(self, grid_thw):
    grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    device = self.pos_embed.weight.device
    dtype = self.pos_embed.weight.dtype
    for t, h, w in zip(grid_ts, grid_hs, grid_ws):
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, dtype=torch.float64)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, dtype=torch.float64)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=dtype, device=device)

    pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

    patch_pos_embeds_permute = []
    merge_size = self.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    return patch_pos_embeds


def rot_pos_emb_ref(self, grid_thw):
    merge_size = self.spatial_merge_size

    max_hw = int(grid_thw[:, 1:].max().item())
    freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
    device = freq_table.device

    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw:
        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(merged_h, device=device)  # block row indices
        block_cols = torch.arange(merged_w, device=device)  # block col indices
        intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
        intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

        # Compute full-resolution positions
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    embeddings = freq_table[pos_ids]  # lookup rotary embeddings
    embeddings = embeddings.flatten(1)
    return embeddings


def compare(args, model, grid_thw, ref_fn, test_fn):
    print(f"Testing {ref_fn.__name__} vs {test_fn.__name__}...")
    # warmup
    for _ in range(args.warmups):
        ref_fn(model, grid_thw)
        test_fn(grid_thw)

    # test for bitwise equal
    ref_result = ref_fn(model, grid_thw)
    test_result = test_fn(grid_thw)
    assert torch.allclose(ref_result, test_result, equal_nan=True)
    print("Bitwise equal test passed!")

    # test for time overhead
    synchronize()
    start_time = time.time()
    for _ in range(args.iters):
        ref_fn(model, grid_thw)
    synchronize()
    ref_time = time.time() - start_time

    synchronize()
    start_time = time.time()
    for _ in range(args.iters):
        test_fn(grid_thw)
    synchronize()
    test_time = time.time() - start_time

    print(f"Reference time: {ref_time / args.iters:.6f} seconds")
    print(f"Test time: {test_time / args.iters:.6f} seconds")
    print(f"Speedup: {(ref_time - test_time) / ref_time:.2f}x")

    return ref_result, test_result


def main(args):
    # Initialize the model and get the visual
    dummy_model = build_foundation_model(
        config_path=args.config_path,
        init_device=args.device,
    ).model.visual

    dummy_model.pos_embed.weight = torch.nn.Parameter(torch.randn_like(dummy_model.pos_embed.weight))
    try:
        dummy_model = dummy_model.to(args.device)
    except Exception as e:
        print(f"Error moving dummy model to device {args.device}: {e}")
        return

    grid_thw = generate_grid_thw(
        batch=args.batch, spatial_merge_size=dummy_model.spatial_merge_size, device=args.device
    )
    print(f"grid_thw: {grid_thw}")
    compare(args, dummy_model, grid_thw, fast_pos_embed_interpolate_ref, dummy_model.fast_pos_embed_interpolate)
    compare(args, dummy_model, grid_thw, rot_pos_emb_ref, dummy_model.rot_pos_emb)


test_cases = [
    {"warmups": 5, "iters": 100, "batch": 10},
    {"warmups": 5, "iters": 100, "batch": 20},
    {"warmups": 5, "iters": 100, "batch": 30},
]


@pytest.mark.parametrize("args", test_cases)
def test_comp(args: dict):
    args = argparse.Namespace(**args)
    args.device = get_device_type()
    args.config_path = os.path.dirname(os.path.abspath(__file__)) + "/../toy_config/qwen3vl_toy"
    print(f"{args=}")
    main(args)
