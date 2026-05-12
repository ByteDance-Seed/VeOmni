"""Tests for the Qwen3.5-VL host-side ViT metadata precompute (``VisionMetadataCollator`` /
``compute_qwen3_5_vision_metadata``).

The metadata replaces what ``Qwen3_5VisionModel.forward`` / ``fast_pos_embed_interpolate`` /
``rot_pos_emb`` derive from the GPU ``grid_thw`` tensor on every forward (each a host-device sync).
Here we check the precomputed tensors against an independent brute-force reference of that logic,
and the collator wiring end-to-end.
"""

import types

import pytest
import torch

from veomni.data.data_transform import compute_qwen3_5_vision_metadata


# Toy Qwen3.5 vision config values (tests/toy_config/qwen3_5_toy): num_position_embeddings=2304,
# spatial_merge_size=2, hidden_size=1152, num_heads=16 -> head_dim=72.
NUM_GRID_PER_SIDE = int(2304**0.5)  # 48
SPATIAL_MERGE_SIZE = 2

GRID_CASES = [
    [[1, 4, 6]],  # single image, one frame
    [[1, 8, 8]],
    [[1, 2, 2], [1, 6, 4]],  # two images
    [[2, 4, 4]],  # a 2-frame "video": both frames share the per-position metadata
    [[1, 4, 6], [3, 2, 8], [1, 8, 8]],  # mixed images + video
]


def _ref_metadata(grid_thw_list, num_grid_per_side, m):
    """Brute-force reference for :func:`compute_qwen3_5_vision_metadata`.

    Mirrors ``Qwen3_5VisionModel.fast_pos_embed_interpolate`` (bilinear corners + weights),
    ``rot_pos_emb`` (row/col coords) and the ViT ``forward`` (per-frame ``cu_seqlens``), laid out in
    the spatial-merge-block + frame order ``(block_row, block_col, intra_row, intra_col)``.
    """
    g = num_grid_per_side
    pos_idx, pos_w, rot, cu = [], [], [], [0]
    max_hw = 0
    for t, h, w in grid_thw_list:
        h_lin = torch.linspace(0, g - 1, h, dtype=torch.float64).tolist()
        w_lin = torch.linspace(0, g - 1, w, dtype=torch.float64).tolist()
        mh, mw = h // m, w // m
        frame_idx, frame_w, frame_rot = [], [], []
        for br in range(mh):
            for bc in range(mw):
                for ir in range(m):
                    for ic in range(m):
                        r, c = br * m + ir, bc * m + ic
                        hi, wi = h_lin[r], w_lin[c]
                        hf, wf = int(hi), int(wi)
                        hc_, wc_ = min(hf + 1, g - 1), min(wf + 1, g - 1)
                        dh, dw = hi - hf, wi - wf
                        w11 = dh * dw
                        w10 = dh - w11
                        w01 = dw - w11
                        w00 = 1 - dh - w01
                        frame_idx.append([hf * g + wf, hf * g + wc_, hc_ * g + wf, hc_ * g + wc_])
                        frame_w.append([w00, w01, w10, w11])
                        frame_rot.append([r, c])
        for _ in range(t):
            pos_idx.extend(frame_idx)
            pos_w.extend(frame_w)
            rot.extend(frame_rot)
            cu.append(cu[-1] + h * w)
        max_hw = max(max_hw, h, w)
    return {
        "pos_embed_indices": torch.tensor(pos_idx, dtype=torch.long),
        "pos_embed_weights": torch.tensor(pos_w, dtype=torch.float32),
        "rot_pos_ids": torch.tensor(rot, dtype=torch.long),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32),
        "max_hw": max_hw,
    }


@pytest.mark.parametrize("grid_thw_list", GRID_CASES)
def test_compute_vision_metadata_matches_reference(grid_thw_list):
    got = compute_qwen3_5_vision_metadata(grid_thw_list, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE)
    exp = _ref_metadata(grid_thw_list, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE)

    n = sum(t * h * w for t, h, w in grid_thw_list)
    assert got["pos_embed_indices"].shape == (n, 4)
    assert got["pos_embed_weights"].shape == (n, 4)
    assert got["rot_pos_ids"].shape == (n, 2)
    assert got["cu_seqlens"].shape == (sum(t for t, _, _ in grid_thw_list) + 1,)
    assert got["pos_embed_indices"].dtype == torch.long
    assert got["rot_pos_ids"].dtype == torch.long
    assert got["cu_seqlens"].dtype == torch.int32
    assert isinstance(got["max_hw"], int)

    assert torch.equal(got["pos_embed_indices"], exp["pos_embed_indices"])
    assert torch.equal(got["rot_pos_ids"], exp["rot_pos_ids"])
    assert torch.equal(got["cu_seqlens"], exp["cu_seqlens"])
    assert torch.allclose(got["pos_embed_weights"], exp["pos_embed_weights"])
    assert got["max_hw"] == exp["max_hw"]
    # bilinear weights sum to 1 per token
    assert torch.allclose(got["pos_embed_weights"].sum(dim=1), torch.ones(n), atol=1e-5)


def test_compute_vision_metadata_parity_with_model():
    """Cross-check against the actual ``Qwen3_5VisionModel`` (requires transformers v5)."""
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

        from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import (
            Qwen3_5VisionModel,
        )
    except Exception as exc:  # transformers v4 env, or import wiring issue
        pytest.skip(f"Qwen3.5 v5 modeling not importable: {exc}")

    config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        num_position_embeddings=NUM_GRID_PER_SIDE**2,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        out_hidden_size=64,
    )
    torch.manual_seed(0)
    model = Qwen3_5VisionModel(config).eval()

    for grid_thw_list in GRID_CASES:
        meta = compute_qwen3_5_vision_metadata(grid_thw_list, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE)
        grid_thw = torch.tensor(grid_thw_list, dtype=torch.long)

        # pos embeds
        ref_pos = model.fast_pos_embed_interpolate(grid_thw)
        new_pos = (
            model.pos_embed(meta["pos_embed_indices"])
            * meta["pos_embed_weights"].to(model.pos_embed.weight.dtype).unsqueeze(-1)
        ).sum(dim=1)
        assert ref_pos.shape == new_pos.shape
        assert torch.allclose(ref_pos, new_pos, atol=1e-5, rtol=1e-4)

        # rotary pos emb
        ref_rot = model.rot_pos_emb(grid_thw)
        new_rot = model.rotary_pos_emb(meta["max_hw"])[meta["rot_pos_ids"]].flatten(1)
        assert ref_rot.shape == new_rot.shape
        assert torch.equal(ref_rot, new_rot)


def _fake_ps(sp_enabled: bool, sp_size: int = 1, sp_rank: int = 0):
    return types.SimpleNamespace(sp_enabled=sp_enabled, sp_size=sp_size, sp_rank=sp_rank)


def test_vision_metadata_collator_via_main_collator(monkeypatch):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))

    collator = m.MainCollator(
        vision_metadata_config={"num_grid_per_side": NUM_GRID_PER_SIDE, "spatial_merge_size": SPATIAL_MERGE_SIZE}
    )
    # ViT block must be present (VisionMetadataCollator runs after PackingCollator)
    pipeline_types = [type(p).__name__ for p in collator.preforward_pipeline]
    assert pipeline_types == ["PrecomputePositionIDsCollator", "PackingCollator", "VisionMetadataCollator"]

    feature = {
        "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "position_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 4, 6], [1, 2, 2]], dtype=torch.long),
    }
    out = collator([feature])
    n = 1 * 4 * 6 + 1 * 2 * 2
    assert out["vision_image_pos_embed_indices"].shape == (n, 4)
    assert out["vision_image_pos_embed_weights"].shape == (n, 4)
    assert out["vision_image_rot_pos_ids"].shape == (n, 2)
    assert out["vision_image_cu_seqlens"].shape == (3,)  # 2 frames -> 3 boundaries
    assert isinstance(out["vision_image_max_hw"], int) and out["vision_image_max_hw"] == 6
    assert "vision_video_pos_embed_indices" not in out  # no video in this batch

    exp = compute_qwen3_5_vision_metadata([[1, 4, 6], [1, 2, 2]], NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE)
    assert torch.equal(out["vision_image_pos_embed_indices"], exp["pos_embed_indices"])


def test_vision_metadata_disabled_by_default(monkeypatch):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))
    collator = m.MainCollator()  # no vision_metadata_config
    assert not any(type(p).__name__ == "VisionMetadataCollator" for p in collator.preforward_pipeline)
