"""Pre-SP pixel-stream merge through MainCollator (exactly-once ViT under SP).

Drives the real model hooks (``merge_pixel_streams_pre_sp`` +
``collate_multimodal_metadata`` from the qwen3_vl generated modeling) through
``MainCollator`` with a monkeypatched sp_size=2 parallel state, and asserts:

1. the raw ``pixel_values`` / ``pixel_values_videos`` are replaced by one
   ``pixel_values_merged`` stream (image rows first), SP-padded once and
   sliced per rank;
2. the two rank slices reassemble exactly to ``cat(image, video, pad)``;
3. ``multimodal_metadata`` carries the merged cu_seqlens (image frames, then
   video frames, then one sp-pad tail), max_seqlen, and the global image
   patch-row count used by Model.forward to split the gathered features.

CPU-only; mirrors the monkeypatch pattern of ``test_collators.py`` and the
hook-protocol coverage of ``test_mm_metadata.py``.
"""

import os
import pickle
import types

import pytest
import torch


os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")


def _fake_ps(sp_enabled: bool, sp_size: int = 1, sp_rank: int = 0):
    return types.SimpleNamespace(sp_enabled=sp_enabled, sp_size=sp_size, sp_rank=sp_rank)


@pytest.fixture(scope="module")
def hooks():
    module = pytest.importorskip("veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu")
    return module.merge_pixel_streams_pre_sp, module.collate_multimodal_metadata


def _mixed_feature(feat_dim: int = 8):
    # One packed sample carrying an image (1x2x2 -> 4 patch rows -> 1 token)
    # and a video (2x2x2 -> 8 patch rows -> 2 tokens) plus text.
    gen = torch.Generator().manual_seed(0)
    seq_len = 8
    return {
        "input_ids": torch.arange(10, 10 + seq_len, dtype=torch.long),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "labels": torch.arange(10, 10 + seq_len, dtype=torch.long),
        "position_ids": torch.arange(seq_len, dtype=torch.int64),
        "pixel_values": torch.randn(4, feat_dim, generator=gen),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        "pixel_values_videos": torch.randn(8, feat_dim, generator=gen),
        "video_grid_thw": torch.tensor([[2, 2, 2]], dtype=torch.long),
    }


def test_pre_sp_merge_slices_one_stream(monkeypatch, hooks):
    import veomni.data.data_collator as m

    pre_sp_hook, metadata_hook = hooks
    sp_size = 2
    feature = _mixed_feature()
    expected_merged = torch.cat([feature["pixel_values"], feature["pixel_values_videos"]], dim=0)

    rank_slices = []
    for sp_rank in range(sp_size):
        monkeypatch.setattr(m, "get_parallel_state", lambda r=sp_rank: _fake_ps(True, sp_size, r))
        collator = m.MainCollator(
            metadata_collate_func=metadata_hook,
            pre_sp_collate_func=pre_sp_hook,
        )
        out = collator([{k: v.clone() for k, v in feature.items()}])

        assert "pixel_values" not in out
        assert "pixel_values_videos" not in out
        assert "pixel_values_merged" in out
        rank_slices.append(out["pixel_values_merged"])

        md = out["multimodal_metadata"]
        assert md["merged_grid_thw_list"] == [[1, 2, 2], [2, 2, 2]]
        # image frame (4 rows), two video frames (4 rows each), sp-pad tail
        # (12 rows -> padded to sp_size * pad_scale=8 multiple = 16 -> pad 4).
        assert md["vit_merged_cu_seqlens"].tolist() == [0, 4, 8, 12, 16]
        assert md["vit_merged_max_seqlen"] == 4
        assert md["vit_merged_n_image_rows"] == 4
        # No per-modality metadata in merged mode.
        assert "vit_image_cu_seqlens" not in md
        assert "vit_video_cu_seqlens" not in md
        # The hook chain must stay picklable for spawned DataLoader workers.
        pickle.dumps((pre_sp_hook, metadata_hook))

    # 16 padded rows split evenly across the two ranks...
    assert all(s.shape[0] == 8 for s in rank_slices)
    # ...and reassemble to image rows, then video rows, then a zero pad tail.
    reassembled = torch.cat(rank_slices, dim=0)
    assert torch.equal(reassembled[:12], expected_merged)
    assert torch.equal(reassembled[12:], torch.zeros(4, expected_merged.shape[1]))


def test_pre_sp_merge_single_modality_and_text_only(monkeypatch, hooks):
    import veomni.data.data_collator as m

    pre_sp_hook, metadata_hook = hooks
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(True, 2, 0))
    collator = m.MainCollator(metadata_collate_func=metadata_hook, pre_sp_collate_func=pre_sp_hook)

    # Image-only: the merged stream is just the image stream.
    feature = _mixed_feature()
    feature.pop("pixel_values_videos")
    feature.pop("video_grid_thw")
    out = collator([feature])
    assert "pixel_values" not in out and "pixel_values_merged" in out
    md = out["multimodal_metadata"]
    assert md["merged_grid_thw_list"] == [[1, 2, 2]]
    # 4 image rows padded to 8 (sp_size * pad_scale=8): one pad-tail segment.
    assert md["vit_merged_cu_seqlens"].tolist() == [0, 4, 8]
    assert md["vit_merged_n_image_rows"] == 4

    # Text-only: hook is a no-op; no merged key, no metadata.
    feature = _mixed_feature()
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        feature.pop(key)
    out = collator([feature])
    assert "pixel_values_merged" not in out
    assert "multimodal_metadata" not in out
