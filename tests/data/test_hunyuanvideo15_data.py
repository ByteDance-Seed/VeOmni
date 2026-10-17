import json
from pathlib import Path

import pytest
import torch

from scripts.dataset.prepare_tom_and_jerry import prepare_dataset
from veomni.models.diffusers.hunyuanvideo15.conditioning import prepare_video


def test_aligned_manifest_and_no_overwrite(tmp_path):
    (tmp_path / "videos").mkdir()
    for name in ("one.mp4", "two.mp4"):
        (tmp_path / "videos" / name).write_bytes(b"fixture; decoding is tested separately")
    (tmp_path / "captions.txt").write_text("Tom runs.\nJerry hides.\n")
    (tmp_path / "videos.txt").write_text("videos/one.mp4\nvideos/two.mp4\n")
    output = tmp_path / "prepared/train.jsonl"
    metadata = prepare_dataset(tmp_path, output)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert metadata["samples"] == 2
    assert records[1]["prompt"] == "Jerry hides."
    assert Path(records[1]["video_bytes"]).name == "two.mp4"
    with pytest.raises(FileExistsError):
        prepare_dataset(tmp_path, output)


@pytest.mark.parametrize("videos", ["videos/missing.mp4\n", "../escape.mp4\n", "videos/a.mp4\nvideos/a.mp4\n"])
def test_reject_invalid_lists_before_writing(tmp_path, videos):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos/a.mp4").touch()
    (tmp_path / "captions.txt").write_text("caption\n" * len(videos.splitlines()))
    (tmp_path / "videos.txt").write_text(videos)
    output = tmp_path / "out.jsonl"
    with pytest.raises(ValueError):
        prepare_dataset(tmp_path, output)
    assert not output.exists()


def test_video_layout_range_and_frame_requirement():
    raw = torch.zeros(5, 3, 24, 32)
    raw[:, 1] = 255
    actual = prepare_video(raw, 16, 16, 5)
    assert actual.shape == (1, 3, 5, 16, 16)
    assert torch.all(actual[:, 0] == -1)
    torch.testing.assert_close(actual[:, 1], torch.ones_like(actual[:, 1]), atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        prepare_video(raw, 16, 16, 4)
