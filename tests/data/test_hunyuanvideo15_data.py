from pathlib import Path

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.diffusers.hunyuanvideo15.hunyuanvideo15_condition.conditioning import prepare_video


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_video_layout_range_and_frame_requirement():
    raw = torch.zeros(5, 3, 24, 32)
    raw[:, 1] = 255
    actual = prepare_video(raw, 16, 16, 5)
    assert actual.shape == (1, 3, 5, 16, 16)
    assert torch.all(actual[:, 0] == -1)
    torch.testing.assert_close(actual[:, 1], torch.ones_like(actual[:, 1]), atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        prepare_video(raw, 16, 16, 4)


def test_training_config_resizes_video_only_at_conditioning(monkeypatch):
    import sys

    from veomni.arguments import parse_args
    from veomni.data.multimodal.video_utils import smart_resize
    from veomni.trainer.dit_trainer import VeOmniDiTArguments

    config_path = Path(__file__).resolve().parents[2] / "configs/dit/hunyuanvideo15_t2v.yaml"
    monkeypatch.setattr(sys, "argv", ["train_dit.py", str(config_path)])
    config = parse_args(VeOmniDiTArguments)
    assert config.data.dataloader.shuffle is False
    raw = torch.rand(1, 3, 720, 1280) * 255
    decoded = smart_resize(raw, **config.data.mm_configs)
    assert decoded.shape == raw.shape
    torch.testing.assert_close(decoded, raw, rtol=0, atol=0)
    condition = config.model.condition_model_cfg
    actual = prepare_video(decoded, condition["height"], condition["width"], 1)
    assert actual.shape == (1, 3, 1, 480, 832)
    expected = prepare_video(raw, condition["height"], condition["width"], 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
