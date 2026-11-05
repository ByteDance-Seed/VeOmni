"""DiT examples must reach the native parser and shared data transform."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import parse_args
from veomni.data import build_data_transform
from veomni.models.auto import build_config
from veomni.trainer.dit_trainer import VeOmniDiTArguments


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("image_input", ["path", "bytes", "url"])
@pytest.mark.parametrize("image_mode", [None, "RGBA"])
def test_dit_online_image_mode_preserves_alpha_only_when_requested(tmp_path, monkeypatch, image_input, image_mode):
    """Default RGB decoding stays stable while RGBA survives loading and resizing."""
    image_path = tmp_path / "transparent.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(image_path)
    image = str(image_path)
    if image_input == "bytes":
        image = image_path.read_bytes()
    elif image_input == "url":
        image = "https://example.com/transparent.png"
        monkeypatch.setattr(
            "veomni.data.multimodal.image_utils.requests.get",
            lambda *args, **kwargs: SimpleNamespace(content=image_path.read_bytes(), raise_for_status=lambda: None),
        )
    kwargs = {} if image_mode is None else {"image_mode": image_mode}
    transform = build_data_transform("dit_online", image_min_pixels=256, **kwargs)

    features = transform({"text": "a seed", "image": image}, source_name="Qwen-Image")
    loaded = features[0]["images"][0]

    assert loaded.size == (16, 16)
    assert loaded.mode == (image_mode or "RGB")
    assert loaded.getpixel((0, 0)) == ((0, 0, 0, 0) if image_mode == "RGBA" else (255, 0, 0))


@pytest.mark.parametrize("name", ["wan2.2_I2V_5B", "wan2.2_I2V_14B"])
def test_wan_i2v_yaml_uses_declared_arguments(name, monkeypatch):
    monkeypatch.chdir(ROOT)
    # Kernel availability is independent of the example's dataclass contract.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dit.py",
            str(ROOT / "configs" / "dit" / f"{name}.yaml"),
            "--model.ops_implementation.load_balancing_loss_implementation",
            "eager",
        ],
    )
    args = parse_args(VeOmniDiTArguments)
    assert Path(args.model.config_path).is_file()
    config = build_config(args.model.config_path)
    assert config.model_type == "WanTransformer3DModel"
    assert args.model.accelerator.fsdp_config.fsdp_mode == "fsdp2"
    assert args.model.accelerator.fsdp_config.mixed_precision.param_dtype == "bfloat16"
    assert not args.model.accelerator.offload_config.enable_async_activation
    assert not args.model.accelerator.offload_config.enable_activation
    assert args.model.accelerator.gradient_checkpointing.enable
    assert args.train.training_task == "online_training"
