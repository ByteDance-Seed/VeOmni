"""Per-module checkpoint path resolution in ``OmniConfig._resolve_model_path``.

A SeedOmni V2 checkpoint is a root folder with one subfolder per OmniModule, so
a module's ``model_path`` is normally a bare name joined under that root. These
tests pin down when the join must *not* happen.
"""

from __future__ import annotations

from typing import Any

from veomni.models.seed_omni.configuration_omni import OmniConfig


def _resolved(root: str, module_path: str) -> str:
    modules_config: dict[str, Any] = {"vision": {"model": {"model_path": module_path}}}
    return OmniConfig._resolve_model_path(root, modules_config)["vision"]["model"]["model_path"]


def test_relative_module_path_joins_under_root() -> None:
    assert _resolved("/local/root", "vision_encoder") == "/local/root/vision_encoder"


def test_absolute_module_path_is_left_alone() -> None:
    assert _resolved("/local/root", "/elsewhere/vision_encoder") == "/elsewhere/vision_encoder"


def test_remote_module_path_is_left_alone() -> None:
    """``os.path.isabs`` is False for a remote scheme, so without an explicit
    check this would become ``/local/root/hdfs://ns/vision_encoder``."""
    assert _resolved("/local/root", "hdfs://ns/vision_encoder") == "hdfs://ns/vision_encoder"


def test_config_and_tokenizer_paths_track_the_resolved_model_path() -> None:
    modules_config: dict[str, Any] = {"vision": {"model": {"model_path": "vision_encoder"}}}
    model = OmniConfig._resolve_model_path("/local/root", modules_config)["vision"]["model"]

    assert model["config_path"] == "/local/root/vision_encoder"
    assert model["tokenizer_path"] == "/local/root/vision_encoder"


def test_empty_modules_config_returns_empty_dict() -> None:
    assert OmniConfig._resolve_model_path("/local/root", None) == {}
    assert OmniConfig._resolve_model_path("/local/root", {}) == {}
