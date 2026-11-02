# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

import pytest
import yaml

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.arguments_types import CheckpointConfig


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_checkpoint_config_preserves_legacy_positional_arguments():
    config = CheckpointConfig("checkpoint-dir", "dcp", False)
    assert config.output_dir == "checkpoint-dir"
    assert config.manager == "dcp"
    assert config.save_async is False
    assert config.extra_parallel_canonical_layout is False


def test_checkpoint_config_accepts_explicit_extra_parallel_layout():
    config = CheckpointConfig("checkpoint-dir", extra_parallel_canonical_layout=True)
    assert config.output_dir == "checkpoint-dir"
    assert config.extra_parallel_canonical_layout is True


def test_checkpoint_config_parser_preserves_explicit_layout():
    from veomni.arguments.parser import _instantiate_recursive

    config = _instantiate_recursive(
        CheckpointConfig, {"output_dir": "checkpoint-dir", "extra_parallel_canonical_layout": True}
    )
    assert config.output_dir == "checkpoint-dir"
    assert config.extra_parallel_canonical_layout is True


@pytest.mark.parametrize("value", ["'false'", "'true'", "0", "1", "null"])
def test_checkpoint_config_parser_rejects_non_boolean_layout(value):
    from veomni.arguments.parser import _instantiate_recursive

    config = yaml.safe_load(f"extra_parallel_canonical_layout: {value}")
    with pytest.raises(ValueError, match="extra_parallel_canonical_layout must be a boolean"):
        _instantiate_recursive(CheckpointConfig, config)


@pytest.mark.parametrize("value", [False, True])
def test_checkpoint_config_cli_converts_layout_boolean(monkeypatch, value):
    from veomni.arguments.parser import parse_args

    monkeypatch.setattr("sys.argv", ["test", "--extra_parallel_canonical_layout", str(value).lower()])
    assert parse_args(CheckpointConfig).extra_parallel_canonical_layout is value
