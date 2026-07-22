# Copyright 2025 Bytedance Ltd. and/or its affiliates
"""Guards for should_skip_hf_weight_load (resume skips HF materialization)."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from veomni.distributed import torch_parallelize
from veomni.distributed.torch_parallelize import build_parallelize_model, parallelize_model_fsdp2
from veomni.utils.checkpoint_utils import should_skip_hf_weight_load


def test_skip_hf_weight_load_when_full_non_lora_resume():
    assert should_skip_hf_weight_load("/tmp/ckpt/global_step_200", {}) is True
    assert should_skip_hf_weight_load("/tmp/ckpt/global_step_200", None) is True


def test_keep_hf_weight_load_for_fresh_or_lora():
    assert should_skip_hf_weight_load(None, {}) is False
    assert should_skip_hf_weight_load("/tmp/ckpt/global_step_200", {"r": 8}) is False


def test_parallelize_apis_expose_should_skip_hf_weight_load():
    assert "should_skip_hf_weight_load" in inspect.signature(build_parallelize_model).parameters
    assert "should_skip_hf_weight_load" in inspect.signature(parallelize_model_fsdp2).parameters


def test_build_parallelize_model_forwards_should_skip_hf_weight_load(monkeypatch):
    model = MagicMock()
    parallelized_model = MagicMock()
    parallelize_fsdp2 = MagicMock(return_value=parallelized_model)
    parallel_state = SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="fsdp2")
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", parallelize_fsdp2)

    result = build_parallelize_model(
        model,
        mixed_precision=SimpleNamespace(enable=False),
        enable_gradient_checkpointing=False,
        should_skip_hf_weight_load=True,
    )

    assert result is parallelized_model
    assert parallelize_fsdp2.call_args.kwargs["should_skip_hf_weight_load"] is True


@pytest.mark.parametrize("parallelize", [build_parallelize_model, parallelize_model_fsdp2])
def test_parallelize_apis_reject_renamed_skip_weights_load(parallelize):
    with pytest.raises(TypeError, match="'skip_weights_load' was renamed to 'should_skip_hf_weight_load'"):
        parallelize(MagicMock(), skip_weights_load=True)
