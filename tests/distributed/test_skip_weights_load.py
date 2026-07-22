# Copyright 2025 Bytedance Ltd. and/or its affiliates
"""Guards for should_skip_hf_weight_load (resume skips HF materialization)."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from veomni.distributed import torch_parallelize
from veomni.distributed.torch_parallelize import build_parallelize_model, parallelize_model_fsdp2
from veomni.models.module_utils import init_empty_weights
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


def test_dcp_resume_preserves_nonpersistent_buffers_and_forward(monkeypatch, tmp_path):
    class ModelWithDerivedBuffer(nn.Module):
        _no_split_modules = []

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            self.register_buffer("scale", torch.tensor([0.25, 2.0]), persistent=False)

        def forward(self, x):
            return (x @ self.weight) * self.scale

        def init_weights(self):
            raise AssertionError("DCP resume must not initialize model parameters")

    original = ModelWithDerivedBuffer()
    assert "scale" not in original.state_dict()
    inputs = torch.tensor([[2.0, -1.0]])
    expected_output = original(inputs)
    checkpoint_dir = tmp_path / "dcp"
    dcp.save({"model": original}, checkpoint_id=checkpoint_dir)

    with init_empty_weights():
        resumed = ModelWithDerivedBuffer()
    assert resumed.weight.is_meta
    assert not resumed.scale.is_meta
    parallel_state = SimpleNamespace(any_extra_parallel_enabled=False, extra_parallel_names=[], fsdp_mesh=None)
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(torch_parallelize, "fully_shard", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_parallelize, "get_device_type", lambda: "cpu")

    resumed = parallelize_model_fsdp2(
        resumed,
        weights_path="unused-hf-path",
        mixed_precision=SimpleNamespace(enable=False),
        should_skip_hf_weight_load=True,
        init_device="meta",
    )
    dcp.load({"model": resumed}, checkpoint_id=checkpoint_dir)

    torch.testing.assert_close(resumed.scale, original.scale, rtol=0, atol=0)
    torch.testing.assert_close(resumed(inputs), expected_output, rtol=0, atol=0)


@pytest.mark.parametrize("parallelize", [build_parallelize_model, parallelize_model_fsdp2])
def test_parallelize_apis_reject_renamed_skip_weights_load(parallelize):
    with pytest.raises(TypeError, match="'skip_weights_load' was renamed to 'should_skip_hf_weight_load'"):
        parallelize(MagicMock(), skip_weights_load=True)
