# Copyright 2025 Bytedance Ltd. and/or its affiliates
"""Guards for skip_weights_load (DCP resume skips HF materialization)."""

from veomni.utils.checkpoint_utils import should_skip_hf_weight_load


def test_skip_hf_weight_load_when_full_dcp_resume():
    assert should_skip_hf_weight_load("/tmp/ckpt/global_step_200", {}) is True
    assert should_skip_hf_weight_load("/tmp/ckpt/global_step_200", None) is True


def test_keep_hf_weight_load_for_fresh_or_lora():
    assert should_skip_hf_weight_load(None, {}) is False
    assert should_skip_hf_weight_load("/tmp/ckpt/global_step_200", {"r": 8}) is False
