from types import SimpleNamespace

import pytest

from veomni.arguments import ModelArguments
from veomni.trainer.text_dpo_trainer import DPOReferenceModelRuntime, _assert_matching_dpo_parallelism
from veomni.utils.checkpoint_utils import should_skip_hf_weight_load


def test_reference_runtime_never_skips_hf_weight_load_on_policy_resume():
    runtime = DPOReferenceModelRuntime.__new__(DPOReferenceModelRuntime)
    runtime.args = ModelArguments(model_path="./policy")
    runtime.model_name = "reference"
    runtime.train = SimpleNamespace(checkpoint=SimpleNamespace(load_path="/ckpt"))

    assert should_skip_hf_weight_load("/ckpt", {}) is True
    assert runtime.skip_hf_weight_load is False
    assert runtime.model_name == "reference"


def _acc(ulysses_size=1, cp_size=1, dp_size=8):
    return SimpleNamespace(ulysses_size=ulysses_size, cp_size=cp_size, dp_size=dp_size)


def test_matching_dpo_parallelism_is_silent():
    _assert_matching_dpo_parallelism(_acc(2, 1, 4), _acc(2, 1, 4))


def test_mismatched_dpo_sp_raises():
    with pytest.raises(ValueError, match="must match the policy"):
        _assert_matching_dpo_parallelism(_acc(2, 1, 4), _acc(1, 1, 4))


def test_mismatched_dpo_dp_raises():
    with pytest.raises(ValueError, match="must match the policy"):
        _assert_matching_dpo_parallelism(_acc(1, 1, 8), _acc(1, 1, 4))
