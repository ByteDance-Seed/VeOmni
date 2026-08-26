from types import SimpleNamespace

from veomni.arguments import ModelRuntimeArguments
from veomni.trainer.text_dpo_trainer import DPOReferenceModelRuntime
from veomni.utils.checkpoint_utils import should_skip_hf_weight_load


def test_reference_runtime_never_skips_hf_weight_load_on_policy_resume():
    runtime = DPOReferenceModelRuntime.__new__(DPOReferenceModelRuntime)
    runtime.args = ModelRuntimeArguments(model_path="./policy")
    runtime.model_name = "reference"
    runtime.train = SimpleNamespace(checkpoint=SimpleNamespace(load_path="/ckpt"))

    assert should_skip_hf_weight_load("/ckpt", {}) is True
    assert runtime.skip_hf_weight_load is False
    assert runtime.model_name == "reference"
