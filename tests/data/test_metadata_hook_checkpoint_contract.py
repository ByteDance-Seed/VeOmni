import copy
import importlib
import pickle

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize(
    ("module_path", "model_name", "helper_name", "identity"),
    [
        (
            "veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu",
            "Qwen3_5ForConditionalGeneration",
            "collate_multimodal_metadata",
            "veomni:qwen3.5-metadata-v1",
        ),
        (
            "veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu",
            "Qwen3_5MoeForConditionalGeneration",
            "collate_multimodal_metadata",
            "veomni:qwen3.5-moe-metadata-v1",
        ),
        (
            "veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu",
            "Qwen4ExpForConditionalGeneration",
            "qwen4_exp_collate_metadata",
            "veomni:qwen4-exp-metadata-v1",
        ),
    ],
)
def test_metadata_hook_contract_preserves_outputs_and_rng(module_path, model_name, helper_name, identity):
    module = importlib.import_module(module_path)
    hook = pickle.loads(pickle.dumps(getattr(module, model_name).get_metadata_collate_func(None)))
    assert hook.supports_rng_free_collation is True
    assert hook.checkpoint_collator_identity == identity
    sample = {"image_grid_thw": torch.tensor([[1, 4, 4], [2, 2, 2]])}
    expected = copy.deepcopy(sample)
    rng = torch.get_rng_state().clone()
    hook(sample, {"pixel_values": 3})
    assert torch.equal(torch.get_rng_state(), rng)
    getattr(module, helper_name)(expected, {"pixel_values": 3})

    def assert_equal(actual, reference):
        if torch.is_tensor(actual):
            assert torch.equal(actual, reference)
        elif isinstance(actual, dict):
            assert actual.keys() == reference.keys()
            for key in actual:
                assert_equal(actual[key], reference[key])
        else:
            assert actual == reference

    assert_equal(sample, expected)
