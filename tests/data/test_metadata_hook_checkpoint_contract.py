import copy
import importlib
import pickle
import random

import numpy as np
import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize(
    ("module_path", "model_name", "helper_name", "identity"),
    [
        (
            "veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_{backend}",
            "Qwen3_5ForConditionalGeneration",
            "collate_multimodal_metadata",
            "veomni:qwen3.5-metadata-v1",
        ),
        (
            "veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_{backend}",
            "Qwen3_5MoeForConditionalGeneration",
            "collate_multimodal_metadata",
            "veomni:qwen3.5-moe-metadata-v1",
        ),
        (
            "veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_{backend}",
            "Qwen4ExpForConditionalGeneration",
            "qwen4_exp_collate_metadata",
            "veomni:qwen4-exp-metadata-v1",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["gpu", "npu"])
def test_metadata_hook_contract_preserves_outputs_and_rng(module_path, model_name, helper_name, identity, backend):
    module = importlib.import_module(module_path.format(backend=backend))
    hook = pickle.loads(pickle.dumps(getattr(module, model_name).get_metadata_collate_func(None)))
    assert hook.supports_rng_free_collation is True
    assert hook.checkpoint_collator_identity == identity
    sample = {"image_grid_thw": torch.tensor([[1, 4, 4], [2, 2, 2]])}
    expected = copy.deepcopy(sample)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    hook(sample, {"pixel_values": 3})
    assert random.getstate() == python_rng
    current_numpy_rng = np.random.get_state()
    assert current_numpy_rng[0] == numpy_rng[0]
    assert np.array_equal(current_numpy_rng[1], numpy_rng[1])
    assert current_numpy_rng[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
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
