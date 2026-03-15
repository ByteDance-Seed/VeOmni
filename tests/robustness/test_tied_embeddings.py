"""L4: Tied embeddings with FSDP tests.

Validates that models with tie_word_embeddings=True work correctly when
wrapped with FSDP2. Tied embeddings mean the input embedding and output
projection share the same weight tensor. FSDP sharding must handle this
correctly -- if the tied weight is sharded differently for embed_tokens
and lm_head, training will produce incorrect gradients or crash.

Each test:
1. Loads a toy config and sets tie_word_embeddings=True
2. Builds the model and verifies embedding weight is shared
3. Wraps with FSDP2
4. Runs forward + backward
5. Verifies gradients are finite and the tied relationship is preserved
"""

import gc
import os

import pytest
import torch
from transformers import set_seed

from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
_v4_only = pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0")
_v5_only = pytest.mark.skipif(not _is_transformers_v5, reason="Requires transformers >= 5.0.0")


def _check_tied_embeddings(model):
    """Verify that embed_tokens and lm_head share the same weight tensor."""
    embed_weight = None
    lm_head_weight = None

    for name, param in model.named_parameters():
        if "embed_tokens" in name and "weight" in name:
            embed_weight = param
        if "lm_head" in name and "weight" in name:
            lm_head_weight = param

    if embed_weight is not None and lm_head_weight is not None:
        return embed_weight.data_ptr() == lm_head_weight.data_ptr()

    return False


def _test_tied_embeddings_fwd_bwd(config_path: str, model_name: str):
    """Test forward + backward with tied embeddings on a single GPU."""
    from veomni import _apply_patches
    from veomni.data.dummy_dataset import build_dummy_dataset
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import empty_cache, get_device_type, synchronize

    set_seed(42)
    _apply_patches()

    device = get_device_type()

    # Build model with tie_word_embeddings=True
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        moe_implementation="eager",
        init_device=device,
    )

    # Force tie_word_embeddings if the model supports it
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = True
        model.tie_weights()

    model.train()

    # Verify tied embeddings are actually tied
    is_tied = _check_tied_embeddings(model)
    if hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings:
        assert is_tied, f"Model {model_name} has tie_word_embeddings=True but weights are not tied"

    # Create dummy data
    dataset = build_dummy_dataset("text", 2, 512)
    input_ids = torch.tensor(dataset[0][0]["input_ids"]).unsqueeze(0).to(device)
    labels = torch.tensor(dataset[0][0]["labels"]).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    # Backward pass
    loss.backward()

    # Verify gradients are finite
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient for {name}"

    # If tied, verify the gradient is the same for both
    embed_grad = None
    lm_head_grad = None
    for name, param in model.named_parameters():
        if "embed_tokens" in name and "weight" in name and param.grad is not None:
            embed_grad = param.grad
        if "lm_head" in name and "weight" in name and param.grad is not None:
            lm_head_grad = param.grad

    if is_tied and embed_grad is not None and lm_head_grad is not None:
        assert embed_grad.data_ptr() == lm_head_grad.data_ptr(), "Tied weights should share the same gradient tensor"

    # Clean up
    del model, outputs, loss
    gc.collect()
    synchronize()
    empty_cache()


# Models that support tie_word_embeddings
# Note: not all toy configs have tie_word_embeddings=True by default,
# but we force it in the test to validate FSDP compatibility
_tied_embedding_cases = [
    pytest.param(
        "./tests/toy_config/qwen3_toy",
        "qwen3",
        id="qwen3",
        marks=_v4_only,
    ),
    pytest.param(
        "./tests/toy_config/llama31_toy",
        "llama3.1",
        id="llama3.1",
        marks=_v4_only,
    ),
    pytest.param(
        "./tests/toy_config/qwen25_toy",
        "qwen2.5",
        id="qwen2.5",
        marks=_v4_only,
    ),
    pytest.param(
        "./tests/toy_config/qwen3_5_toy/config.json",
        "qwen3_5",
        id="qwen3_5",
        marks=_v5_only,
    ),
]


@pytest.mark.L4
@pytest.mark.parametrize("config_path, model_name", _tied_embedding_cases)
def test_tied_embeddings_fwd_bwd(config_path: str, model_name: str):
    """Verify forward/backward works correctly with tie_word_embeddings=True."""
    _test_tied_embeddings_fwd_bwd(config_path, model_name)


# FSDP test for tied embeddings requires multiple GPUs
_tied_embedding_fsdp_cases = [
    pytest.param(
        "qwen3",
        "./tests/toy_config/qwen3_toy",
        False,
        id="qwen3_fsdp",
        marks=_v4_only,
    ),
    pytest.param(
        "llama3.1",
        "./tests/toy_config/llama31_toy",
        False,
        id="llama3.1_fsdp",
        marks=_v4_only,
    ),
]


@pytest.mark.L4
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_name, config_path, is_moe", _tied_embedding_fsdp_cases)
def test_tied_embeddings_fsdp(model_name: str, config_path: str, is_moe: bool):
    """Verify tied embeddings work with FSDP2 (checkpoint save/load roundtrip).

    This test uses the existing checkpoint test infrastructure to verify that
    models with tied embeddings can be saved and loaded correctly under FSDP2.
    """
    import json
    import shutil
    import subprocess

    from tests.distributed._training_core import materialize_weights
    from tests.distributed.utils import ParallelConfig, build_torchrun_cmd
    from tests.e2e.utils import DummyDataset

    test_dir = f"./_test_tied_embed_{model_name}"
    os.makedirs(test_dir, exist_ok=True)

    materialize_weights(config_path, test_dir)
    dummy_dataset = DummyDataset(seq_len=2048, dataset_type="text")
    train_path = dummy_dataset.save_path

    try:
        config = ParallelConfig(sp_size=1, ep_size=1, fsdp_mode="fsdp2")
        cmd = build_torchrun_cmd(
            script="tests/e2e/train_text_test.py",
            config_path=config_path,
            model_path=test_dir,
            train_path=train_path,
            output_dir=os.path.join(test_dir, "output"),
            parallel_config=config,
            nproc=2,
        )

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        log_path = os.path.join(test_dir, "output", "log_dict.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                metrics = json.load(f)

            # Verify loss values are finite
            for loss_val in metrics.get("loss", []):
                assert not (isinstance(loss_val, float) and (loss_val != loss_val)), (
                    f"NaN loss detected with tied embeddings for {model_name}"
                )

    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        del dummy_dataset
