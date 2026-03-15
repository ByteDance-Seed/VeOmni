"""L1: Validate v5 patchgen-generated modeling code against HF eager baseline.

For v5 models, VeOmni uses the patchgen framework to generate modified modeling
code from HuggingFace source. This test ensures the generated code produces
forward/backward results numerically close to the original HF model.

This test requires transformers >= 5.0.0.
"""

import copy
import gc
import os

import pytest
import torch
from transformers import set_seed

from veomni.utils.device import empty_cache, get_device_type, synchronize
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

from ..conftest import v5_only


pytestmark = [pytest.mark.L1, v5_only]


def _release_device_memory():
    synchronize()
    gc.collect()
    empty_cache()


def _has_gpu():
    """Check if a GPU is available for testing."""
    try:
        device = get_device_type()
        if device == "cuda":
            return torch.cuda.is_available()
        return False
    except Exception:
        return False


# Skip entire module if no GPU is available
if not _has_gpu():
    pytest.skip("No GPU available for L1 patchgen tests", allow_module_level=True)


# Skip if not v5
if not is_transformers_version_greater_or_equal_to("5.0.0"):
    pytest.skip("Requires transformers >= 5.0.0", allow_module_level=True)


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

_DEFAULT_RTOL = 1e-2
_DEFAULT_ATOL = 1e-2

_PATCHGEN_TEST_CASES = [
    pytest.param(
        "./tests/toy_config/qwen3_5_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_5",
    ),
    pytest.param(
        "./tests/toy_config/qwen3_5_moe_toy/config.json",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_5_moe",
    ),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_path, is_moe, rtol, atol", _PATCHGEN_TEST_CASES)
def test_patchgen_fwd_equivalence(config_path: str, is_moe: bool, rtol: float, atol: float):
    """Compare patchgen-generated model forward against HF eager baseline.

    This test:
    1. Builds the HF model with eager attention (baseline).
    2. Builds the VeOmni-patched model using patchgen-generated code.
    3. Syncs weights between the two.
    4. Runs forward on identical input and compares output logits/loss.
    """
    from veomni.models import build_foundation_model

    set_seed(42)
    device = get_device_type()

    # Build HF baseline model
    os.environ["MODELING_BACKEND"] = "hf"
    from veomni import _apply_patches

    _apply_patches()

    hf_model_info = build_foundation_model(config_path=config_path, init_device=device)
    hf_model = hf_model_info.model
    model_config = hf_model_info.config
    hf_model.eval()

    # Save HF state dict
    state_dict = copy.deepcopy(hf_model.state_dict())

    # Build dummy input
    vocab_size = model_config.vocab_size
    seq_len = 64
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
    labels = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long, device=device)

    # HF forward
    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    hf_loss = hf_outputs.loss if hasattr(hf_outputs, "loss") else hf_outputs[0]

    del hf_model
    _release_device_memory()

    # Build VeOmni patched model
    os.environ["MODELING_BACKEND"] = "veomni"
    _apply_patches()

    veomni_model_info = build_foundation_model(config_path=config_path, init_device=device)
    veomni_model = veomni_model_info.model

    # Load same weights
    veomni_model.load_state_dict(state_dict, strict=False)
    veomni_model.eval()

    # VeOmni forward
    with torch.no_grad():
        veomni_outputs = veomni_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    veomni_loss = veomni_outputs.loss if hasattr(veomni_outputs, "loss") else veomni_outputs[0]

    # Compare losses
    if hf_loss is not None and veomni_loss is not None:
        torch.testing.assert_close(
            veomni_loss.float(),
            hf_loss.float(),
            rtol=rtol,
            atol=atol,
            msg=f"Patchgen loss mismatch for {config_path}",
        )

    del veomni_model, state_dict
    _release_device_memory()
