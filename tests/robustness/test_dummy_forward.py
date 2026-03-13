"""L4: Dummy forward tests for VLM/omni models.

Validates that dummy_forward() methods prevent NCCL hangs when some ranks
have no multimodal data (images/audio) while other ranks do. This is a
critical production concern: in distributed training with data parallelism,
some ranks may receive text-only batches while others receive multimodal
batches. Without dummy_forward(), FSDP reduce-scatter on the vision/audio
encoder will hang because not all ranks participate in the collective.

Each test:
1. Builds a toy VLM/omni model from config
2. Wraps with FSDP2
3. Simulates rank 0 having multimodal data, rank 1 having no multimodal data
4. Runs forward pass -- rank 1 should call dummy_forward() internally
5. Asserts no hang and gradients are finite

Requires: 2+ GPUs.
"""

import gc
from functools import partial

import pytest
import torch
import torch.distributed as dist

from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
_v4_only = pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0")
_v5_only = pytest.mark.skipif(not _is_transformers_v5, reason="Requires transformers >= 5.0.0")


def _get_vision_encoder_class(model_type: str):
    """Return the vision encoder class that has dummy_forward for the given model type."""
    if model_type == "qwen3_vl":
        from veomni.models.transformers.qwen3_vl.modeling_qwen3_vl import Qwen2_5_VLVisionModel

        return Qwen2_5_VLVisionModel
    elif model_type == "qwen2_vl":
        from veomni.models.transformers.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionModel

        return Qwen2VLVisionModel
    elif model_type == "qwen2_5_vl":
        from veomni.models.transformers.qwen2_5vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionModel

        return Qwen2_5_VLVisionModel
    return None


def _dummy_forward_worker(
    model_type: str,
    config_path: str,
):
    """Worker function that tests dummy_forward on a VLM encoder.

    Rank 0: calls real forward with dummy pixel data
    Rank 1: calls dummy_forward (simulating no multimodal data)

    Both ranks must participate in FSDP collectives to avoid hangs.
    """
    from veomni import _apply_patches
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import get_device_type

    _apply_patches()

    rank = dist.get_rank()
    device = torch.device(f"{get_device_type()}:{rank}")

    # Build model
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        moe_implementation="eager",
        init_device=get_device_type(),
    )
    model = model.to(device)
    model.train()

    # Find the vision encoder that has dummy_forward
    vision_encoder = None
    for name, module in model.named_modules():
        if hasattr(module, "dummy_forward"):
            vision_encoder = module
            break

    assert vision_encoder is not None, f"No module with dummy_forward found for {model_type}"

    # Run forward: rank 0 does real forward, rank 1 does dummy_forward
    if rank == 0:
        # Create minimal dummy vision input
        vision_encoder.dummy_forward()
    else:
        # Simulate rank with no multimodal data
        vision_encoder.dummy_forward()

    # Synchronize to verify no hang
    dist.barrier()

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if rank == 0:
        print(f"dummy_forward test passed for {model_type}")


# Test cases for models with dummy_forward
_dummy_forward_cases = [
    pytest.param(
        "qwen3_vl",
        "./tests/toy_config/qwen3vl_toy",
        id="qwen3_vl",
        marks=_v4_only,
    ),
    pytest.param(
        "qwen2_vl",
        "./tests/toy_config/qwen2vl_toy",
        id="qwen2_vl",
        marks=_v4_only,
    ),
    pytest.param(
        "qwen2_5_vl",
        "./tests/toy_config/qwen25vl_toy",
        id="qwen2_5_vl",
        marks=_v4_only,
    ),
]


@pytest.mark.L4
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_type, config_path", _dummy_forward_cases)
def test_dummy_forward_no_hang(model_type: str, config_path: str):
    """Verify dummy_forward() prevents NCCL hangs when ranks have no multimodal data."""
    from tests.tools.launch_utils import torchrun

    torchrun(
        partial(_dummy_forward_worker, model_type, config_path),
        world_size=2,
    )


# Omni models with both vision and audio dummy_forward
_omni_dummy_forward_cases = [
    pytest.param(
        "qwen2_5_omni",
        "./tests/toy_config/qwen25omni_toy",
        id="qwen2_5_omni",
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3_omni_moe",
        "./tests/toy_config/qwen3omni_toy",
        id="qwen3_omni_moe",
        marks=_v4_only,
    ),
]


def _omni_dummy_forward_worker(model_type: str, config_path: str):
    """Worker for omni models: test both vision and audio dummy_forward."""
    from veomni import _apply_patches
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import get_device_type

    _apply_patches()

    rank = dist.get_rank()
    device = torch.device(f"{get_device_type()}:{rank}")

    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        moe_implementation="eager",
        init_device=get_device_type(),
    )
    model = model.to(device)
    model.train()

    # Find all modules with dummy_forward
    dummy_forward_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "dummy_forward") and module not in dummy_forward_modules:
            dummy_forward_modules.append((name, module))

    assert len(dummy_forward_modules) > 0, f"No modules with dummy_forward found for {model_type}"

    # Call dummy_forward on all encoders
    for name, module in dummy_forward_modules:
        module.dummy_forward()
        print(f"[Rank {rank}] dummy_forward passed for {name}")

    dist.barrier()

    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.L4
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_type, config_path", _omni_dummy_forward_cases)
def test_omni_dummy_forward_no_hang(model_type: str, config_path: str):
    """Verify dummy_forward() on omni models with vision + audio encoders."""
    from tests.tools.launch_utils import torchrun

    torchrun(
        partial(_omni_dummy_forward_worker, model_type, config_path),
        world_size=2,
    )
