"""Asymmetric multimodal forward tests under FSDP2.

Validates that forward + backward complete without NCCL hangs when some ranks
have multimodal data (images/audio/video) while other ranks have text-only data.
The model's dummy_forward() must fire on text-only ranks so that all ranks
participate in FSDP collectives.

Requires: 2+ GPUs.
"""

import gc
import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from veomni.arguments import MixedPrecisionConfig
from veomni.utils.device import empty_cache
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


pytestmark = pytest.mark.unit

_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.2.0")
_v4_only = pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.2.0")
_v5_only = pytest.mark.skipif(not _is_transformers_v5, reason="Requires transformers >= 5.2.0")

_TEXT_SEQ_LEN = 64
_VOCAB_SIZE = 1024


# ---------------------------------------------------------------------------
# Batch construction helpers
# ---------------------------------------------------------------------------


def _vlm_batch(*, rank, device, dtype, patch_size):
    """Build VLM batch: rank 0 gets images + video, other ranks get text-only."""
    h, w = 4, 4
    image_t, video_t = 2, 10
    merge_size, temporal_patch_size = 2, 2

    image_seqlen = h * w // (merge_size**2) * image_t
    video_seqlen = h * w // (merge_size**2) * video_t

    if rank == 0:
        seq_len = _TEXT_SEQ_LEN + image_seqlen + video_seqlen
        pixel_dim = patch_size**2 * temporal_patch_size * 3

        mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        image_mask = mask.clone()
        image_mask[0, :image_seqlen] = True
        video_mask = mask.clone()
        video_mask[0, -video_seqlen:] = True

        return {
            "input_ids": torch.randint(0, _VOCAB_SIZE, (1, seq_len), device=device),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long, device=device),
            "labels": torch.randint(0, _VOCAB_SIZE, (1, seq_len), device=device),
            "pixel_values": torch.rand(image_t * h * w, pixel_dim, dtype=dtype, device=device),
            "pixel_values_videos": torch.rand(video_t * h * w, pixel_dim, dtype=dtype, device=device),
            "image_mask": image_mask,
            "video_mask": video_mask,
            "image_grid_thw": torch.tensor([[1, h, w]] * image_t, dtype=torch.long, device=device),
            "video_grid_thw": torch.tensor([[video_t, h, w]], dtype=torch.long, device=device),
        }
    else:
        return {
            "input_ids": torch.randint(0, _VOCAB_SIZE, (1, _TEXT_SEQ_LEN), device=device),
            "attention_mask": torch.ones(1, _TEXT_SEQ_LEN, dtype=torch.long, device=device),
            "labels": torch.randint(0, _VOCAB_SIZE, (1, _TEXT_SEQ_LEN), device=device),
            "image_mask": torch.zeros(1, _TEXT_SEQ_LEN, dtype=torch.bool, device=device),
            "video_mask": torch.zeros(1, _TEXT_SEQ_LEN, dtype=torch.bool, device=device),
        }


def _omni_batch(*, rank, device, dtype, patch_size, is_qwen3_omni=False):
    """Build omni batch: rank 0 gets images + audio + video, others get text-only."""
    h, w = 4, 4
    image_t, video_t = 2, 10
    merge_size, temporal_patch_size = 2, 2
    audio_token_num, audio_num = 100, 2

    image_seqlen = h * w // (merge_size**2) * image_t
    video_seqlen = h * w // (merge_size**2) * video_t

    if is_qwen3_omni:
        raw = audio_num * audio_token_num * 4
        leave = raw % 100
        feat = (leave - 1) // 2 + 1
        audio_seqlen = ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (raw // 100) * 13
    else:
        audio_seqlen = audio_num * audio_token_num

    if rank == 0:
        seq_len = _TEXT_SEQ_LEN + image_seqlen + audio_seqlen + video_seqlen
        pixel_dim = patch_size**2 * temporal_patch_size * 3

        mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        start = _TEXT_SEQ_LEN
        image_mask = mask.clone()
        image_mask[0, start : start + image_seqlen] = True
        start += image_seqlen
        audio_mask = mask.clone()
        audio_mask[0, start : start + audio_seqlen] = True
        start += audio_seqlen
        video_mask = mask.clone()
        video_mask[0, start : start + video_seqlen] = True

        return {
            "input_ids": torch.randint(0, _VOCAB_SIZE, (1, seq_len), device=device),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long, device=device),
            "labels": torch.randint(0, _VOCAB_SIZE, (1, seq_len), device=device),
            "pixel_values": torch.rand(image_t * h * w, pixel_dim, dtype=dtype, device=device),
            "pixel_values_videos": torch.rand(video_t * h * w, pixel_dim, dtype=dtype, device=device),
            "input_features": torch.rand(4 * audio_token_num * audio_num, 128, dtype=dtype, device=device),
            "image_mask": image_mask,
            "video_mask": video_mask,
            "audio_mask": audio_mask,
            "image_grid_thw": torch.tensor([[1, h, w]] * image_t, dtype=torch.long, device=device),
            "video_grid_thw": torch.tensor([[video_t, h, w]], dtype=torch.long, device=device),
            "audio_feature_lengths": torch.tensor([4 * audio_token_num] * audio_num, dtype=torch.long, device=device),
        }
    else:
        return {
            "input_ids": torch.randint(0, _VOCAB_SIZE, (1, _TEXT_SEQ_LEN), device=device),
            "attention_mask": torch.ones(1, _TEXT_SEQ_LEN, dtype=torch.long, device=device),
            "labels": torch.randint(0, _VOCAB_SIZE, (1, _TEXT_SEQ_LEN), device=device),
            "image_mask": torch.zeros(1, _TEXT_SEQ_LEN, dtype=torch.bool, device=device),
            "video_mask": torch.zeros(1, _TEXT_SEQ_LEN, dtype=torch.bool, device=device),
            "audio_mask": torch.zeros(1, _TEXT_SEQ_LEN, dtype=torch.bool, device=device),
        }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _asymmetric_forward_worker(model_type, config_path, batch_fn):
    """Rank 0 gets multimodal data, other ranks get text-only. Verifies no NCCL hang."""
    from veomni import _apply_patches
    from veomni.distributed.parallel_state import init_parallel_state
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import get_device_type

    _apply_patches()

    # Tight NCCL timeout so a missing dummy_forward fails fast instead of hanging
    os.environ["NCCL_TIMEOUT"] = "120"

    world_size = dist.get_world_size()
    init_parallel_state(dp_size=world_size, dp_shard_size=world_size, dp_mode="fsdp2")

    rank = dist.get_rank()
    device = torch.device(f"{get_device_type()}:{rank}")

    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        moe_implementation="eager",
        init_device="meta",
    )
    model.train()

    model = build_parallelize_model(
        model,
        weights_path=None,
        init_device="meta",
        mixed_precision=MixedPrecisionConfig(enable=True),
        enable_gradient_checkpointing=False,
        basic_modules=[],
    )

    batch = batch_fn(rank=rank, device=device, dtype=torch.bfloat16)

    output = model(**batch)
    loss = output.loss
    assert loss is not None, f"[Rank {rank}] Loss is None for {model_type}"
    assert torch.isfinite(loss), f"[Rank {rank}] Loss is not finite: {loss.item()}"

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"[Rank {rank}] Non-finite gradient in {name}"

    dist.barrier()

    del model
    gc.collect()
    empty_cache()

    if rank == 0:
        print(f"Asymmetric forward test passed for {model_type}")


# ---------------------------------------------------------------------------
# VLM test cases
# ---------------------------------------------------------------------------

_vlm_cases = [
    pytest.param(
        "qwen3_vl",
        "./tests/toy_config/qwen3vl_toy",
        partial(_vlm_batch, patch_size=16),
        id="qwen3_vl",
        marks=_v4_only,
    ),
    pytest.param(
        "qwen2_vl",
        "./tests/toy_config/qwen2vl_toy",
        partial(_vlm_batch, patch_size=14),
        id="qwen2_vl",
        marks=_v4_only,
    ),
    pytest.param(
        "qwen2_5_vl",
        "./tests/toy_config/qwen25vl_toy",
        partial(_vlm_batch, patch_size=14),
        id="qwen2_5_vl",
        marks=_v4_only,
    ),
    pytest.param(
        "qwen2_5_vl",
        "./tests/toy_config/qwen25vl_toy",
        partial(_vlm_batch, patch_size=14),
        id="qwen2_5_vl_v5",
        marks=_v5_only,
    ),
    pytest.param(
        "qwen3_vl",
        "./tests/toy_config/qwen3vl_toy",
        partial(_vlm_batch, patch_size=16),
        id="qwen3_vl_v5",
        marks=_v5_only,
    ),
    pytest.param(
        "qwen3_vl_moe",
        "./tests/toy_config/qwen3vlmoe_toy",
        partial(_vlm_batch, patch_size=16),
        id="qwen3_vl_moe_v5",
        marks=_v5_only,
    ),
]


@pytest.mark.parametrize("model_type, config_path, batch_fn", _vlm_cases)
def test_asymmetric_forward_vlm(model_type: str, config_path: str, batch_fn):
    """Verify no NCCL hang when some ranks lack image/video data under FSDP2."""
    from ..tools.launch_utils import torchrun

    torchrun(
        partial(_asymmetric_forward_worker, model_type, config_path, batch_fn),
        world_size=2,
    )


# ---------------------------------------------------------------------------
# Omni test cases (vision + audio encoders)
# ---------------------------------------------------------------------------

_omni_cases = [
    pytest.param(
        "qwen2_5_omni",
        "./tests/toy_config/qwen25omni_toy",
        partial(_omni_batch, patch_size=14, is_qwen3_omni=False),
        id="qwen2_5_omni",
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3_omni_moe",
        "./tests/toy_config/qwen3omni_toy",
        partial(_omni_batch, patch_size=16, is_qwen3_omni=True),
        id="qwen3_omni_moe",
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3_omni_moe",
        "./tests/toy_config/qwen3omni_toy",
        partial(_omni_batch, patch_size=16, is_qwen3_omni=True),
        id="qwen3_omni_moe_v5",
        marks=_v5_only,
    ),
]


@pytest.mark.parametrize("model_type, config_path, batch_fn", _omni_cases)
def test_asymmetric_forward_omni(model_type: str, config_path: str, batch_fn):
    """Verify no NCCL hang when some ranks lack image/audio/video data under FSDP2."""
    from ..tools.launch_utils import torchrun

    torchrun(
        partial(_asymmetric_forward_worker, model_type, config_path, batch_fn),
        world_size=2,
    )
