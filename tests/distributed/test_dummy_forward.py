"""Asymmetric multimodal forward tests for VLM/omni models under FSDP2.

Validates that forward + backward complete without NCCL hangs when some ranks
have multimodal data (images/audio/video) while other ranks have text-only data.

This is a critical production scenario: in distributed training with data
parallelism, some ranks may receive text-only batches while others receive
multimodal batches. The model's internal dummy_forward() must fire on
text-only ranks so that all ranks participate in FSDP reduce-scatter
collectives and no rank is left waiting.

Each test:
1. Builds a toy VLM/omni model from config
2. Wraps encoder modules (those with dummy_forward) and root model with FSDP2
3. Rank 0 receives a multimodal batch; rank 1 receives a text-only batch
4. Runs forward + backward
5. Asserts no hang, loss is finite, and gradients are finite

Requires: 2+ GPUs.
"""

import gc
from functools import partial

import pytest
import torch
import torch.distributed as dist

from veomni.utils.device import empty_cache
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
_v4_only = pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0")

_TEXT_SEQ_LEN = 64
_VOCAB_SIZE = 1024


# ---------------------------------------------------------------------------
# Batch construction helpers
# ---------------------------------------------------------------------------


def _vlm_batch(*, rank, device, dtype, patch_size):
    """Build VLM batch: rank 0 gets images + video, other ranks get text-only.

    All batches include image_mask / video_mask (all-False for text-only) so
    that models relying on pre-computed masks (e.g. Qwen3-VL) don't need to
    fall back to computing them from input_ids.
    """
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
    """Build omni batch: rank 0 gets images + audio + video, others get text-only.

    Always includes image_mask / video_mask / audio_mask since Qwen3-Omni-MoE
    asserts their presence in kwargs unconditionally.
    """
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
    """Test forward + backward with asymmetric multimodal data under FSDP2.

    Rank 0 receives a multimodal batch (images, audio, video), while rank 1
    receives a text-only batch.  The model's internal logic should call
    dummy_forward() on rank 1, keeping FSDP collectives in sync.
    """
    from torch.distributed._composable.fsdp import fully_shard

    from veomni import _apply_patches
    from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import get_device_type

    _apply_patches()

    world_size = dist.get_world_size()
    init_parallel_state(dp_size=world_size, dp_shard_size=world_size, dp_mode="fsdp2")

    rank = dist.get_rank()
    device = torch.device(f"{get_device_type()}:{rank}")

    # Build model on real device
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        moe_implementation="eager",
        init_device=get_device_type(),
    )
    model = model.to(device=device, dtype=torch.float32)
    model.train()

    # Wrap encoder modules (those with dummy_forward) as separate FSDP units
    # so their reduce-scatter happens independently -- this is the scenario
    # where a missing dummy_forward would cause a hang.
    fsdp_mesh = get_parallel_state().fsdp_mesh
    seen_ids = set()
    for _name, module in model.named_modules():
        if hasattr(module, "dummy_forward") and id(module) not in seen_ids:
            fully_shard(module, mesh=fsdp_mesh)
            seen_ids.add(id(module))

    # Wrap root model
    fully_shard(model, mesh=fsdp_mesh)

    # Construct rank-specific batch
    batch = batch_fn(rank=rank, device=device, dtype=torch.float32)

    # Forward
    output = model(**batch)
    loss = output.loss
    assert loss is not None, f"[Rank {rank}] Loss is None for {model_type}"
    assert torch.isfinite(loss), f"[Rank {rank}] Loss is not finite: {loss.item()}"

    # Backward -- triggers reduce-scatter on all FSDP units
    loss.backward()

    # Verify gradients are finite (no NaN/Inf from dummy_forward interaction)
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"[Rank {rank}] Non-finite gradient in {name}"

    # Barrier confirms no rank is stuck
    dist.barrier()

    # Clean up
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
]


@pytest.mark.parametrize("model_type, config_path, batch_fn", _omni_cases)
def test_asymmetric_forward_omni(model_type: str, config_path: str, batch_fn):
    """Verify no NCCL hang when some ranks lack image/audio/video data under FSDP2."""
    from ..tools.launch_utils import torchrun

    torchrun(
        partial(_asymmetric_forward_worker, model_type, config_path, batch_fn),
        world_size=2,
    )
