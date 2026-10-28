"""Asymmetric multimodal forward tests under FSDP2.

Validates that forward + backward complete without NCCL hangs when some ranks
have multimodal data (images/audio/video) while other ranks have text-only data.
The model's dummy_forward() must fire on text-only ranks so that all ranks
participate in FSDP collectives.

Requires: 2+ GPUs.
"""

import gc
import importlib
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import MixedPrecisionConfig
from veomni.utils.device import empty_cache, get_device_type


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
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models import build_foundation_model
    from veomni.utils.device import get_device_type

    from ..tools.training_utils import make_eager_ops_config

    # Tight NCCL timeout so a missing dummy_forward fails fast instead of hanging
    os.environ["NCCL_TIMEOUT"] = "120"

    world_size = dist.get_world_size()
    _init_parallel_state(dp_size=world_size, dp_shard_size=world_size, dp_mode="fsdp2")

    rank = dist.get_rank()
    device = torch.device(f"{get_device_type()}:{rank}")

    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=make_eager_ops_config(),
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
        "qwen2_5_vl",
        "./tests/toy_config/qwen25vl_toy",
        partial(_vlm_batch, patch_size=14),
        id="qwen2_5_vl",
    ),
    pytest.param(
        "qwen3_vl",
        "./tests/toy_config/qwen3vl_toy",
        partial(_vlm_batch, patch_size=16),
        id="qwen3_vl",
    ),
    pytest.param(
        "qwen3_vl_moe",
        "./tests/toy_config/qwen3vlmoe_toy",
        partial(_vlm_batch, patch_size=16),
        id="qwen3_vl_moe",
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
    ),
    pytest.param(
        "qwen3_omni_moe",
        "./tests/toy_config/qwen3omni_toy",
        partial(_omni_batch, patch_size=16, is_qwen3_omni=True),
        id="qwen3_omni_moe",
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("family", ["qwen3_vl", "qwen4_exp"])
@pytest.mark.parametrize("backend", ["gpu", "npu"])
def test_vlm_dummy_vision_input_construction(family, backend):
    """The dummy input is a local patch shard with global vision metadata."""
    result = subprocess.run(
        [sys.executable, "-m", "tests.distributed.test_dummy_forward", "dummy", family, backend],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VLM_DUMMY_INPUT_CASES_OK=6" in result.stdout


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("backend", ["gpu", "npu"])
@pytest.mark.parametrize("attention", ["eager", "veomni_sdpa", "veomni_flash_attention_2"])
def test_qwen4_exp_vision_forward_contract(backend, attention):
    """Test actual vision forward separately from dummy input construction."""
    result = subprocess.run(
        [sys.executable, "-m", "tests.distributed.test_dummy_forward", "forward", backend, attention],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VLM_FORWARD_CASES_OK=5" in result.stdout


def _vision_real_image_sp_non_fa_worker(attention, parallel_mode):
    from transformers import Qwen4ExpConfig

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state
    from veomni.ops.config import set_ops_config

    backend = "npu" if get_device_type() == "npu" else "gpu"
    module = importlib.import_module(
        f"veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_{backend}"
    )
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    try:
        state = _init_parallel_state(
            dp_size=1,
            cp_size=world_size if parallel_mode == "cp" else 1,
            ulysses_size=world_size if parallel_mode == "ulysses" else 1,
            device_type=get_device_type(),
            name=parallel_mode,
        )
        config = Qwen4ExpConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json").vision_config
        config._attn_implementation = attention
        set_ops_config(
            SimpleNamespace(
                rms_norm_gated_implementation="fla",
                causal_conv1d_implementation="fla",
                chunk_gated_delta_rule_implementation="fla",
                qsa_attention_implementation="eager",
            )
        )
        device = torch.device(get_device_type(), rank)
        vision = module.Qwen4ExpVisionModel(config).to(device=device, dtype=torch.bfloat16)
        for tensor in (*vision.parameters(), *vision.buffers()):
            dist.broadcast(tensor.data, src=0)
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixels = torch.randn(8 // world_size, patch_dim, device=device, dtype=torch.bfloat16)
        grid = torch.tensor([[1, 2, 4]], dtype=torch.long, device=device)
        metadata = {
            "grid_thw_list": [[1, 2, 4]],
            "cu_seqlens": torch.tensor([0, 8], dtype=torch.int32, device=device),
            "max_seqlen": 8,
        }
        with (
            use_parallel_state(state),
            pytest.raises(NotImplementedError, match="Qwen4-Exp vision sequence parallelism requires FlashAttention"),
        ):
            vision(pixels, grid_thw=grid, vit_metadata=metadata, return_dict=True)
    finally:
        clear_parallel_state()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="accelerator"))
@pytest.mark.parametrize(
    "attention,parallel_mode", [("eager", "cp"), ("eager", "ulysses"), ("veomni_sdpa", "ulysses")]
)
def test_qwen4_exp_vision_real_image_sp_rejects_non_flash_attention(attention, parallel_mode):
    from scripts.ci.accelerator.launch_plan import runner_device_count

    from ..tools.launch_utils import torchrun

    if runner_device_count() < 2:
        pytest.skip("requires two accelerators")
    if get_device_type() == "npu" and parallel_mode == "cp":
        pytest.skip("Qwen4-Exp Context Parallelism is currently supported on GPU only.")
    torchrun(_vision_real_image_sp_non_fa_worker, 2, attention, parallel_mode)


def _vision_parallel_state(sp_size, sp_rank):
    return SimpleNamespace(
        sp_enabled=sp_size > 1,
        sp_size=sp_size,
        sp_rank=sp_rank,
        ulysses_enabled=sp_size > 1,
        ulysses_size=sp_size,
        ulysses_rank=sp_rank,
        ulysses_group=None,
        cp_enabled=False,
        cp_size=1,
        cp_rank=0,
    )


def _install_vision_state(monkeypatch, module, sp_size, sp_rank):
    import veomni.distributed.sequence_parallel.data as sp_data

    state = _vision_parallel_state(sp_size, sp_rank)
    monkeypatch.setattr(module, "get_parallel_state", lambda: state)
    monkeypatch.setattr(sp_data, "get_parallel_state", lambda: state)


def _check_dummy_vision_inputs(family, backend):
    module = importlib.import_module(
        f"veomni.models.transformers.{family}.generated.patched_modeling_{family}_{backend}"
    )
    vision_cls = getattr(module, "Qwen3VLVisionModel" if family == "qwen3_vl" else "Qwen4ExpVisionModel")
    scale = 2 if family == "qwen3_vl" else 1

    class VisionProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=self.weight))
            self.config = SimpleNamespace(in_channels=3, temporal_patch_size=2, patch_size=4)
            self.spatial_merge_size = 2

        @property
        def dtype(self):
            return self.weight.dtype

        @property
        def device(self):
            return self.weight.device

        def forward(self, *, hidden_states, grid_thw, vit_metadata):
            return hidden_states, grid_thw, vit_metadata

    for sp_size, sp_rank in ((1, 0), (2, 0), (2, 1), (8, 7), (16, 15), (32, 31)):
        with pytest.MonkeyPatch.context() as monkeypatch:
            _install_vision_state(monkeypatch, module, sp_size, sp_rank)
            probe = VisionProbe()
            pixels, grid, metadata = vision_cls.dummy_forward(probe)
        h, w = scale * probe.spatial_merge_size * sp_size, scale * probe.spatial_merge_size
        local_rows = h * w // sp_size
        assert pixels.shape == (local_rows, 3 * 2 * 4**2)
        assert pixels.dtype == probe.weight.dtype and pixels.device == probe.weight.device
        assert not torch.count_nonzero(pixels)
        assert grid.tolist() == [[1, h, w]]
        assert metadata["grid_thw_list"] == [[1, h, w]]
        assert metadata["cu_seqlens"].device.type == "cpu"
        assert metadata["cu_seqlens"].tolist() == [0, h * w]
        assert metadata["max_seqlen"] == h * w
    print("VLM_DUMMY_INPUT_CASES_OK=6")


def _check_qwen4_exp_vision_forward(backend, attention):
    from transformers import Qwen4ExpConfig

    module = importlib.import_module(
        f"veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_{backend}"
    )
    for sp_size in (1, 2, 8, 16, 32):
        with pytest.MonkeyPatch.context() as monkeypatch:
            _install_vision_state(monkeypatch, module, sp_size, 0)
            config = Qwen4ExpConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json").vision_config
            # Construct without a hardware FlashAttention dependency; the
            # recorder below still exercises the real vision forward path.
            config._attn_implementation = "eager" if attention == "veomni_flash_attention_2" else attention
            torch.manual_seed(42)
            model = module.Qwen4ExpVisionModel(config).eval()
            merge = config.spatial_merge_size
            patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
            pixels = torch.zeros(merge * merge, patch_dim, dtype=torch.float32)
            grid = torch.tensor([[1, merge * sp_size, merge]], dtype=torch.long)
            global_rows = merge * merge * sp_size
            metadata = {
                "grid_thw_list": [[1, merge * sp_size, merge]],
                "cu_seqlens": torch.tensor([0, global_rows], dtype=torch.int32),
                "max_seqlen": global_rows,
            }
            if attention == "veomni_flash_attention_2":
                captured = []

                def record_attention(_module, query, _key, _value, *, _captured=captured, **kwargs):
                    _captured.append((query.shape[2], kwargs["cu_seq_lens_q"].tolist(), int(kwargs["max_length_q"])))
                    return query, None

                for active_config in (config, *(block.attn.config for block in model.blocks)):
                    active_config._attn_implementation = attention
                    if hasattr(active_config, "_attn_implementation_internal"):
                        active_config._attn_implementation_internal = attention
                for block in model.blocks:
                    block.attn.veomni_rope = lambda query, key, _cos, _sin: (query, key)
                    block.attn.veomni_attn = record_attention
                output = model.dummy_forward().pooler_output
                assert torch.isfinite(output).all()
                assert captured
                assert all(call == (merge * merge, [0, global_rows], global_rows) for call in captured)
            elif sp_size > 1:
                with pytest.raises(NotImplementedError, match="requires FlashAttention"):
                    model(pixels, grid, return_dict=True, vit_metadata=metadata)
            else:
                expected = model(pixels, grid, return_dict=True, vit_metadata=metadata).pooler_output
                output = model.dummy_forward().pooler_output
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                assert torch.isfinite(output).all()
                output.square().sum().backward()
                assert any(parameter.grad is not None for parameter in model.parameters())
    print("VLM_FORWARD_CASES_OK=5")


if __name__ == "__main__":
    # Isolate the CPU-only MoE import shim from the pytest parent process.
    import types

    package = types.ModuleType("veomni.distributed.moe")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "veomni/distributed/moe")]
    sys.modules[package.__name__] = package
    torch.set_num_threads(1)
    mode, *arguments = sys.argv[1:]
    if mode == "dummy":
        _check_dummy_vision_inputs(*arguments)
    elif mode == "forward":
        _check_qwen4_exp_vision_forward(*arguments)
    else:
        raise ValueError(f"Unknown vision contract mode: {mode}")
