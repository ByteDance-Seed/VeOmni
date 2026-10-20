"""Exercise the real vision forward; a shape-only dummy stub misses SP regressions."""

import copy
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen4ExpConfig


def _check_dummy_vision(monkeypatch, backend, sp_size, device="cpu", dtype=torch.float32):
    module = importlib.import_module(
        f"veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_{backend}"
    )
    state = SimpleNamespace(
        sp_enabled=sp_size > 1,
        sp_size=sp_size,
        sp_rank=0,
        ulysses_enabled=sp_size > 1,
        ulysses_size=sp_size,
        ulysses_rank=0,
        ulysses_group=None,
    )
    monkeypatch.setattr(module, "get_parallel_state", lambda: state)
    # ``sp_pad_and_slice`` lives in this module and resolves the state through
    # its own import, so patching only the generated modeling module would make
    # the reference take a different (global) position-embedding path.
    import veomni.distributed.sequence_parallel.data as sp_data

    monkeypatch.setattr(sp_data, "get_parallel_state", lambda: state)
    config = Qwen4ExpConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json").vision_config
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    model = module.Qwen4ExpVisionModel(config).to(device=device, dtype=dtype).eval()
    reference = copy.deepcopy(model)
    merge = config.spatial_merge_size
    local_count = merge * merge
    grid = torch.tensor([[1, merge * sp_size, merge]], device=device)
    pixels = torch.zeros(
        local_count,
        config.in_channels * config.temporal_patch_size * config.patch_size**2,
        device=device,
        dtype=dtype,
    )
    vit_metadata = {
        "grid_thw_list": [[1, merge * sp_size, merge]],
        "cu_seqlens": torch.tensor([0, local_count], dtype=torch.int32, device="cpu"),
        "max_seqlen": local_count,
    }
    if sp_size > 1:
        with pytest.raises(NotImplementedError, match="requires FlashAttention"):
            reference(pixels, grid, return_dict=True, vit_metadata=vit_metadata)
    expected = reference(pixels, grid, return_dict=True, vit_metadata=vit_metadata, vision_dummy=True)
    actual = model.dummy_forward().pooler_output
    expected = expected.pooler_output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for parameter, other in zip(model.parameters(), reference.parameters(), strict=True):
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)


def test_dummy_vision_matches_global_patch_forward():
    result = subprocess.run([sys.executable, __file__], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VISION_DUMMY_CASES_OK=10" in result.stdout


if __name__ == "__main__":
    # Isolate the CPU-only MoE import shim from the pytest parent. No MoE
    # computation is involved in this real vision forward/backward regression.
    import types

    package = types.ModuleType("veomni.distributed.moe")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "veomni/distributed/moe")]
    sys.modules[package.__name__] = package
    torch.set_num_threads(1)
    native = "--native-npu" in sys.argv
    device, dtype = "cpu", torch.float32
    if native:
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
        device, dtype = "npu:0", torch.bfloat16
    for backend in ("npu",) if native else ("gpu", "npu"):
        for sp_size in (1, 2, 8, 16, 32):
            with pytest.MonkeyPatch.context() as monkeypatch:
                _check_dummy_vision(monkeypatch, backend, sp_size, device, dtype)
    print("VISION_DUMMY_NATIVE_NPU_CASES_OK=5" if native else "VISION_DUMMY_CASES_OK=10")
