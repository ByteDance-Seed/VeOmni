"""Keep Qwen4-Exp dummy vision on the same SP attention contract as real images."""

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen4ExpConfig


_TOY_CONFIG = "tests/toy_config/qwen4_exp_toy/config.json"


def _parallel_state(sp_size):
    return SimpleNamespace(
        sp_enabled=sp_size > 1,
        sp_size=sp_size,
        sp_rank=0,
        ulysses_enabled=sp_size > 1,
        ulysses_size=sp_size,
        ulysses_rank=0,
        ulysses_group=None,
        cp_enabled=False,
        cp_size=1,
        cp_rank=0,
    )


def _model(module, *, attention):
    config = Qwen4ExpConfig.from_pretrained(_TOY_CONFIG).vision_config
    # Only the FlashAttention recorder case defers binding a hardware kernel.
    # Eager and SDPA must bind their actual implementations during construction.
    config._attn_implementation = "eager" if attention == "veomni_flash_attention_2" else attention
    torch.manual_seed(42)
    model = module.Qwen4ExpVisionModel(config).eval()
    if attention == "veomni_flash_attention_2":
        for active_config in (config, *(block.attn.config for block in model.blocks)):
            active_config._attn_implementation = attention
            if hasattr(active_config, "_attn_implementation_internal"):
                active_config._attn_implementation_internal = attention
    return config, model


def _patch_parallel_state(monkeypatch, module, state):
    monkeypatch.setattr(module, "get_parallel_state", lambda: state)
    import veomni.distributed.sequence_parallel.data as sp_data

    # ``sp_pad_and_slice`` resolves the state through its own module import.
    monkeypatch.setattr(sp_data, "get_parallel_state", lambda: state)


def _dummy_inputs(config, sp_size):
    merge = config.spatial_merge_size
    local_count = merge * merge
    global_count = merge * merge * sp_size
    flattened_patch_size = config.in_channels * config.temporal_patch_size * config.patch_size**2
    pixels = torch.zeros(local_count, flattened_patch_size, dtype=torch.float32)
    grid = torch.tensor([[1, merge * sp_size, merge]], dtype=torch.long)
    metadata = {
        "grid_thw_list": [[1, merge * sp_size, merge]],
        "cu_seqlens": torch.tensor([0, global_count], dtype=torch.int32),
        "max_seqlen": global_count,
    }
    return pixels, grid, metadata, local_count, global_count


def _check_non_flash_contract(monkeypatch, backend, sp_size, attention):
    module = importlib.import_module(
        f"veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_{backend}"
    )
    _patch_parallel_state(monkeypatch, module, _parallel_state(sp_size))
    config, model = _model(module, attention=attention)
    pixels, grid, metadata, _, _ = _dummy_inputs(config, sp_size)

    if sp_size > 1:
        with pytest.raises(NotImplementedError, match="requires FlashAttention"):
            model.dummy_forward()
        return

    expected = model(pixels, grid, return_dict=True, vit_metadata=metadata).pooler_output
    actual = model.dummy_forward().pooler_output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def _check_flash_contract(monkeypatch, backend, sp_size):
    module = importlib.import_module(
        f"veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_{backend}"
    )
    _patch_parallel_state(monkeypatch, module, _parallel_state(sp_size))
    config, model = _model(module, attention="veomni_flash_attention_2")
    _, _, _, local_count, global_count = _dummy_inputs(config, sp_size)
    captured = []

    def fake_rope(query_states, key_states, _cos, _sin):
        return query_states, key_states

    def fake_attention(_module, query_states, _key_states, _value_states, **kwargs):
        captured.append(
            {
                "query_rows": query_states.shape[2],
                "cu_seqlens": kwargs["cu_seq_lens_q"].tolist(),
                "max_seqlen": int(kwargs["max_length_q"]),
            }
        )
        return query_states, None

    for block in model.blocks:
        block.attn.veomni_rope = fake_rope
        block.attn.veomni_attn = fake_attention

    output = model.dummy_forward().pooler_output
    assert torch.isfinite(output).all()
    assert captured
    for call in captured:
        # FlashAttention gathers the local rows into the global sequence before
        # applying the global varlen boundaries.
        assert call["query_rows"] == local_count
        assert call["cu_seqlens"] == [0, global_count]
        assert call["max_seqlen"] == global_count


def test_dummy_vision_keeps_the_real_image_sp_attention_contract():
    result = subprocess.run([sys.executable, __file__], capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VISION_DUMMY_CONTRACT_CASES_OK=30" in result.stdout


if __name__ == "__main__":
    # The CPU-only MoE import shim is isolated from the pytest parent. No MoE
    # computation is involved in this vision-forward regression.
    import types

    package = types.ModuleType("veomni.distributed.moe")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "veomni/distributed/moe")]
    sys.modules[package.__name__] = package
    torch.set_num_threads(1)
    cases = 0
    for backend in ("gpu", "npu"):
        for sp_size in (1, 2, 8, 16, 32):
            for attention in ("eager", "veomni_sdpa"):
                with pytest.MonkeyPatch.context() as monkeypatch:
                    _check_non_flash_contract(monkeypatch, backend, sp_size, attention)
                cases += 1
            with pytest.MonkeyPatch.context() as monkeypatch:
                _check_flash_contract(monkeypatch, backend, sp_size)
            cases += 1
    print(f"VISION_DUMMY_CONTRACT_CASES_OK={cases}")
