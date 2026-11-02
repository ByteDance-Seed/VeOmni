# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Transient RoPE buffers are not part of checkpoint ownership identity."""

from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.optimizer import MultiOptimizer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _TinyQwenRope(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 8)
        config = Qwen3Config(
            hidden_size=32,
            head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
            rope_parameters={"rope_type": "dynamic", "factor": 2.0, "rope_theta": 10000},
        )
        self.rope = Qwen3RotaryEmbedding(config)


def _build():
    model = _TinyQwenRope()
    leaf = torch.optim.AdamW(model.parameters(), lr=0.01)
    return model, MultiOptimizer(model, {"model": leaf}, ["model"])


def _update(model, optimizer, value):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, value)
    optimizer.step()
    optimizer.zero_grad()


def _rope_update(model, optimizer):
    inputs = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8) / 64
    projected = model.projection(inputs)
    cos, sin = model.rope(projected, torch.arange(8).reshape(1, -1))
    loss = (projected * cos + sin).square().mean()
    loss.backward()
    trace = {"cos": cos.detach().clone(), "sin": sin.detach().clone(), "loss": loss.detach().clone()}
    trace.update({f"grad.{name}": value.grad.detach().clone() for name, value in model.named_parameters()})
    optimizer.step()
    optimizer.zero_grad()
    return trace


def _fresh_resume(_rank, checkpoint, expected_path):
    model, optimizer = _build()
    assert model.rope.inv_freq is not model.rope.original_inv_freq
    DistributedCheckpointer.load(
        checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    assert model.rope.inv_freq is not model.rope.original_inv_freq
    trace = _rope_update(model, optimizer)
    expected = torch.load(expected_path, map_location="cpu", weights_only=True)
    actual = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "trace": trace}
    for group in ("model", "optimizer", "trace"):
        assert actual[group].keys() == expected[group].keys()
        for key, value in actual[group].items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, expected[group][key], rtol=0, atol=0)
            else:
                assert value == expected[group][key]


def test_old_all_buffer_identity_rejects_real_rope_cache_reset(tmp_path, monkeypatch):
    import veomni.checkpoint.optimizer_resume_layout as layout

    monkeypatch.setattr(layout, "persistent_named_buffers", lambda model: model.named_buffers(remove_duplicate=False))
    model, optimizer = _build()
    model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
    model.rope(torch.zeros((1, 1, 8)), torch.arange(4).view(1, -1))
    assert model.rope.inv_freq is model.rope.original_inv_freq
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = _build()
    assert target.rope.inv_freq is not target.rope.original_inv_freq
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)


def test_real_qwen3_dynamic_rope_reset_saves_and_fresh_process_resumes(tmp_path):
    model, optimizer = _build()
    _update(model, optimizer, 1.0)
    model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
    model.rope(torch.zeros((1, 1, 8)), torch.arange(4).view(1, -1))
    assert model.rope.inv_freq is model.rope.original_inv_freq
    assert not any("inv_freq" in key for key in model.state_dict())
    checkpoint = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(
        checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
    )
    trace = _rope_update(model, optimizer)
    expected = tmp_path / "expected.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "trace": trace}, expected)
    mp.spawn(_fresh_resume, args=(checkpoint, str(expected)), nprocs=1, join=True)


def test_distinct_overlapping_nonpersistent_buffer_views_still_fail_closed(tmp_path):
    model, optimizer = _build()
    storage = torch.zeros(8)
    model.register_buffer("cache_view_a", storage[:6], persistent=False)
    model.register_buffer("cache_view_b", storage[2:], persistent=False)
    with pytest.raises(RuntimeError, match="checkpoint model storage alias"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )


@pytest.mark.parametrize("source_alias", [False, True])
@pytest.mark.parametrize("anchor_kind", ["parameter", "persistent_buffer"])
def test_nonpersistent_buffer_persistent_alias_mismatch_still_fails_before_load(tmp_path, source_alias, anchor_kind):
    class Model(torch.nn.Module):
        def __init__(self, alias):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("persistent", torch.ones(2))
            anchor = self.weight if anchor_kind == "parameter" else self.persistent
            self.register_buffer("cache", anchor if alias else anchor.detach().clone(), persistent=False)

    def build(alias):
        model = Model(alias)
        leaf = torch.optim.AdamW(model.parameters(), lr=0.01)
        return model, MultiOptimizer(model, {"model": leaf}, ["model"])

    model, optimizer = build(source_alias)
    with torch.no_grad():
        model.weight.fill_(5)
        model.persistent.fill_(7)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = build(not source_alias)
    before = target.weight.detach().clone()
    persistent_before = target.persistent.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)
    torch.testing.assert_close(target.persistent, persistent_before, rtol=0, atol=0)
