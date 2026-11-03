# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Dynamic RoPE cache and alias state must not silently change on resume."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.optimizer import MultiOptimizer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_runtime_preflight_rejects_unsupported_dynamic_rope_before_training():
    from veomni.models.model_runtime import VeOmniModelRuntime

    model, optimizer = _build()
    runtime = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        model_name="policy",
        train_args=SimpleNamespace(checkpoint=SimpleNamespace(manager="dcp", save_async=False)),
    )
    with pytest.raises(RuntimeError, match="cannot yet safely resume dynamic RoPE"):
        VeOmniModelRuntime._validate_optimizer_checkpoint_mode(runtime)
    # A normal, static RoPE module and the standard optimizer remain allowed.
    model.rope.rope_type = "default"
    VeOmniModelRuntime._validate_optimizer_checkpoint_mode(runtime)


def _runtime_rope_guard_io_failure(rank, endpoint):
    from unittest.mock import patch

    from veomni.checkpoint import dynamic_rope
    from veomni.models.model_runtime import VeOmniModelRuntime

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(7)
        model, optimizer = _build()
        model.rope.rope_type = "default"  # Same supported model configuration on every rank.
        runtime = SimpleNamespace(
            model=model,
            optimizer=optimizer,
            model_name="policy",
            train_args=SimpleNamespace(checkpoint=SimpleNamespace(manager="dcp", save_async=False)),
        )
        original = dynamic_rope.reject_dynamic_rope_multi_optimizer_training
        failure = OSError("injected local RoPE validator failure") if rank == 1 else None
        with patch.object(
            dynamic_rope, "reject_dynamic_rope_multi_optimizer_training", wraps=original, side_effect=failure
        ) as guard:
            error = OSError if rank == 1 else RuntimeError
            message = (
                "injected local RoPE validator failure" if rank == 1 else "dynamic RoPE checkpoint mode.*another rank"
            )
            with pytest.raises(error, match=message):
                VeOmniModelRuntime._validate_optimizer_checkpoint_mode(runtime)
            guard.assert_called_once_with(model)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_dynamic_rope_validator_local_failure_reaches_all_ranks():
    spawn_with_rendezvous(_runtime_rope_guard_io_failure, nprocs=2)


class _TinyQwenRope(torch.nn.Module):
    def __init__(self, theta=10000):
        super().__init__()
        self.projection = torch.nn.Linear(8, 8)
        config = Qwen3Config(
            hidden_size=32,
            head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
            rope_parameters={"rope_type": "dynamic", "factor": 2.0, "rope_theta": theta},
        )
        self.rope = Qwen3RotaryEmbedding(config)


def _build(theta=10000):
    model = _TinyQwenRope(theta)
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
    DistributedCheckpointer.load(
        checkpoint,
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=SimpleNamespace(dp_mode="ddp"),
    )
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


def test_standard_qwen4_rope_128k_does_not_require_dynamic_cache_contract():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

    from veomni.checkpoint.dynamic_rope import dynamic_rope_checkpoint_contract

    config = Qwen4ExpTextConfig(
        hidden_size=32,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=262144,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
        },
    )
    rope = Qwen4ExpTextRotaryEmbedding(config)
    positions = torch.arange(131072).reshape(1, 1, -1).expand(3, 1, -1)
    rope(torch.zeros((1, 131072, 8)), positions)
    assert rope.max_seq_len_cached == 262144
    assert dynamic_rope_checkpoint_contract(rope) == []


def test_expanded_dynamic_rope_cache_is_rejected_before_checkpoint_write(tmp_path):
    model, optimizer = _build()
    model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
    assert int(model.rope.max_seq_len_cached) == 32
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="Dynamic RoPE cache.*depends on prior sequence lengths"):
        DistributedCheckpointer.save(
            str(checkpoint),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not checkpoint.exists()


def test_old_unproven_expanded_rope_checkpoint_is_rejected_before_model_load(tmp_path, monkeypatch):
    import veomni.checkpoint.dynamic_rope as dynamic_rope

    model, optimizer = _build()
    with torch.no_grad():
        model.projection.weight.fill_(7)
    model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
    parallel = SimpleNamespace(dp_mode="ddp")
    with monkeypatch.context() as old:
        old.setattr(dynamic_rope, "dynamic_rope_checkpoint_contract", lambda _model: [])
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    target, restored = _build()
    before = target.projection.weight.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.projection.weight, before, rtol=0, atol=0)


def test_changed_dynamic_rope_config_is_rejected_before_model_load(tmp_path):
    model, optimizer = _build()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = _build(theta=5000)
    before = target.projection.weight.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.projection.weight, before, rtol=0, atol=0)


def _rank_local_rope_error(rank, checkpoint, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model, optimizer = _build()
        if rank == 1:
            model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
        error = "Dynamic RoPE cache.*prior sequence lengths" if rank == 1 else "layout failed on another rank"
        with pytest.raises(RuntimeError, match=error):
            DistributedCheckpointer.save(
                checkpoint,
                {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_rank_local_expanded_rope_error_is_coordinated(tmp_path):
    spawn_with_rendezvous(_rank_local_rope_error, args=(str(tmp_path / "checkpoint"),), nprocs=2)
    assert not (tmp_path / "checkpoint").exists()


def test_real_qwen3_dynamic_rope_reset_is_rejected_before_save(tmp_path):
    model, optimizer = _build()
    _update(model, optimizer, 1.0)
    model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
    model.rope(torch.zeros((1, 1, 8)), torch.arange(4).view(1, -1))
    assert model.rope.inv_freq is model.rope.original_inv_freq
    assert not any("inv_freq" in key for key in model.state_dict())
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="Cannot prove dynamic RoPE checkpoint cache"):
        DistributedCheckpointer.save(
            str(checkpoint),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not checkpoint.exists()


def test_unexpanded_dynamic_rope_fresh_process_matches_training_update(tmp_path):
    model, optimizer = _build()
    _update(model, optimizer, 1.0)
    checkpoint = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(
        checkpoint,
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=SimpleNamespace(dp_mode="ddp"),
    )
    trace = _rope_update(model, optimizer)
    expected = tmp_path / "expected.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "trace": trace}, expected)
    mp.spawn(_fresh_resume, args=(checkpoint, str(expected)), nprocs=1, join=True)


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
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = build(not source_alias)
    before = target.weight.detach().clone()
    persistent_before = target.persistent.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)
    torch.testing.assert_close(target.persistent, persistent_before, rtol=0, atol=0)


@pytest.mark.parametrize("source_alias", [False, True])
def test_non_rope_transient_buffer_alias_mismatch_still_fails_before_load(tmp_path, source_alias):
    class Model(torch.nn.Module):
        def __init__(self, alias):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            first = torch.ones(2)
            self.register_buffer("cache_a", first, persistent=False)
            self.register_buffer("cache_b", first if alias else first.clone(), persistent=False)

    def build(alias):
        model = Model(alias)
        leaf = torch.optim.AdamW(model.parameters(), lr=0.01)
        return model, MultiOptimizer(model, {"model": leaf}, ["model"])

    model, optimizer = build(source_alias)
    with torch.no_grad():
        model.weight.fill_(5)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = build(not source_alias)
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_custom_dynamic_rope_alias_mismatch_still_fails_before_load(tmp_path):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    class CustomRope(torch.nn.Module):
        def __init__(self, alias):
            super().__init__()
            self.config = Qwen3Config(
                hidden_size=32,
                head_dim=8,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=16,
                rope_parameters={"rope_type": "dynamic", "factor": 2.0, "rope_theta": 10000},
            )
            self.rope_type = "dynamic"
            self.original_max_seq_len = self.max_seq_len_cached = 16
            inv, self.attention_scaling = ROPE_INIT_FUNCTIONS["dynamic"](self.config, device="cpu")
            self.register_buffer("inv_freq", inv, persistent=False)
            self.register_buffer("original_inv_freq", inv if alias else inv.clone(), persistent=False)

        def forward(self):
            self.inv_freq.add_(1)
            return self.original_inv_freq.clone()

    def build(alias):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(2))
        model.rope = CustomRope(alias)
        leaf = torch.optim.AdamW(model.parameters(), lr=0.01)
        return model, MultiOptimizer(model, {"model": leaf}, ["model"])

    model, optimizer = build(True)
    parallel = SimpleNamespace(dp_mode="ddp")
    with pytest.raises(RuntimeError, match="Cannot prove dynamic RoPE checkpoint cache"):
        DistributedCheckpointer.save(
            str(tmp_path / "checkpoint"),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    assert not (tmp_path / "checkpoint").exists()
