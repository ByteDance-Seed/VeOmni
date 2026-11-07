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


def test_runtime_preflight_allows_dynamic_rope_training_without_checkpoint():
    from veomni.models.model_runtime import VeOmniModelRuntime

    model, optimizer = _build()
    runtime = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        model_name="policy",
        train_args=SimpleNamespace(checkpoint=SimpleNamespace(manager="dcp", save_async=False)),
    )
    VeOmniModelRuntime._validate_optimizer_checkpoint_mode(runtime)
    trace = _rope_update(model, optimizer)
    assert torch.isfinite(trace["loss"])
    model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
    assert int(model.rope.max_seq_len_cached) == 32
    VeOmniModelRuntime._validate_optimizer_checkpoint_mode(runtime)
    _update(model, optimizer, 0.25)


class _TinyQwenRope(torch.nn.Module):
    def __init__(self, theta=10000, rope_class=Qwen3RotaryEmbedding):
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
        self.rope = rope_class(config)


def _build(theta=10000, rope_class=Qwen3RotaryEmbedding):
    model = _TinyQwenRope(theta, rope_class)
    leaf = torch.optim.AdamW(model.parameters(), lr=0.01)
    return model, MultiOptimizer(model, {"model": leaf}, ["model"])


def _registry_rope_class(backend):
    import importlib
    from unittest.mock import patch

    from veomni.models.registry import MODELING_REGISTRY
    from veomni.models.transformers import qwen3

    with patch.object(qwen3, "IS_NPU_AVAILABLE", backend == "npu"):
        model_class = MODELING_REGISTRY["qwen3"]("Qwen3Model")
    assert model_class.__module__.endswith(f"patched_modeling_qwen3_{backend}")
    return importlib.import_module(model_class.__module__).Qwen3RotaryEmbedding


def _update(model, optimizer, value):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, value)
    optimizer.step()
    optimizer.zero_grad()


def _rope_update(model, optimizer, length=8):
    inputs = torch.arange(length * 8, dtype=torch.float32).reshape(1, length, 8) / (length * 8)
    projected = model.projection(inputs)
    cos, sin = model.rope(projected, torch.arange(length).reshape(1, -1))
    loss = (projected * cos + sin).square().mean()
    loss.backward()
    trace = {"cos": cos.detach().clone(), "sin": sin.detach().clone(), "loss": loss.detach().clone()}
    trace.update({f"grad.{name}": value.grad.detach().clone() for name, value in model.named_parameters()})
    optimizer.step()
    optimizer.zero_grad()
    return trace


def _rope_transitions(model, optimizer):
    result = {}
    for index, length in enumerate((8, 16, 17, 16, 15, 32, 24, 4)):
        trace = _rope_update(model, optimizer, length)
        label = f"{index}:{length}"
        result.update({f"{label}.{key}": value for key, value in trace.items()})
        result[f"{label}.inv_freq"] = model.rope.inv_freq.detach().clone()
        result[f"{label}.cached_length"] = torch.tensor(int(model.rope.max_seq_len_cached))
        result.update({f"{label}.param.{name}": value.detach().clone() for name, value in model.named_parameters()})
    return result


def _fresh_resume(_rank, checkpoint, expected_path, backend=None):
    rope_class = Qwen3RotaryEmbedding if backend is None else _registry_rope_class(backend)
    model, optimizer = _build(rope_class=rope_class)
    DistributedCheckpointer.load(
        checkpoint,
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=SimpleNamespace(dp_mode="ddp"),
    )
    trace = _rope_transitions(model, optimizer)
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


def test_qwen4_default_rope_keeps_legacy_checkpoint_layout(tmp_path):
    import json

    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

    from veomni.checkpoint.optimizer_resume_layout import capture_multi_layout

    def build():
        model, optimizer = _build()
        config = Qwen4ExpTextConfig(
            hidden_size=32,
            head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=262144,
            rope_parameters={"rope_type": "default", "rope_theta": 10000000, "mrope_section": [1, 1, 2]},
        )
        model.rope = Qwen4ExpTextRotaryEmbedding(config)
        return model, optimizer

    model, optimizer = build()
    _update(model, optimizer, 1.0)
    layout = json.loads(capture_multi_layout(model, optimizer, None, None))
    assert "dynamic_rope_cache_v1" not in layout and "dynamic_rope_cache_v2" not in layout
    checkpoint = str(tmp_path / "checkpoint")
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        checkpoint, {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None}, parallel_state=parallel
    )
    target, restored = build()
    DistributedCheckpointer.load(
        checkpoint, {"model": target, "optimizer": restored, "_veomni_scheduler_object": None}, parallel_state=parallel
    )
    for actual, expected in zip(target.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                restored.optimizers_dict["model"].state[actual][key],
                optimizer.optimizers_dict["model"].state[expected][key],
                rtol=0,
                atol=0,
            )


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


def _skew_device_recomputation(monkeypatch):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    original = ROPE_INIT_FUNCTIONS["dynamic"]

    def skewed(config, device, seq_len=None, **kwargs):
        values, scale = original(config, device, seq_len=seq_len, **kwargs)
        # Model the observed CPU/NPU 1-ULP difference without requiring an NPU.
        if isinstance(device, torch.device):
            values = torch.nextafter(values, torch.full_like(values, float("inf")))
        return values, scale

    monkeypatch.setitem(ROPE_INIT_FUNCTIONS, "dynamic", skewed)


@pytest.mark.parametrize("backend", ["hf", "gpu", "npu"])
def test_cpu_origin_rope_with_device_rounding_difference_resumes_exactly(tmp_path, monkeypatch, backend):
    rope_class = Qwen3RotaryEmbedding if backend == "hf" else _registry_rope_class(backend)
    model, optimizer = _build(rope_class=rope_class)
    _update(model, optimizer, 1.0)
    _skew_device_recomputation(monkeypatch)
    checkpoint = str(tmp_path / "checkpoint")
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        checkpoint, {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None}, parallel_state=parallel
    )
    target, restored = _build(rope_class=rope_class)
    DistributedCheckpointer.load(
        checkpoint, {"model": target, "optimizer": restored, "_veomni_scheduler_object": None}, parallel_state=parallel
    )
    expected = _rope_transitions(model, optimizer)
    actual = _rope_transitions(target, restored)
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["hf", "gpu", "npu"])
def test_distinct_cpu_and_device_origin_rope_is_rejected_before_restore(tmp_path, monkeypatch, backend):
    rope_class = Qwen3RotaryEmbedding if backend == "hf" else _registry_rope_class(backend)
    model, optimizer = _build(rope_class=rope_class)
    _skew_device_recomputation(monkeypatch)
    checkpoint = str(tmp_path / "checkpoint")
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        checkpoint, {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None}, parallel_state=parallel
    )
    target, restored = _build(rope_class=rope_class)
    shifted = torch.nextafter(target.rope.inv_freq, torch.full_like(target.rope.inv_freq, float("inf")))
    target.rope.inv_freq.copy_(shifted)
    target.rope.original_inv_freq.copy_(shifted)
    before = target.projection.weight.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(
            checkpoint,
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.projection.weight, before, rtol=0, atol=0)
    assert not restored.optimizers_dict["model"].state


def test_signed_zero_in_rope_table_is_not_a_canonical_match(tmp_path):
    model, optimizer = _build(theta=1e12)
    model.rope.inv_freq = model.rope.inv_freq.to(torch.float16)
    model.rope.original_inv_freq = model.rope.original_inv_freq.to(torch.float16)
    assert model.rope.inv_freq[-1].item() == 0
    model.rope.inv_freq[-1] = -0.0
    model.rope.original_inv_freq[-1] = -0.0
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="Dynamic RoPE cache.*depends on prior sequence lengths"):
        DistributedCheckpointer.save(
            str(checkpoint),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not checkpoint.exists()


def test_nonfinite_canonical_rope_is_rejected(tmp_path, monkeypatch):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    model, optimizer = _build()
    model.rope.inv_freq[0] = float("nan")
    model.rope.original_inv_freq[0] = float("nan")
    original = ROPE_INIT_FUNCTIONS["dynamic"]

    def nonfinite(config, device, seq_len=None, **kwargs):
        value, scale = original(config, device, seq_len=seq_len, **kwargs)
        value[0] = float("nan")
        return value, scale

    monkeypatch.setitem(ROPE_INIT_FUNCTIONS, "dynamic", nonfinite)
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="Cannot prove|Dynamic RoPE cache"):
        DistributedCheckpointer.save(
            str(checkpoint),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not checkpoint.exists()


def test_changed_qwen3_rope_implementation_is_rejected_before_restore(tmp_path):
    model, optimizer = _build()
    checkpoint = str(tmp_path / "checkpoint")
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        checkpoint, {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None}, parallel_state=parallel
    )
    target, restored = _build(rope_class=_registry_rope_class("gpu"))
    torch.testing.assert_close(model.rope.inv_freq, target.rope.inv_freq, rtol=0, atol=0)
    before = target.projection.weight.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(
            checkpoint,
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.projection.weight, before, rtol=0, atol=0)


def test_old_dynamic_rope_contract_without_table_identity_is_rejected(tmp_path, monkeypatch):
    import veomni.checkpoint.optimizer_resume_layout as layout

    model, optimizer = _build()
    original = layout._local_layout

    def capture_legacy(*args, **kwargs):
        payload = original(*args, **kwargs)
        if kwargs.get("capture"):
            payload["dynamic_rope_cache_v1"] = [item[:6] for item in payload.pop("dynamic_rope_cache_v2")]
        return payload

    checkpoint = str(tmp_path / "checkpoint")
    parallel = SimpleNamespace(dp_mode="ddp")
    with monkeypatch.context() as legacy:
        legacy.setattr(layout, "_local_layout", capture_legacy)
        DistributedCheckpointer.save(
            checkpoint,
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    target, restored = _build()
    before = target.projection.weight.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(
            checkpoint,
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


def test_real_qwen3_dynamic_rope_reset_fresh_process_matches_next_update(tmp_path):
    model, optimizer = _build()
    _update(model, optimizer, 1.0)
    model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
    model.rope(torch.zeros((1, 1, 8)), torch.arange(4).view(1, -1))
    assert model.rope.inv_freq is model.rope.original_inv_freq
    assert not any("inv_freq" in key for key in model.state_dict())
    checkpoint = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(
        checkpoint,
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=SimpleNamespace(dp_mode="ddp"),
    )
    trace = _rope_transitions(model, optimizer)
    expected = tmp_path / "expected.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "trace": trace}, expected)
    mp.spawn(_fresh_resume, args=(checkpoint, str(expected)), nprocs=1, join=True)


@pytest.mark.parametrize("backend", ["gpu", "npu"])
def test_registry_qwen3_dynamic_rope_reset_fresh_process_matches_next_update(tmp_path, backend):
    model, optimizer = _build(rope_class=_registry_rope_class(backend))
    _update(model, optimizer, 1.0)
    model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
    model.rope(torch.zeros((1, 1, 8)), torch.arange(4).view(1, -1))
    assert model.rope.inv_freq is model.rope.original_inv_freq
    checkpoint = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(
        checkpoint,
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=SimpleNamespace(dp_mode="ddp"),
    )
    trace = _rope_transitions(model, optimizer)
    expected = tmp_path / "expected.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "trace": trace}, expected)
    mp.spawn(_fresh_resume, args=(checkpoint, str(expected), backend), nprocs=1, join=True)


def _rank_local_rope_reset(rank, checkpoint, directory, phase, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(13)
        expected = str(directory / f"expected-{rank}.pt")
        if phase == "load":
            _fresh_resume(rank, checkpoint, expected)
            return
        model, optimizer = _build()
        _update(model, optimizer, 1.0)
        if rank == 0:
            model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
            model.rope(torch.zeros((1, 1, 8)), torch.arange(4).view(1, -1))
        assert (model.rope.inv_freq is model.rope.original_inv_freq) == (rank == 0)
        DistributedCheckpointer.save(
            checkpoint,
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
        trace = _rope_transitions(model, optimizer)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "trace": trace}, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_rank_local_qwen3_rope_reset_fresh_resume(tmp_path):
    checkpoint = str(tmp_path / "checkpoint")
    spawn_with_rendezvous(_rank_local_rope_reset, args=(checkpoint, tmp_path, "save"), nprocs=2)
    spawn_with_rendezvous(_rank_local_rope_reset, args=(checkpoint, tmp_path, "load"), nprocs=2)


def test_qwen3_reset_alias_still_rejects_wrong_frequency_values(tmp_path):
    model, optimizer = _build()
    model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
    model.rope(torch.zeros((1, 1, 8)), torch.arange(4).view(1, -1))
    model.rope.inv_freq.add_(1)
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="Dynamic RoPE cache.*depends on prior sequence lengths"):
        DistributedCheckpointer.save(
            str(checkpoint),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not checkpoint.exists()


def test_qwen3_reset_alias_with_third_buffer_keeps_identity_check(tmp_path):
    model, optimizer = _build()
    model.rope(torch.zeros((1, 1, 8)), torch.arange(32).view(1, -1))
    model.rope(torch.zeros((1, 1, 8)), torch.arange(4).view(1, -1))
    model.rope.register_buffer("other_cache", model.rope.inv_freq, persistent=False)
    checkpoint = str(tmp_path / "checkpoint")
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        checkpoint, {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None}, parallel_state=parallel
    )
    target, restored = _build()
    target.rope.register_buffer("other_cache", target.rope.inv_freq.clone(), persistent=False)
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(
            checkpoint,
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )


def test_unexpanded_dynamic_rope_fresh_process_matches_training_update(tmp_path):
    model, optimizer = _build()
    _update(model, optimizer, 1.0)
    checkpoint = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(
        checkpoint,
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=SimpleNamespace(dp_mode="ddp"),
    )
    trace = _rope_transitions(model, optimizer)
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
