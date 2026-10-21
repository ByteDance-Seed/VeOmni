"""Tests for shared training helpers."""

from __future__ import annotations

import os

import pytest
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from tests.tools import training_utils
from veomni.utils.device import MOE_TRITON_DEVICE_TYPES


@pytest.mark.parametrize("previous_backend", [None, "veomni"])
def test_build_hf_reference_model_uses_models_and_restores_backend(monkeypatch, previous_backend):
    import veomni.models as models

    if previous_backend is None:
        monkeypatch.delenv("MODELING_BACKEND", raising=False)
    else:
        monkeypatch.setenv("MODELING_BACKEND", previous_backend)

    expected = object()
    captured = {}

    def fake_build_foundation_model(**kwargs):
        captured["backend"] = os.environ.get("MODELING_BACKEND")
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(models, "build_foundation_model", fake_build_foundation_model)

    actual = training_utils.build_hf_reference_model(
        "toy-config",
        torch_dtype="float32",
        init_device="cpu",
    )

    assert actual is expected
    assert captured["backend"] == "hf"
    assert captured["kwargs"]["config_path"] == "toy-config"
    assert captured["kwargs"]["weights_path"] is None
    assert captured["kwargs"]["torch_dtype"] == "float32"
    assert captured["kwargs"]["init_device"] == "cpu"
    assert captured["kwargs"]["ops_implementation"].attn_implementation == "eager"
    assert os.environ.get("MODELING_BACKEND") == previous_backend


def test_build_hf_reference_model_restores_backend_after_failure(monkeypatch):
    import veomni.models as models

    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    def fail_build_foundation_model(**_kwargs):
        assert os.environ["MODELING_BACKEND"] == "hf"
        raise RuntimeError("build failed")

    monkeypatch.setattr(models, "build_foundation_model", fail_build_foundation_model)

    with pytest.raises(RuntimeError, match="build failed"):
        training_utils.build_hf_reference_model(
            "toy-config",
            torch_dtype="float32",
            init_device="cpu",
        )

    assert os.environ["MODELING_BACKEND"] == "veomni"


def test_build_hf_reference_model_supports_an_unregistered_family(monkeypatch, tmp_path):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
        architectures=["Qwen3ForCausalLM"],
        attn_implementation="eager",
    )
    config.save_pretrained(tmp_path)
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    model = training_utils.build_hf_reference_model(
        str(tmp_path),
        torch_dtype="float32",
        init_device="cpu",
    )

    assert model.config.model_type == "qwen3"
    assert model.__class__.__module__.startswith("transformers.models.qwen3")
    assert os.environ["MODELING_BACKEND"] == "veomni"


@pytest.mark.parametrize(
    ("device_type", "expected_moe"),
    [
        (MOE_TRITON_DEVICE_TYPES[0], "fused_triton"),
        ("npu", "fused_npu"),
    ],
)
def test_deepseek_v4_ops_overrides_follow_active_device(monkeypatch, device_type, expected_moe):
    monkeypatch.setattr(training_utils, "get_device_type", lambda: device_type)

    overrides = training_utils.resolve_ops_overrides("deepseek_v4")

    assert "--model.ops_implementation.attn_implementation=eager" in overrides
    assert f"--model.ops_implementation.moe_implementation={expected_moe}" in overrides


def test_deepseek_v3_gpu_keeps_default_moe(monkeypatch):
    monkeypatch.setattr(training_utils, "get_device_type", lambda: MOE_TRITON_DEVICE_TYPES[0])

    overrides = training_utils.resolve_ops_overrides("deepseek_v3")

    assert not any(flag.startswith("--model.ops_implementation.moe_implementation=") for flag in overrides)


def test_gpu_pool_reservations_and_full_node_exclusion():
    import threading

    from scripts.ci.gpu_pool import GPUJob, run_jobs

    barrier = threading.Barrier(2, timeout=5)
    lock = threading.Lock()
    owned = set()
    seen = {}

    def execute(job, assigned):
        with lock:
            assert not owned.intersection(assigned)
            owned.update(assigned)
            seen[job.name] = list(assigned)
        if job.gpus == 4:
            barrier.wait()
        else:
            assert len(owned) == 8
        with lock:
            owned.difference_update(assigned)
        return job.name

    jobs = [GPUJob("a", 4, None), GPUJob("b", 4, None), GPUJob("exclusive", 8, None)]
    result = run_jobs(jobs, [str(i) for i in range(8)], execute)
    assert list(result) == ["a", "b", "exclusive"]
    assert set(seen["a"]).isdisjoint(seen["b"])
    assert len(seen["exclusive"]) == 8


def test_gpu_pool_does_not_dispatch_after_failure():
    from scripts.ci.gpu_pool import GPUJob, run_jobs

    seen = []

    def execute(job, assigned):
        seen.append(job.name)
        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError, match="worker failed"):
        run_jobs([GPUJob("bad", 8, None), GPUJob("pending", 8, None)], list("01234567"), execute)
    assert seen == ["bad"]


@pytest.mark.parametrize("size", [0, 9])
def test_gpu_pool_rejects_invalid_reservations(size):
    from scripts.ci.gpu_pool import GPUJob, run_jobs

    with pytest.raises(ValueError, match="reservation"):
        run_jobs([GPUJob("bad", size, None)], list("01234567"), lambda *_: None)


@pytest.mark.parametrize("backend,count", [("cpu", 0), ("cuda", 1), ("npu", 1)])
def test_launch_plan_default_and_sequential_concurrent_peaks(backend, count):
    from scripts.ci.launch_plan import CasePlan, LaunchSpec, RunnerCapabilities, default_plan

    caps = RunnerCapabilities(backend, () if backend == "cpu" else tuple("01234567"))
    assert default_plan(caps).cards(caps) == count
    first, second, cpu = LaunchSpec(nproc=1), LaunchSpec(nproc=2), LaunchSpec(nproc=4, device="cpu")
    sequential = CasePlan.sequential(first, second, cpu)
    concurrent = CasePlan.concurrent(second, second, cpu)
    assert sequential.cards(caps) == (2 if count else 0)
    assert concurrent.cards(caps) == (4 if count else 0)
    assert CasePlan.from_dict(concurrent.to_dict()) == concurrent
    assert cpu.nproc == 4 and cpu.cards(caps) == 0


def test_launch_plan_parameters_use_owned_capacity(monkeypatch, tmp_path):
    import json

    from scripts.ci.launch_plan import PlanContext, from_parameter, runner_capabilities, runner_device_count

    path = tmp_path / "caps.json"
    path.write_text(json.dumps(dict(backend="cuda", devices=list("01234567"))))
    monkeypatch.setenv("VEOMNI_CI_CAPABILITIES", str(path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert runner_device_count() == 8
    context = PlanContext(dict(cp_size=4), runner_capabilities())
    assert from_parameter("cp_size")(context).nproc == 4


@pytest.mark.parametrize("nproc,nnodes", [(0, 1), (True, 1), (2, 2)])
def test_launch_plan_rejects_invalid_local_launch(nproc, nnodes):
    from scripts.ci.launch_plan import LaunchSpec, ResourcePlanError

    with pytest.raises(ResourcePlanError):
        LaunchSpec(nproc=nproc, nnodes=nnodes)


def test_launch_validation_rejects_underdeclared_or_foreign_devices(monkeypatch, tmp_path):
    import json

    from scripts.ci.launch_plan import CasePlan, LaunchSpec, ResourcePlanError
    from scripts.ci.pytest_plan import validate_launch

    path = tmp_path / "caps.json"
    path.write_text(json.dumps(dict(backend="cuda", devices=list("01234567"))))
    monkeypatch.setenv("VEOMNI_CI_CAPABILITIES", str(path))
    monkeypatch.setenv("VEOMNI_CI_ACTIVE_PLAN", json.dumps(CasePlan.sequential(LaunchSpec(nproc=2)).to_dict()))
    monkeypatch.setenv("VEOMNI_CI_LEASE", '["4", "5"]')
    validate_launch(2, env={"CUDA_VISIBLE_DEVICES": "4,5"}, record=False)
    with pytest.raises(ResourcePlanError, match="disagrees"):
        validate_launch(4, record=False)
    with pytest.raises(ResourcePlanError, match="visibility"):
        validate_launch(2, env={"CUDA_VISIBLE_DEVICES": "0,1"}, record=False)
    mixed = CasePlan.sequential(LaunchSpec(nproc=2), LaunchSpec(nproc=2, device="cpu"))
    monkeypatch.setenv("VEOMNI_CI_ACTIVE_PLAN", json.dumps(mixed.to_dict()))
    validate_launch(2, env={"CUDA_VISIBLE_DEVICES": ""}, record=False)
    validate_launch(2, env={"CUDA_VISIBLE_DEVICES": "4,5"}, record=False)


def test_npu_adapter_preserves_owned_host_slice_and_resets_ranks():
    from scripts.ci.accelerator import child_environment
    from scripts.ci.launch_plan import ResourcePlanError, RunnerCapabilities

    caps = RunnerCapabilities("npu", tuple(str(i) for i in range(8, 16)))
    env = child_environment(caps, ["10", "11"], environ={"ASCEND_DEVICE_ID": "8", "RANK": "7"})
    assert env["ASCEND_VISIBLE_DEVICES"] == env["ASCEND_RT_VISIBLE_DEVICES"] == "10,11"
    assert "ASCEND_DEVICE_ID" not in env and "RANK" not in env
    assert env["HCCL_HOST_SOCKET_PORT_RANGE"] == env["HCCL_NPU_SOCKET_PORT_RANGE"] == "auto"
    with pytest.raises(ResourcePlanError, match="owned slice"):
        child_environment(caps, ["0", "1"], environ={})
    with pytest.raises(ResourcePlanError, match="ascending"):
        child_environment(caps, ["11", "10"], environ={})
    cpu = child_environment(caps, [], cpu=True, environ={})
    assert cpu["CUDA_VISIBLE_DEVICES"] == cpu["ASCEND_RT_VISIBLE_DEVICES"] == ""
    assert cpu["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"


def test_case_batches_preserve_inventory_and_resource_requests():
    from scripts.ci.run_accelerator_tests import make_batches

    rows = [dict(nodeid=f"case-{i}", file="test.py", cards=2 if i % 2 else 4) for i in range(9)]
    batches = make_batches([(dict(name="test"), rows)], target_seconds=120, max_cases=2)
    observed = [row for batch in batches for row in batch["cases"]]
    assert sorted(row["nodeid"] for row in observed) == sorted(row["nodeid"] for row in rows)
    assert all(len(batch["cases"]) <= 2 for batch in batches)
    assert all(row["cards"] == batch["cards"] for batch in batches for row in batch["cases"])
    assert [batch["cards"] for batch in batches] == [4, 4, 4, 2, 2]
    assert [row["nodeid"] for batch in batches if batch["cards"] == 4 for row in batch["cases"]] == [
        "case-0",
        "case-2",
        "case-4",
        "case-6",
        "case-8",
    ]
    assert all(batch["estimated_seconds"] == len(batch["cases"]) * 60 for batch in batches)

    # Fixed weights split one-card and CPU work without consulting measured time.
    for cards in (0, 1):
        single = [dict(nodeid=f"single-{i}", file="test.py", cards=cards) for i in range(61)]
        batches = make_batches([(dict(name="single"), single)])
        assert [len(batch["cases"]) for batch in batches] == [60, 1]
        assert [batch["estimated_seconds"] for batch in batches] == [120, 2]


def test_device_pool_drains_for_waiting_large_job():
    from scripts.ci.device_pool import DeviceJob, run_jobs

    order = []
    jobs = [DeviceJob("first", 4, None), DeviceJob("large", 8, None), DeviceJob("backfill", 4, None)]
    run_jobs(jobs, list("01234567"), lambda job, assigned: order.append(job.name), drain_after=0)
    assert order == ["first", "large", "backfill"]


def test_device_pool_can_run_cpu_work_beside_an_exclusive_device_job():
    import threading

    from scripts.ci.device_pool import DeviceJob, run_jobs

    barrier = threading.Barrier(2, timeout=5)
    assignments = {}

    def execute(job, assigned):
        assignments[job.name] = assigned
        barrier.wait()

    run_jobs([DeviceJob("accelerator", 8, None), DeviceJob("cpu", 0, None)], list("01234567"), execute)
    assert assignments["cpu"] == []
    assert assignments["accelerator"] == list("01234567")


def test_case_workspaces_isolate_relative_checkpoints_and_caches(tmp_path):
    from scripts.ci.run_accelerator_tests import _workspace

    first, second = tmp_path / "first", tmp_path / "second"
    _workspace(first)
    _workspace(second)
    (first / "cache" / "dataset").write_text("first")
    assert not (second / "cache" / "dataset").exists()
    assert (first / "tests").resolve() == (second / "tests").resolve()


def test_rendezvous_allocator_does_not_reuse_a_pending_launch_port(monkeypatch, tmp_path):
    from scripts.ci import ports

    candidates = iter([12345, 12345, 12346])
    monkeypatch.setattr(ports, "_free_port", lambda: next(candidates))
    registry = str(tmp_path / "ports.json")
    assert ports.find_free_port(registry) == 12345
    assert ports.find_free_port(registry) == 12346


def test_callable_launch_provider_is_not_used_as_a_pytest_decorator():
    from scripts.ci.launch_plan import PlanContext, RunnerCapabilities, from_parameter

    provider = from_parameter("nproc")
    marker = pytest.mark.launch_plan(provider=provider)

    @marker
    def sample():
        return "test body"

    assert sample() == "test body"
    context = PlanContext(dict(nproc=4), RunnerCapabilities("cuda", tuple("01234567")))
    assert sample.pytestmark[0].kwargs["provider"](context).nproc == 4


def test_unittest_plan_reads_world_size_without_running_constructor(monkeypatch):
    from types import SimpleNamespace

    from scripts.ci import pytest_plan
    from scripts.ci.launch_plan import LaunchSpec, RunnerCapabilities

    class DistributedCase:
        def __init__(self):
            raise AssertionError("Collection must not construct a distributed test")

        @property
        def world_size(self):
            return 4

        def ci_launch_plan(self):
            return LaunchSpec(nproc=self.world_size)

    monkeypatch.setattr(pytest_plan, "runner_capabilities", lambda: RunnerCapabilities("cuda", tuple("01234567")))
    item = SimpleNamespace(instance=None, cls=DistributedCase, get_closest_marker=lambda _: None)
    assert pytest_plan.resolve_plan(item).launches[0].nproc == 4
