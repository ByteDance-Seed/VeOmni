"""CPU tests for accelerator CI planning, batching, and scheduling."""

import pytest

from scripts.ci.accelerator.launch_plan import CasePlan, LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_device_pool_reservations_and_full_node_exclusion():
    import threading

    from scripts.ci.accelerator.scheduler import DeviceJob, run_jobs

    barrier = threading.Barrier(2, timeout=5)
    lock = threading.Lock()
    owned = set()
    seen = {}

    def execute(job, assigned):
        with lock:
            assert not owned.intersection(assigned)
            owned.update(assigned)
            seen[job.name] = list(assigned)
        if job.cards == 4:
            barrier.wait()
        else:
            assert len(owned) == 8
        with lock:
            owned.difference_update(assigned)
        return job.name

    jobs = [DeviceJob("a", 4, None), DeviceJob("b", 4, None), DeviceJob("exclusive", 8, None)]
    result = run_jobs(jobs, [str(i) for i in range(8)], execute)
    assert list(result) == ["a", "b", "exclusive"]
    assert set(seen["a"]).isdisjoint(seen["b"])
    assert len(seen["exclusive"]) == 8


def test_device_pool_does_not_dispatch_after_failure():
    from scripts.ci.accelerator.scheduler import DeviceJob, run_jobs

    seen = []

    def execute(job, assigned):
        seen.append(job.name)
        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError, match="worker failed"):
        run_jobs([DeviceJob("bad", 8, None), DeviceJob("pending", 8, None)], list("01234567"), execute)
    assert seen == ["bad"]


@pytest.mark.parametrize("backend,count", [("cpu", 0), ("cuda", 1), ("npu", 1)])
def test_launch_plan_default_and_sequential_concurrent_peaks(backend, count):
    from scripts.ci.accelerator.launch_plan import CasePlan, LaunchSpec, RunnerCapabilities, default_plan

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

    from scripts.ci.accelerator.launch_plan import (
        PlanContext,
        from_parameter,
        runner_capabilities,
        runner_device_count,
    )

    path = tmp_path / "caps.json"
    path.write_text(json.dumps(dict(backend="cuda", devices=list("01234567"))))
    monkeypatch.setenv("VEOMNI_CI_CAPABILITIES", str(path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert runner_device_count() == 8
    context = PlanContext(dict(cp_size=4), runner_capabilities())
    assert from_parameter("cp_size")(context).nproc == 4


@pytest.mark.parametrize("backend", ["cuda", "npu"])
def test_e2e_alignment_preserves_fused_moe_training_paths(backend):
    from scripts.ci.accelerator.launch_plan import RunnerCapabilities
    from tests.e2e.utils import alignment_plan, prepare_exec_cmd

    caps = RunnerCapabilities(backend, tuple("01234567"))
    for model_name in ("gpt_oss", "qwen3_moe"):
        options = dict(is_moe=True, model_name=model_name)
        commands = prepare_exec_cmd(
            ["train_text_test"],
            config_path="config",
            model_path="weights",
            train_path="data",
            output_dir="outputs",
            **options,
        )
        modes = {(cmd["parallel_config"].sp_size, cmd["parallel_config"].ep_size): cmd["nproc"] for _, cmd in commands}
        if model_name == "gpt_oss":
            assert modes == {(1, 1): 4, (2, 1): 4, (2, 2): 4}
        else:
            assert modes == {(1, 1): 4, (1, 2): 4, (2, 1): 8, (2, 2): 8}
        plan = alignment_plan(caps, **options)
        assert plan.cards(caps) == max(modes.values())
        assert all(len(group) == 1 for group in plan.groups)


def test_wan_dit_uses_bfloat16_and_flash_attention():
    from tests.e2e.utils import prepare_exec_cmd
    from tests.tools.training_utils import build_torchrun_cmd

    command_list = prepare_exec_cmd(
        ["train_dit_test"],
        "wan_t2v",
        "./tests/toy_config/wan_t2v_toy",
        model_path="./wan_t2v",
        train_path="./dummy_wan_t2v",
        output_dir="./wan_t2v",
        is_moe=False,
        max_sp_size=1,
    )

    assert command_list
    for _, cmd_kwargs in command_list:
        cmd = build_torchrun_cmd(**cmd_kwargs)
        assert cmd_kwargs["extra_args"] == [
            "--model.accelerator.fsdp_config.mixed_precision.enable=True",
            "--model.accelerator.fsdp_config.mixed_precision.param_dtype=bfloat16",
            "--model.accelerator.fsdp_config.mixed_precision.cast_forward_inputs=True",
        ]
        assert "--model.ops_implementation.attn_implementation=flash_attention_2" in cmd


def test_deepseek_v4_e2e_casts_fused_triton_moe_to_bfloat16():
    from tests.e2e.utils import prepare_exec_cmd

    command_list = prepare_exec_cmd(
        ["train_text_test"],
        "deepseek_v4",
        "./tests/toy_config/deepseek_v4_toy",
        model_path="./deepseek_v4",
        train_path="./dummy_text",
        output_dir="./deepseek_v4",
        is_moe=True,
        max_ep_size=1,
    )

    assert command_list
    for _, cmd_kwargs in command_list:
        assert cmd_kwargs["extra_args"] == [
            "--model.accelerator.fsdp_config.mixed_precision.enable=True",
            "--model.accelerator.fsdp_config.mixed_precision.param_dtype=bfloat16",
            "--model.accelerator.fsdp_config.mixed_precision.cast_forward_inputs=True",
        ]


def test_gpt_oss_parallel_uses_fa4_and_quack():
    from tests.e2e.utils import prepare_exec_cmd
    from tests.tools.training_utils import build_torchrun_cmd

    command_list = prepare_exec_cmd(
        ["train_text_test"],
        "gpt_oss",
        "./tests/toy_config/gpt_oss_toy",
        model_path="./gpt_oss",
        train_path="./dummy_text",
        output_dir="./gpt_oss",
        is_moe=True,
    )

    assert [
        (cmd_kwargs["parallel_config"].sp_size, cmd_kwargs["parallel_config"].ep_size)
        for _, cmd_kwargs in command_list
    ] == [(1, 1), (2, 1), (2, 2)]
    for _, cmd_kwargs in command_list:
        cmd = build_torchrun_cmd(**cmd_kwargs)
        assert "--model.ops_implementation.attn_implementation=flash_attention_4" in cmd
        assert "--model.ops_implementation.moe_implementation=fused_quack" in cmd


def test_npu_adapter_preserves_owned_host_slice_and_resets_ranks():
    import threading

    from scripts.ci.accelerator.launch_plan import RunnerCapabilities
    from scripts.ci.accelerator.runtime.devices import child_environment
    from scripts.ci.accelerator.scheduler import DeviceJob, run_jobs

    caps = RunnerCapabilities("npu", tuple(str(i) for i in range(8, 16)))
    env = child_environment(caps, ["10", "11"], environ={"ASCEND_DEVICE_ID": "8", "RANK": "7"})
    assert env["ASCEND_VISIBLE_DEVICES"] == env["ASCEND_RT_VISIBLE_DEVICES"] == "10,11"
    assert "ASCEND_DEVICE_ID" not in env and "RANK" not in env
    assert env["HCCL_HOST_SOCKET_PORT_RANGE"] == env["HCCL_NPU_SOCKET_PORT_RANGE"] == "auto"
    cpu = child_environment(caps, [], cpu=True, environ={})
    assert cpu["CUDA_VISIBLE_DEVICES"] == cpu["ASCEND_RT_VISIBLE_DEVICES"] == ""
    assert cpu["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"

    # Exercise simultaneous 4+2+2 leases on the upper physical-host slice.
    barrier = threading.Barrier(3, timeout=5)

    def execute(job, assigned):
        env = child_environment(caps, assigned, environ={"ASCEND_DEVICE_ID": "8", "RANK": "7"})
        assert len(assigned) == job.cards
        assert "ASCEND_DEVICE_ID" not in env and "RANK" not in env
        assert env["ASCEND_VISIBLE_DEVICES"] == env["ASCEND_RT_VISIBLE_DEVICES"]
        barrier.wait()
        return set(env["ASCEND_RT_VISIBLE_DEVICES"].split(","))

    jobs = [DeviceJob("four", 4, None), DeviceJob("two-a", 2, None), DeviceJob("two-b", 2, None)]
    slices = list(run_jobs(jobs, list(caps.devices), execute).values())
    assert set.union(*slices) == set(caps.devices)
    assert sum(map(len, slices)) == len(set.union(*slices))  # No device is leased twice.


def test_case_batches_preserve_inventory_and_resource_requests():
    from scripts.ci.accelerator.batching import make_batches

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


def test_skipped_oversized_cases_preserve_plans_and_use_owned_capacity():
    from scripts.ci.accelerator.batching import make_batches
    from scripts.ci.accelerator.scheduler import DeviceJob, run_jobs

    rows = [
        dict(nodeid="skipped-four", file="test.py", cards=4, skip="requires four devices"),
        dict(nodeid="runnable-two", file="test.py", cards=2, skip=None),
    ]
    batches = make_batches([(dict(name="test"), rows)], capacity=2)
    assert len(batches) == 2  # Do not mix an oversized skip with runnable work.
    assert all(batch["cards"] == 2 for batch in batches)
    assert batches[0]["cases"][0]["cards"] == 4
    jobs = [DeviceJob(batch["name"], batch["cards"], batch) for batch in batches]
    assigned = run_jobs(jobs, ["0", "1"], lambda job, devices: devices)
    assert all(devices == ["0", "1"] for devices in assigned.values())


@pytest.mark.parametrize("ignore_term", [False, True])
def test_worker_cleanup_drains_group_after_launcher_exit(monkeypatch, tmp_path, ignore_term):
    import os
    import sys

    import psutil

    from scripts.ci.accelerator.runtime.workers import run_process

    ready = tmp_path / "child.pid"
    child_code = (
        "import os, signal, time\nfrom pathlib import Path\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + f"Path({str(ready)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    launcher = (
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "deadline = time.monotonic() + 5\n"
        f"while not Path({str(ready)!r}).exists():\n"
        "    if time.monotonic() >= deadline: raise RuntimeError('child did not start')\n"
        "    time.sleep(0.01)\n"
    )
    # Deterministically model a child born between the regular polling samples.
    monkeypatch.setattr(psutil.Process, "children", lambda self, recursive=False: [])
    try:
        with (tmp_path / "worker.log").open("w") as log:
            run_process([sys.executable, "-c", launcher], os.environ.copy(), log, 10)
        assert ready.exists()
        try:
            child = psutil.Process(int(ready.read_text()))
            assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    finally:
        if ready.exists():
            try:
                psutil.Process(int(ready.read_text())).kill()
            except psutil.Error:
                pass


def test_device_pool_drains_for_waiting_large_job():
    from scripts.ci.accelerator.scheduler import DeviceJob, run_jobs

    order = []
    jobs = [DeviceJob("first", 4, None), DeviceJob("large", 8, None), DeviceJob("backfill", 4, None)]
    run_jobs(jobs, list("01234567"), lambda job, assigned: order.append(job.name), drain_after=0)
    assert order == ["first", "large", "backfill"]


def test_device_pool_can_run_cpu_work_beside_an_exclusive_device_job():
    import threading

    from scripts.ci.accelerator.scheduler import DeviceJob, run_jobs

    barrier = threading.Barrier(2, timeout=5)
    assignments = {}

    def execute(job, assigned):
        assignments[job.name] = assigned
        barrier.wait()

    run_jobs([DeviceJob("accelerator", 8, None), DeviceJob("cpu", 0, None)], list("01234567"), execute)
    assert assignments["cpu"] == []
    assert assignments["accelerator"] == list("01234567")


@pytest.mark.parametrize("cpu", [False, True])
@pytest.mark.parametrize("use_reentrant", [False, True])
def test_cpu_checkpoint_does_not_query_registered_accelerator(monkeypatch, cpu, use_reentrant):
    from types import SimpleNamespace

    import torch
    from torch.utils.checkpoint import DefaultDeviceType, checkpoint

    from scripts.ci.accelerator import pytest_plugin
    from scripts.ci.accelerator.launch_plan import CasePlan, LaunchSpec

    # Model torch_npu's registered backend without requiring its runtime.
    backend = torch._C._get_privateuse1_backend_name()

    def reject_amp_query():
        raise RuntimeError("CPU checkpoint queried hidden accelerator AMP")

    monkeypatch.setattr(torch, backend, SimpleNamespace(get_amp_supported_dtype=reject_amp_query), raising=False)
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", backend)
    monkeypatch.delenv("CI_CASE_PROGRESS", raising=False)
    plan = CasePlan.sequential(LaunchSpec(device="cpu" if cpu else "accelerator"))
    config = SimpleNamespace(_ci_batch_cases={"sample": {"plan": plan.to_dict()}}, _ci_results={})
    item = SimpleNamespace(nodeid="sample", config=config)
    protocol = pytest_plugin.pytest_runtest_protocol(item, None)
    next(protocol)
    try:
        value = torch.randn(4, requires_grad=True)

        def run_checkpoint():
            checkpoint(lambda x: x.sin().square(), value, use_reentrant=use_reentrant).sum().backward()

        if cpu:
            assert DefaultDeviceType.get_device_type() == "cpu"
            run_checkpoint()
            torch.testing.assert_close(value.grad, 2 * value.detach().sin() * value.detach().cos())
        else:
            assert DefaultDeviceType.get_device_type() == backend
            with pytest.raises(RuntimeError, match="queried hidden accelerator AMP"):
                run_checkpoint()
    finally:
        # Cleanup also runs when setup, the test or teardown raises.
        protocol.close()
    assert DefaultDeviceType.get_device_type() == backend


@pytest.mark.parametrize("cpu", [False, True])
@pytest.mark.parametrize("foreach", [None, False, True])
@pytest.mark.parametrize("optimizer_name", ["AdamW", "SGD"])
def test_cpu_optimizer_does_not_query_hidden_accelerator(monkeypatch, cpu, foreach, optimizer_name):
    from types import SimpleNamespace

    import torch
    import torch.optim.optimizer as optimizer_module

    from scripts.ci.accelerator import pytest_plugin
    from scripts.ci.accelerator.launch_plan import CasePlan, LaunchSpec

    def vendor_device_query():
        raise RuntimeError("Optimizer queried hidden accelerator device")

    # torch_npu replaces this alias with a policy that queries its device name,
    # including when AdamW/SGD parameters are entirely CPU tensors.
    monkeypatch.setattr(optimizer_module, "_get_foreach_kernels_supported_devices", vendor_device_query)
    monkeypatch.delenv("CI_CASE_PROGRESS", raising=False)
    plan = CasePlan.sequential(LaunchSpec(device="cpu" if cpu else "accelerator"))
    item = SimpleNamespace(
        nodeid="sample", config=SimpleNamespace(_ci_batch_cases={"sample": {"plan": plan.to_dict()}}, _ci_results={})
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    constructor = getattr(torch.optim, optimizer_name)
    optimizer = constructor([parameter], lr=0.01, foreach=foreach)
    reference_optimizer = constructor([reference], lr=0.01, foreach=False)
    parameter.grad = torch.tensor([0.25, -0.5])
    reference.grad = parameter.grad.clone()
    reference_optimizer.step()
    protocol = pytest_plugin.pytest_runtest_protocol(item, None)
    next(protocol)
    try:
        if not cpu and foreach is None:
            with pytest.raises(RuntimeError, match="queried hidden accelerator device"):
                optimizer.step()
        else:
            optimizer.step()
            torch.testing.assert_close(parameter, reference)
        assert optimizer.param_groups[0]["foreach"] is foreach
    finally:
        protocol.close()
    assert optimizer_module._get_foreach_kernels_supported_devices is vendor_device_query


@pytest.mark.parametrize("cpu", [False, True])
@pytest.mark.parametrize("fail_body", [False, True])
def test_cpu_dataloader_avoids_vendor_device_queries_and_restores_patch(monkeypatch, cpu, fail_body):
    import sys
    from types import SimpleNamespace

    from torch.utils.data import DataLoader, dataloader

    from scripts.ci.accelerator import pytest_plugin

    original_init = dataloader._MultiProcessingDataLoaderIter.__init__
    device_queries = []

    def vendor_init(self, *args, **kwargs):
        # torch_npu 2.10 queries NPU affinity even for an all-CPU DataLoader.
        device_queries.append("npu_affinity")
        original_init(self, *args, **kwargs)

    vendor = SimpleNamespace(_mpdl_iter_init=vendor_init, origin_mpdl_iter_init=original_init)
    monkeypatch.setitem(sys.modules, "torch_npu.utils._module", vendor)
    monkeypatch.setattr(dataloader._MultiProcessingDataLoaderIter, "__init__", vendor_init)
    monkeypatch.delenv("CI_CASE_PROGRESS", raising=False)
    plan = CasePlan.sequential(LaunchSpec(device="cpu" if cpu else "accelerator"))
    item = SimpleNamespace(
        nodeid="sample", config=SimpleNamespace(_ci_batch_cases={"sample": {"plan": plan.to_dict()}}, _ci_results={})
    )
    protocol = pytest_plugin.pytest_runtest_protocol(item, None)
    next(protocol)
    try:
        assert list(DataLoader(list(range(8)), batch_size=None, num_workers=2)) == list(range(8))
        assert device_queries == ([] if cpu else ["npu_affinity"])
    finally:
        if fail_body:
            with pytest.raises(RuntimeError, match="test body failed"):
                protocol.throw(RuntimeError("test body failed"))
        else:
            protocol.close()
        assert dataloader._MultiProcessingDataLoaderIter.__init__ is vendor_init


@pytest.mark.parametrize("cpu", [False, True])
@pytest.mark.parametrize("previously_disabled", [False, True])
@pytest.mark.parametrize("fail_body", [False, True])
def test_cpu_inductor_avoids_npu_registration_and_restores_policy(monkeypatch, cpu, previously_disabled, fail_body):
    import sys
    from types import SimpleNamespace

    import torch
    from torch._dynamo.exc import BackendCompilerFailed
    from torch._dynamo.utils import counters

    from scripts.ci.accelerator import pytest_plugin

    class VendorRegistry:
        _disabled_register = previously_disabled

        @classmethod
        def disable_register(cls):
            cls._disabled_register = True

        @classmethod
        def register_inductor_npu(cls):
            if not cls._disabled_register:
                raise RuntimeError("Inductor queried hidden NPU")

    original_call = torch._TorchCompileInductorWrapper.__call__

    def vendor_call(self, model, inputs):
        VendorRegistry.register_inductor_npu()
        return original_call(self, model, inputs)

    monkeypatch.setitem(sys.modules, "torch_npu.utils._dynamo", SimpleNamespace(_InductorNpuRegistry=VendorRegistry))
    monkeypatch.setattr(torch._TorchCompileInductorWrapper, "__call__", vendor_call)
    monkeypatch.delenv("CI_CASE_PROGRESS", raising=False)
    plan = CasePlan.sequential(LaunchSpec(device="cpu" if cpu else "accelerator"))
    item = SimpleNamespace(
        nodeid="sample", config=SimpleNamespace(_ci_batch_cases={"sample": {"plan": plan.to_dict()}}, _ci_results={})
    )
    torch.compiler.reset()
    protocol = pytest_plugin.pytest_runtest_protocol(item, None)
    next(protocol)
    try:
        value = torch.arange(8, dtype=torch.float32, requires_grad=True)
        compiled = torch.compile(lambda x: x.square() + 1, fullgraph=True)
        if cpu or previously_disabled:
            graphs = counters["stats"]["unique_graphs"]
            torch.testing.assert_close(compiled(value), value.square() + 1)
            compiled(value).sum().backward()
            torch.testing.assert_close(value.grad, 2 * value.detach())
            assert counters["stats"]["unique_graphs"] > graphs
        else:
            with pytest.raises(BackendCompilerFailed, match="Inductor queried hidden NPU"):
                compiled(value)
    finally:
        try:
            if fail_body:
                with pytest.raises(RuntimeError, match="test body failed"):
                    protocol.throw(RuntimeError("test body failed"))
            else:
                protocol.close()
            assert VendorRegistry._disabled_register is previously_disabled
        finally:
            torch.compiler.reset()


@pytest.mark.parametrize("cpu", [False, True])
@pytest.mark.parametrize("existing_cpu", [False, True])
def test_cpu_inductor_preserves_vendor_and_existing_cpu_overrides(monkeypatch, cpu, existing_cpu):
    from types import SimpleNamespace

    import torch
    from torch._inductor.codegen import common
    from torch._inductor.codegen.cpu_device_op_overrides import CpuDeviceOpOverrides

    from scripts.ci.accelerator import pytest_plugin

    npu_override = object()
    cpu_override = CpuDeviceOpOverrides()
    overrides = {"npu": npu_override}
    if existing_cpu:
        overrides["cpu"] = cpu_override
    # torch_npu populates this registry at import. Torch 2.10 only imports its
    # CPU entry when the whole registry is empty; Torch 2.11 repairs missing keys.
    monkeypatch.setattr(common, "device_op_overrides_dict", overrides)
    monkeypatch.delenv("CI_CASE_PROGRESS", raising=False)
    plan = CasePlan.sequential(LaunchSpec(device="cpu" if cpu else "accelerator"))
    item = SimpleNamespace(
        nodeid="sample", config=SimpleNamespace(_ci_batch_cases={"sample": {"plan": plan.to_dict()}}, _ci_results={})
    )
    protocol = pytest_plugin.pytest_runtest_protocol(item, None)
    next(protocol)
    try:
        assert overrides["npu"] is npu_override
        if existing_cpu:
            assert overrides["cpu"] is cpu_override
        elif cpu:
            assert isinstance(overrides.get("cpu"), CpuDeviceOpOverrides)
        else:
            assert "cpu" not in overrides
        if cpu:
            torch.compiler.reset()
            value = torch.arange(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(lambda x: x + 1, fullgraph=True)(value), value + 1)
    finally:
        protocol.close()
        torch.compiler.reset()
    assert overrides["npu"] is npu_override


@pytest.mark.parametrize(
    "failure", ["foreach_lookup", "cpu_codegen", "dataloader_original", "registry_policy", "registry_disable"]
)
def test_cpu_isolation_restores_state_when_setup_fails(monkeypatch, failure):
    import sys
    from types import SimpleNamespace

    import torch.optim.optimizer as optimizer_module
    from torch._inductor.codegen import common
    from torch.utils.checkpoint import DefaultDeviceType
    from torch.utils.data import dataloader

    from scripts.ci.accelerator import pytest_plugin

    original_device = DefaultDeviceType.get_device_type()
    original_foreach = optimizer_module._get_foreach_kernels_supported_devices
    original_init = dataloader._MultiProcessingDataLoaderIter.__init__

    def vendor_init(*args, **kwargs):
        raise AssertionError("hidden accelerator queried")

    class VendorRegistry:
        _disabled_register = False

        @classmethod
        def disable_register(cls):
            cls._disabled_register = True
            if failure == "registry_disable":
                raise RuntimeError("vendor setup failed")

    vendor = SimpleNamespace(_mpdl_iter_init=vendor_init, origin_mpdl_iter_init=original_init)
    monkeypatch.setitem(sys.modules, "torch_npu.utils._module", vendor)
    monkeypatch.setitem(sys.modules, "torch_npu.utils._dynamo", SimpleNamespace(_InductorNpuRegistry=VendorRegistry))
    monkeypatch.setattr(dataloader._MultiProcessingDataLoaderIter, "__init__", vendor_init)
    monkeypatch.delenv("CI_CASE_PROGRESS", raising=False)
    if failure == "foreach_lookup":
        monkeypatch.delattr(optimizer_module, "_get_foreach_kernels_supported_devices")
    elif failure == "cpu_codegen":
        monkeypatch.setattr(common, "device_op_overrides_dict", {})

        def fail_codegen(*args):
            raise RuntimeError("CPU codegen setup failed")

        monkeypatch.setattr(common, "register_device_op_overrides", fail_codegen)
    elif failure == "dataloader_original":
        del vendor.origin_mpdl_iter_init
    elif failure == "registry_policy":
        del VendorRegistry._disabled_register

    plan = CasePlan.sequential(LaunchSpec(device="cpu"))
    item = SimpleNamespace(
        nodeid="sample", config=SimpleNamespace(_ci_batch_cases={"sample": {"plan": plan.to_dict()}}, _ci_results={})
    )
    protocol = pytest_plugin.pytest_runtest_protocol(item, None)
    expected_error = AttributeError if failure == "foreach_lookup" else RuntimeError
    with pytest.raises(expected_error):
        next(protocol)
    assert DefaultDeviceType.get_device_type() == original_device
    if failure == "foreach_lookup":
        assert not hasattr(optimizer_module, "_get_foreach_kernels_supported_devices")
    else:
        assert optimizer_module._get_foreach_kernels_supported_devices is original_foreach
    assert dataloader._MultiProcessingDataLoaderIter.__init__ is vendor_init
    if failure == "registry_policy":
        assert not hasattr(VendorRegistry, "_disabled_register")
    else:
        assert VendorRegistry._disabled_register is False


def test_case_workspaces_isolate_relative_checkpoints_and_caches(tmp_path):
    from scripts.ci.accelerator.runtime.workers import create_workspace

    first, second = tmp_path / "first", tmp_path / "second"
    create_workspace(first)
    create_workspace(second)
    (first / "cache" / "dataset").write_text("first")
    assert not (second / "cache" / "dataset").exists()
    assert (first / "tests").resolve() == (second / "tests").resolve()


def test_rendezvous_allocator_does_not_reuse_a_pending_launch_port(monkeypatch, tmp_path):
    from scripts.ci.accelerator.runtime import ports

    candidates = iter([12345, 12345, 12346])
    monkeypatch.setattr(ports, "_free_port", lambda: next(candidates))
    registry = str(tmp_path / "ports.json")
    assert ports.find_free_port(registry) == 12345
    assert ports.find_free_port(registry) == 12346


def test_live_rendezvous_reserves_the_socket_and_closes_on_error(monkeypatch, tmp_path):
    import errno
    import json
    import os
    import socket
    from datetime import timedelta

    import psutil

    from scripts.ci.accelerator.runtime import ports

    def reject_probe():
        pytest.fail("A live rendezvous must not probe and close a free port")

    monkeypatch.setattr(ports, "_free_port", reject_probe)
    events = tmp_path / "rendezvous.jsonl"
    monkeypatch.setenv("CI_RENDEZVOUS_EVENTS", str(events))
    with pytest.raises(RuntimeError, match="caller failed"):
        with ports.rendezvous_endpoint(timeout=timedelta(seconds=10)) as endpoint:
            with socket.socket() as competitor:
                with pytest.raises(OSError) as error:
                    competitor.bind((endpoint.host, endpoint.port))
                assert error.value.errno == errno.EADDRINUSE
            client = endpoint.connect()
            client.set("key", "value")
            assert client.get("key") == b"value"
            del client
            raise RuntimeError("caller failed")

    assert not any(
        connection.status == psutil.CONN_LISTEN and connection.laddr.port == endpoint.port
        for connection in psutil.Process().net_connections(kind="tcp")
    )
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert [row["action"] for row in rows] == ["ready", "closed"]
    assert {row["pid"] for row in rows} == {os.getpid()}
    assert len({row["launch_id"] for row in rows}) == 1


def _rendezvous_sum_worker(rank, world_size, output, base, endpoint):
    import json
    import os
    from datetime import timedelta
    from pathlib import Path

    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=20)
    )
    try:
        value = torch.tensor(base + rank)
        dist.all_reduce(value)
        assert value.item() == world_size * base + world_size * (world_size - 1) // 2
        Path(output, f"rank-{rank}.json").write_text(json.dumps(dict(port=endpoint.port, pid=os.getpid())))
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(
    provider=CasePlan.concurrent(LaunchSpec(nproc=2, device="cpu"), LaunchSpec(nproc=2, device="cpu"))
)
def test_concurrent_spawn_rendezvous_are_isolated_and_join_before_release(monkeypatch, tmp_path):
    import json
    import socket
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import psutil

    from tests.tools import launch_utils

    original_spawn = launch_utils.mp.spawn
    ready = threading.Barrier(2, timeout=30)

    def spawn(func, args, **kwargs):
        endpoint = args[-1]
        with socket.socket() as competitor:
            with pytest.raises(OSError):
                competitor.bind((endpoint.host, endpoint.port))
        ready.wait()
        return original_spawn(func, args=args, **kwargs)

    monkeypatch.setattr(launch_utils.mp, "spawn", spawn)

    def run(index):
        output = tmp_path / str(index)
        output.mkdir()
        launch_utils.spawn_with_rendezvous(_rendezvous_sum_worker, args=(2, str(output), 10 * index), nprocs=2)
        return [json.loads(path.read_text()) for path in sorted(output.glob("rank-*.json"))]

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(run, (1, 2)))
    assert len(first) == len(second) == 2
    assert {row["port"] for row in first}.isdisjoint(row["port"] for row in second)
    assert all(not psutil.pid_exists(row["pid"]) for row in first + second)


def _common_torchrun_sum(output):
    import json
    from pathlib import Path

    import torch
    import torch.distributed as dist

    assert dist.get_backend() == "gloo"
    value = torch.tensor(dist.get_rank() + 1)
    dist.all_reduce(value)
    assert value.item() == 3
    Path(output, str(dist.get_rank())).write_text(json.dumps(dict(backend=dist.get_backend(), value=value.item())))


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_common_spawn_launcher_uses_live_store_instead_of_free_port(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from scripts.ci.accelerator.launch_plan import RunnerCapabilities
    from scripts.ci.accelerator.runtime.devices import child_environment

    env = child_environment(RunnerCapabilities("cpu"), [])
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import sys\nfrom unittest.mock import patch\n"
        "from tests.ci.test_accelerator import _common_torchrun_sum\n"
        "from tests.tools import launch_utils\n"
        "with patch.object(launch_utils, 'find_free_port', side_effect=AssertionError('probe-close rendezvous was used')):\n"
        "    launch_utils.torchrun(_common_torchrun_sum, 2, sys.argv[1])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], env=env, capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == ["0", "1"]


def _failing_rendezvous_worker(rank, endpoint):
    from datetime import timedelta

    store = endpoint.connect()
    store.set(str(rank), "ready")
    if rank == 0:
        store.wait(["1"], timedelta(seconds=20))
        raise RuntimeError("injected worker failure")
    store.wait(["never-published"], timedelta(seconds=30))


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_spawn_worker_failure_reaps_peers_and_closes_store(monkeypatch, tmp_path):
    import json

    import psutil
    import torch.multiprocessing as mp

    from tests.tools import launch_utils

    original_spawn = mp.spawn
    pids = []

    def spawn(*args, **kwargs):
        context = original_spawn(*args, **kwargs)
        pids.extend(process.pid for process in context.processes)
        return context

    events = tmp_path / "rendezvous.jsonl"
    monkeypatch.setenv("CI_RENDEZVOUS_EVENTS", str(events))
    monkeypatch.setattr(mp, "spawn", spawn)
    with pytest.raises(mp.ProcessRaisedException, match="injected worker failure"):
        launch_utils.spawn_with_rendezvous(_failing_rendezvous_worker, nprocs=2)
    assert len(pids) == 2 and all(not psutil.pid_exists(pid) for pid in pids)
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert [row["action"] for row in rows] == ["ready", "closed"]
    assert not any(
        connection.status == psutil.CONN_LISTEN and connection.laddr.port == rows[0]["port"]
        for connection in psutil.Process().net_connections(kind="tcp")
    )


def test_spawn_parent_interruption_reaps_workers_before_closing_store(monkeypatch):
    import psutil

    from tests.tools import launch_utils

    class Worker:
        alive = True
        joined = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            pass  # An unresponsive worker still needs the bounded kill path.

        def kill(self):
            self.alive = False

        def join(self, timeout):
            self.joined = True

    workers = [Worker(), Worker()]
    endpoint = None

    class Context:
        processes = workers

        def join(self):
            raise KeyboardInterrupt

    def spawn(func, args, **kwargs):
        nonlocal endpoint
        endpoint = args[-1]
        return Context()

    monkeypatch.setattr(launch_utils.mp, "spawn", spawn)
    with pytest.raises(KeyboardInterrupt):
        launch_utils.spawn_with_rendezvous(None, nprocs=2)
    assert all(not worker.alive and worker.joined for worker in workers)
    assert not any(
        connection.status == psutil.CONN_LISTEN and connection.laddr.port == endpoint.port
        for connection in psutil.Process().net_connections(kind="tcp")
    )


@pytest.mark.launch_plan(
    provider=CasePlan.concurrent(LaunchSpec(nproc=2, device="cpu"), LaunchSpec(nproc=2, device="cpu"))
)
def test_concurrent_standalone_commands_keep_agent_store_alive(monkeypatch, tmp_path):
    import json
    import os
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from tests.tools.training_utils import build_torchrun_cmd

    monkeypatch.setenv("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "0")
    script = tmp_path / "worker.py"
    script.write_text(
        "import json,os\nfrom pathlib import Path\n"
        "import torch\nimport torch.distributed as dist\n"
        "dist.init_process_group('gloo')\n"
        "assert os.environ['TORCHELASTIC_USE_AGENT_STORE'] == 'True'\n"
        "value=torch.tensor(dist.get_rank()+1)\ndist.all_reduce(value)\nassert value.item()==3\n"
        "Path(os.environ['OUTPUT'],str(dist.get_rank())).write_text(json.dumps(dict(port=os.environ['MASTER_PORT'])))\n"
        "dist.destroy_process_group()\n"
    )

    def run(index):
        output = tmp_path / str(index)
        output.mkdir()
        command = build_torchrun_cmd(str(script), "unused", "unused", "unused", str(output), nproc=2)
        result = subprocess.run(
            [sys.executable, "-m", "torch.distributed.run", *command[1:]],
            env={**os.environ, "OUTPUT": str(output)},
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return [json.loads(path.read_text())["port"] for path in sorted(output.iterdir())]

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(run, (1, 2)))
    assert len(first) == len(second) == 2
    assert len(set(first)) == len(set(second)) == 1
    assert set(first).isdisjoint(second)


def test_unittest_plan_reads_world_size_without_running_constructor(monkeypatch):
    from types import SimpleNamespace

    from scripts.ci.accelerator import pytest_plugin
    from scripts.ci.accelerator.launch_plan import LaunchSpec, RunnerCapabilities

    class DistributedCase:
        def __init__(self):
            raise AssertionError("Collection must not construct a distributed test")

        @property
        def world_size(self):
            return 4

        def ci_launch_plan(self):
            return LaunchSpec(nproc=self.world_size)

    monkeypatch.setattr(pytest_plugin, "runner_capabilities", lambda: RunnerCapabilities("cuda", tuple("01234567")))
    item = SimpleNamespace(instance=None, cls=DistributedCase, get_closest_marker=lambda _: None)
    assert pytest_plugin.resolve_plan(item).launches[0].nproc == 4


@pytest.mark.parametrize("stage", ["scheduling", "collection", "shutdown"])
def test_interrupt_drains_owned_workers_before_executor_shutdown(stage, tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    import psutil

    pid_path = tmp_path / "owned.pid"
    driver = tmp_path / "driver.py"
    worker_code = (
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    shutdown_ready = tmp_path / "shutdown.ready"
    shutdown_driver = (
        "import time\nfrom pathlib import Path\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "original_shutdown = ThreadPoolExecutor.shutdown\n"
        "def shutdown(self, *args, **kwargs):\n"
        f"    Path({str(shutdown_ready)!r}).touch()\n"
        "    return original_shutdown(self, *args, **kwargs)\n"
        "ThreadPoolExecutor.shutdown = shutdown\n"
        "def execute(job, assigned):\n"
        "    if job.name == 'owned': return launch()\n"
        f"    while not Path({str(pid_path)!r}).exists(): time.sleep(0.01)\n"
        "    raise RuntimeError('another callback failed')\n"
        "run_jobs([DeviceJob('owned', 0, None), DeviceJob('bad', 0, None)], [], execute)\n"
    )
    driver.write_text(
        "import os, sys\n"
        "from scripts.ci.accelerator.runtime.workers import run_process\n"
        "from scripts.ci.accelerator.scheduler import DeviceJob, run_jobs\n"
        "def launch(*args):\n"
        "    with open(sys.argv[2], 'w') as log:\n"
        f"        run_process([sys.executable, '-c', {worker_code!r}], os.environ.copy(), log, 60)\n"
        + (
            "run_jobs([DeviceJob('owned', 0, None)], [], launch)\n"
            if stage == "scheduling"
            else shutdown_driver
            if stage == "shutdown"
            else "from scripts.ci.accelerator.runtime.workers import interruptible_executor\n"
            "with interruptible_executor(max_workers=1) as (executor, scope):\n"
            "    list(executor.map(lambda value: scope.run(launch), [None]))\n"
        )
    )
    with (tmp_path / "driver.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(driver), str(pid_path), str(tmp_path / "worker.log")],
            env={**os.environ, "PYTHONPATH": os.getcwd()},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            ready = shutdown_ready if stage == "shutdown" else pid_path
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert ready.exists(), (tmp_path / "driver.log").read_text()
            owned = psutil.Process(int(pid_path.read_text()))
            process.send_signal(signal.SIGINT)
            process.wait(timeout=12)
            assert process.returncode != 0
            assert not owned.is_running() or owned.status() == psutil.STATUS_ZOMBIE
            assert unrelated.poll() is None
        finally:
            if pid_path.exists():
                try:
                    os.killpg(int(pid_path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            unrelated.terminate()
            unrelated.wait(timeout=5)


@pytest.mark.parametrize(
    "failure",
    ["missing_results", "interrupt", "keep_going_failure", "cleanup_failure", "child_failure"],
)
def test_runner_failure_artifacts_match_execution(monkeypatch, tmp_path, capsys, failure):
    import json
    import sys
    from pathlib import Path

    from scripts.ci.accelerator import runner
    from scripts.ci.accelerator.runtime.workers import ProcessCleanupError

    root = tmp_path / "checkout"
    (root / "tests").mkdir(parents=True)
    source = root / "pyproject.toml"
    source.write_text("original\n")
    (root / "uv.lock").write_text("lock\n")
    test_file = root / "tests/test_sample.py"
    test_file.write_text("def test_one(): pass\n")
    output = tmp_path / "results"
    nodeid = "tests/test_sample.py::test_one"

    def run_process(command, env, log, timeout, **kwargs):
        if "scripts.ci.accelerator.runtime.devices" in command:
            Path(command[-2]).write_text(json.dumps(dict(backend="cpu", devices=[])))
        elif "--collect-only" in command:
            inventory = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--ci-inventory="))
            rows = [
                dict(
                    nodeid=nodeid,
                    file=str(test_file),
                    cards=0,
                    skip=None,
                    plan=dict(groups=[[dict(nproc=1, device="cpu", nnodes=1)]]),
                )
            ]
            if failure == "cleanup_failure":
                pending_file = root / "tests/test_pending.py"
                pending_file.write_text("def test_pending(): pass\n")
                rows.append({**rows[0], "nodeid": "tests/test_pending.py::test_pending", "file": str(pending_file)})
            Path(inventory).write_text(json.dumps(rows))
        elif failure == "interrupt":
            raise KeyboardInterrupt()
        elif failure == "cleanup_failure":
            raise ProcessCleanupError("owned worker survived cleanup")
        elif failure == "child_failure":
            import subprocess

            try:
                subprocess.run(
                    [sys.executable, "-c", "raise RuntimeError('original training failure')"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except subprocess.CalledProcessError:
                # A long pytest/subprocess report must not bury the rank error.
                log.write("subprocess wrapper frame\n" * 100)
                log.flush()
                raise
        elif failure == "keep_going_failure":
            Path(env["CI_CASE_PROGRESS"]).write_text(
                json.dumps(dict(kind="failure", nodeid=nodeid, outcome="failed", detail="original assertion")) + "\n"
            )
            kwargs["on_poll"]()
            import subprocess

            exc = subprocess.CalledProcessError(1, command)
            kwargs["on_failure"](exc)
            raise exc

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "run_process", run_process)
    if failure == "child_failure":
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
    manifest = json.loads((Path(runner.__file__).parent / "suites/gpu.json").read_text())
    ci_selection = next(row["name"] for row in manifest if any(arg.rstrip("/") == "tests/ci" for arg in row["args"]))
    monkeypatch.setattr(
        sys,
        "argv",
        ["ci", "--suite", "unit", "--backend", "cpu", "--only", ci_selection, "--output-dir", str(output)],
    )
    if failure in ("keep_going_failure", "cleanup_failure"):
        sys.argv.append("--keep-going")
    if failure == "cleanup_failure":
        sys.argv.append("--serial")
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            runner.main()
    else:
        assert runner.main() == 1
    summary = json.loads((output / "summary.json").read_text())
    record = next(iter(summary["commands"].values()))
    assert record["returncode"] != 0
    if failure == "missing_results":
        assert "ResourcePlanError" in record["error"] and ".results.json" in record["error"]
    elif failure == "keep_going_failure":
        assert nodeid in summary["error"] and record["status"] == "failed"
    elif failure == "cleanup_failure":
        assert len(summary["commands"]) == 1  # Never release the lease to the pending batch.
        assert "ProcessCleanupError" in record["error"] and "survived cleanup" in summary["error"]
    elif failure == "child_failure":
        foreground = capsys.readouterr().out
        assert "RuntimeError: original training failure" in foreground
        assert foreground.count("::group::") == foreground.count("::endgroup::") == 1
    else:
        assert "KeyboardInterrupt" in summary["error"]


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_first_failure_cancels_running_process_and_stops_dispatch(tmp_path, cleanup_failure):
    import os
    import sys
    import time

    import psutil

    from scripts.ci.accelerator.runtime.workers import ProcessCleanupError, run_process
    from scripts.ci.accelerator.scheduler import DeviceJob, run_jobs

    ready = tmp_path / "worker.pid"
    called = []
    peer_failures = []

    def execute(job, assigned):
        called.append(job.name)
        if job.name == "bad":
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists()
            error_type = ProcessCleanupError if cleanup_failure else RuntimeError
            raise error_type("first failure")
        assert job.name == "running"
        code = f"import os,time; from pathlib import Path; Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
        with (tmp_path / "worker.log").open("w") as log:
            run_process([sys.executable, "-c", code], os.environ.copy(), log, 5, on_failure=peer_failures.append)

    try:
        with pytest.raises(RuntimeError, match="first failure"):
            run_jobs(
                [DeviceJob(name, 0, None) for name in ("running", "bad", "pending")],
                [],
                execute,
                cancel_on_error=not cleanup_failure,
            )
        assert sorted(called) == ["bad", "running"]
        assert not peer_failures  # The peer was cancelled, rather than reaching its timeout.
        worker = psutil.Process(int(ready.read_text()))
        assert not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    finally:
        if ready.exists():
            try:
                os.kill(int(ready.read_text()), 9)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("keep_going", [False, True])
@pytest.mark.parametrize("suite,assigned,devices", [("unit", [], "cpu"), ("e2e", ["0", "1"], "0,1")])
def test_case_progress_failure_is_visible_and_can_cancel_peers(tmp_path, capsys, keep_going, suite, assigned, devices):
    import json
    from concurrent.futures import CancelledError

    from scripts.ci.accelerator.reporting import BatchProgress, CaseFailure, RunReporter

    selection = dict(name="09-unit" if suite == "unit" else "npu-e2e-07")
    rows = [dict(nodeid="test_a.py::test_bad"), dict(nodeid="test_a.py::test_peer")]
    batches = [dict(name=name, selection=selection, cases=[row]) for name, row in zip(("failed-batch", "peer"), rows)]
    reporter = RunReporter(
        2,
        2,
        suite=suite,
        inventories=[(selection, rows)],
        batches=list(reversed(batches)),
        keep_going=keep_going,
        heartbeat_seconds=0,
    )
    prefix = "unit-09" if suite == "unit" else "e2e-07"
    path = tmp_path / "progress.jsonl"
    failed = BatchProgress(
        reporter, "failed-batch", assigned, path, tmp_path / "failed.log", batch_file="test_a.py", case_count=1
    )
    peer = BatchProgress(reporter, "peer", assigned, tmp_path / "peer.jsonl", tmp_path / "peer.log")
    peer.path.write_text(json.dumps(dict(kind="start", nodeid=rows[1]["nodeid"])) + "\n")
    peer.poll(check_abort=False)
    event = dict(kind="failure", nodeid="test_a.py::test_bad", outcome="failed", detail="assert 1 == 2")
    failed.start()
    encoded = json.dumps(event)
    path.write_text(encoded[:15])
    failed.poll()
    with path.open("a") as handle:
        handle.write(encoded[15:] + "\n")
    if keep_going:
        failed.poll()
        peer.poll()
    else:
        with pytest.raises(CaseFailure, match="test_bad"):
            failed.poll()
        with pytest.raises(CancelledError, match="test_bad"):
            peer.poll()
    with path.open("a") as handle:
        handle.write(json.dumps(dict(event, kind="result")) + "\n")
    failed.poll(check_abort=False)
    output = capsys.readouterr().out
    assert f"[{prefix}][START][devices={devices}] test_a.py (cases=1)" in output
    assert f"[{prefix}-case-0001][FAILED (1/2)][devices={devices}] test_a.py::test_bad" in output
    assert "    assert 1 == 2" in output
    assert output.count("[FAILED (1/2)]") == 1 and f"[{prefix}-case-0002][RUNNING][devices={devices}]" in output
    peer.cancelled("another case failed")
    failed.done(dict(wall_seconds=1.2, returncode=1))
    output = capsys.readouterr().out
    assert f"[{prefix}-case-0002][CANCELLED][devices={devices}]" in output
    assert f"[{prefix}][DONE][devices={devices}] test_a.py (cases=1, elapsed=1.2s, rc=1)" in output
    assert failed.partial_results()["cases"][0]["outcome"] == "failed"
    reporter.summary({"peer": dict(status="cancelled")})
    assert "'cancelled': 1" in capsys.readouterr().out


def test_pytest_failure_reports_before_slow_teardown_and_keeps_partial_outcome(tmp_path, capsys):
    import os
    import sys
    import time

    from scripts.ci.accelerator.reporting import BatchProgress, CaseFailure, RunReporter
    from scripts.ci.accelerator.runtime.workers import run_process

    test_file = tmp_path / "test_fails.py"
    test_file.write_text(
        "import pytest,time\n"
        "@pytest.fixture\n"
        "def slow_cleanup():\n"
        "    yield\n"
        "    time.sleep(60)\n"
        "def test_bad(slow_cleanup):\n"
        "    print('useful child diagnostic', flush=True)\n"
        "    assert 1 == 2\n"
    )
    progress_file = tmp_path / "progress.jsonl"
    log_path = tmp_path / "worker.log"
    progress = BatchProgress(RunReporter(1, 1), "failed", [], progress_file, log_path)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        "-p",
        "scripts.ci.accelerator.pytest_plugin",
        str(test_file),
    ]
    env = {**os.environ, "PYTHONPATH": os.getcwd(), "CI_CASE_PROGRESS": str(progress_file), "PYTHONUNBUFFERED": "1"}
    for key in ("VEOMNI_CI_ACTIVE_PLAN", "CI_CASE_RESULTS"):
        env.pop(key, None)
    started = time.monotonic()
    with log_path.open("w") as log:
        with pytest.raises(CaseFailure, match="test_bad"):
            run_process(command, env, log, 20, on_poll=progress.poll)
    assert time.monotonic() - started < 20
    assert "assert 1 == 2" in capsys.readouterr().out
    assert "useful child diagnostic" in log_path.read_text()
    assert progress.partial_results()["cases"][0]["outcome"] == "failed"


def test_reported_failure_stops_admission_before_failed_batch_finishes():
    import threading

    from scripts.ci.accelerator.reporting import RunReporter
    from scripts.ci.accelerator.scheduler import DeviceJob, run_jobs

    reporter = RunReporter(3, 3)
    peer_started = threading.Event()
    release_cleanup = threading.Event()
    called = []

    def execute(job, assigned):
        called.append(job.name)
        if job.name == "bad":
            assert peer_started.wait(5)
            reporter.fail("bad", "test_failed")
            assert release_cleanup.wait(5)
            raise RuntimeError("original failure")
        if job.name == "short":
            peer_started.set()
            assert reporter.stop_event.wait(5)
            threading.Timer(0.5, release_cleanup.set).start()
            return
        pytest.fail("A queued batch started after the failed phase was reported")

    from concurrent.futures import CancelledError

    with pytest.raises((CancelledError, RuntimeError)):
        run_jobs(
            [DeviceJob(name, 0, None) for name in ("bad", "short", "pending")],
            [],
            execute,
            cancel_on_error=True,
            stop_event=reporter.stop_event,
        )
    assert set(called) == {"bad", "short"}
    assert reporter.first_failure == "bad: test_failed"


def test_worker_exit_notifies_failure_before_descendant_cleanup(tmp_path):
    import os
    import subprocess
    import sys

    import psutil

    from scripts.ci.accelerator.runtime.workers import run_process

    ready = tmp_path / "child.pid"
    child_code = (
        f"import os,time; from pathlib import Path; Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    launcher = (
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
        f"while not Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
        "sys.exit(3)\n"
    )
    observed = []

    def failure(exc):
        child = psutil.Process(int(ready.read_text()))
        observed.append((exc.returncode, child.is_running() and child.status() != psutil.STATUS_ZOMBIE))

    try:
        with (tmp_path / "worker.log").open("w") as log:
            with pytest.raises(subprocess.CalledProcessError):
                run_process([sys.executable, "-c", launcher], os.environ.copy(), log, 5, on_failure=failure)
        assert observed == [(3, True)]
        try:
            child = psutil.Process(int(ready.read_text()))
            assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    finally:
        if ready.exists():
            try:
                os.kill(int(ready.read_text()), 9)
            except ProcessLookupError:
                pass
