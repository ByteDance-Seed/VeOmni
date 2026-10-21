"""Collect launch plans and validate distributed launches inside pytest workers."""

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import pytest

from scripts.ci.accelerator.launch_plan import (
    CasePlan,
    PlanContext,
    ResourcePlanError,
    as_case_plan,
    default_plan,
    runner_capabilities,
)


def resolve_plan(item):
    instance = item.instance
    cls = getattr(item, "cls", None)
    if instance is None and cls is not None and hasattr(cls, "ci_launch_plan"):
        # Pytest 7 exposes unittest methods unbound until setup. Plan resolution
        # reads pure class properties without constructing or setting up a test.
        instance = object.__new__(cls)
    context = PlanContext(getattr(getattr(item, "callspec", None), "params", {}), runner_capabilities(), instance)
    marker = item.get_closest_marker("launch_plan")
    if marker:
        value = marker.kwargs["provider"]
        plan = as_case_plan(value(context) if callable(value) else value)
    elif hasattr(instance, "ci_launch_plan"):
        plan = as_case_plan(instance.ci_launch_plan())
    else:
        plan = default_plan(context.capabilities)
    if not isinstance(plan, CasePlan):
        raise ResourcePlanError(f"Invalid launch plan for {item.nodeid}: {plan!r}")
    return plan


def validate_launch(nproc, *, command=None, env=None, nnodes=1, record=True):
    """Reject underdeclared launches before a skip, fallback, or worker start."""
    encoded = os.environ.get("VEOMNI_CI_ACTIVE_PLAN")
    if not encoded:
        return
    plan = CasePlan.from_dict(json.loads(encoded))
    if nnodes != 1 or nproc not in {spec.nproc for spec in plan.launches}:
        raise ResourcePlanError(f"Launch nproc={nproc}, nnodes={nnodes} disagrees with {plan.to_dict()}")
    env = os.environ if env is None else env
    caps = runner_capabilities()
    encoded_lease = os.environ.get("VEOMNI_CI_LEASE")
    if encoded_lease is None:
        raise ResourcePlanError("VEOMNI_CI_ACTIVE_PLAN is set without VEOMNI_CI_LEASE")
    lease = json.loads(encoded_lease)
    specs = [spec for spec in plan.launches if spec.nproc == nproc]
    key = "CUDA_VISIBLE_DEVICES" if caps.backend == "cuda" else "ASCEND_RT_VISIBLE_DEVICES"
    visible = [value for value in env.get(key, "").split(",") if value]
    cpu = not visible and any(spec.device == "cpu" for spec in specs)
    if not cpu and not any(spec.device == "accelerator" for spec in specs):
        raise ResourcePlanError("A CPU launch must not expose accelerator devices")
    if not cpu and (len(visible) < nproc or not set(visible) <= set(lease)):
        raise ResourcePlanError(f"nproc={nproc} cannot run on lease {lease}, child visibility {visible}")
    event = dict(
        nodeid=os.environ.get("VEOMNI_CI_ACTIVE_NODEID"),
        nproc=nproc,
        device="cpu" if cpu else caps.backend,
        backend="gloo" if cpu else {"cuda": "nccl", "npu": "hccl"}.get(caps.backend),
        assigned=visible,
        command=command,
        timestamp=time.time(),
    )
    output = os.environ.get("CI_LAUNCH_EVENTS")
    if output and record:
        with open(output, "a") as handle:
            handle.write(json.dumps(event) + "\n")


def _validate_command(command, env):
    if isinstance(command, str):
        args = shlex.split(command.replace("\\\n", " "))
    elif isinstance(command, (list, tuple)):
        args = [str(value) for value in command]
    else:
        return
    if not any(Path(arg).name == "torchrun" or arg == "torch.distributed.run" for arg in args):
        return
    options = {}
    for index, arg in enumerate(args):
        if not arg.startswith("--"):
            continue
        key, separator, value = arg.partition("=")
        options[key.replace("-", "_")] = value if separator else (args[index + 1] if index + 1 < len(args) else "")
    try:
        nproc = int(options["__nproc_per_node"])
        nnodes = int(options.get("__nnodes", "1"))
    except (KeyError, ValueError) as exc:
        raise ResourcePlanError("CI torchrun launches need explicit integer nproc and nnodes") from exc
    validate_launch(nproc, nnodes=nnodes, command=args, env=env)


def pytest_addoption(parser):
    group = parser.getgroup("launch plans")
    group.addoption("--ci-inventory", type=Path)
    group.addoption("--ci-batch", type=Path)


def pytest_configure(config):
    global _active_config
    _active_config = config
    config._ci_results = {}
    config.addinivalue_line("markers", "launch_plan(provider): shared launch configuration for this case")
    config._ci_original_popen = subprocess.Popen

    def popen(command, *args, **kwargs):
        _validate_command(command, kwargs.get("env", os.environ))
        return config._ci_original_popen(command, *args, **kwargs)

    subprocess.Popen = popen
    import torch.multiprocessing as mp

    config._ci_original_spawn = mp.spawn

    def spawn(fn, args=(), nprocs=1, **kwargs):
        validate_launch(nprocs)
        return config._ci_original_spawn(fn, args=args, nprocs=nprocs, **kwargs)

    mp.spawn = spawn


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    batch_path = config.getoption("ci_batch")
    if batch_path:
        batch = json.loads(batch_path.read_text())
        config._ci_batch_cases = {row["nodeid"]: row for row in batch["cases"]}
        selected = [item for item in items if item.nodeid in config._ci_batch_cases]
        if {item.nodeid for item in selected} != set(config._ci_batch_cases):
            raise ResourcePlanError("Worker collection differs from the full-visibility inventory")
        deselected = [item for item in items if item not in selected]
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_collection_finish(session):
    output = session.config.getoption("ci_inventory")
    if not output:
        return
    from _pytest.skipping import evaluate_skip_marks

    rows = []
    for item in session.items:
        skip = evaluate_skip_marks(item)
        plan = resolve_plan(item)
        cards = plan.cards(runner_capabilities())
        if cards > len(runner_capabilities().devices) and skip is None:
            raise ResourcePlanError(f"{item.nodeid} requires {cards} devices")
        rows.append(
            dict(
                nodeid=item.nodeid,
                file=str(item.path),
                plan=plan.to_dict(),
                cards=cards,
                skip=skip.reason if skip else None,
            )
        )
    output.write_text(json.dumps(rows, indent=2) + "\n")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    rows = getattr(item.config, "_ci_batch_cases", None)
    if rows is None:
        return
    row = rows[item.nodeid]
    os.environ["VEOMNI_CI_ACTIVE_PLAN"] = json.dumps(row["plan"])
    os.environ["VEOMNI_CI_ACTIVE_NODEID"] = item.nodeid
    plan = resolve_plan(item)
    if plan.to_dict() != row["plan"]:
        raise ResourcePlanError(f"Launch plan changed after device assignment: {item.nodeid}")
    # Eligibility is evaluated with the runner's full owned visibility.
    item.own_markers[:] = [mark for mark in item.own_markers if mark.name not in ("skipif", "skip")]
    original_iter = item.iter_markers
    item.iter_markers = lambda name=None: (mark for mark in original_iter(name) if mark.name not in ("skipif", "skip"))
    if row["skip"] is not None:
        pytest.skip(row["skip"])
    if runner_capabilities().backend == "cpu" and any(spec.device != "cpu" for spec in plan.launches):
        pytest.skip("Requires an accelerator")


def pytest_unconfigure(config):
    subprocess.Popen = config._ci_original_popen
    import torch.multiprocessing as mp

    mp.spawn = config._ci_original_spawn


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    started = time.time()
    _progress(dict(kind="start", nodeid=item.nodeid, timestamp=started))
    yield
    row = item.config._ci_results.get(item.nodeid)
    if row:
        row["wall_seconds"] = time.time() - started
        _progress(dict(row, kind="result"))


def _progress(row):
    output = os.environ.get("CI_CASE_PROGRESS")
    if output:
        with open(output, "a") as handle:
            handle.write(json.dumps(row) + "\n")


def pytest_runtest_logreport(report):
    row = _active_config._ci_results.setdefault(
        report.nodeid, dict(nodeid=report.nodeid, outcome="incomplete", phases=[])
    )
    row["phases"].append(dict(when=report.when, outcome=report.outcome, duration=report.duration))
    if report.failed:
        row.update(outcome="failed", detail=str(report.longrepr))
        _progress(dict(row, kind="failure"))
    elif row["outcome"] != "failed":
        if report.skipped:
            row.update(outcome="xfail" if hasattr(report, "wasxfail") else "skipped", detail=str(report.longrepr))
        elif report.when == "call":
            row["outcome"] = "xpass" if hasattr(report, "wasxfail") else "passed"


def pytest_sessionfinish(session, exitstatus):
    output = os.environ.get("CI_CASE_RESULTS")
    if output:
        Path(output).write_text(
            json.dumps(dict(exitstatus=int(exitstatus), cases=list(session.config._ci_results.values())), indent=2)
            + "\n"
        )
