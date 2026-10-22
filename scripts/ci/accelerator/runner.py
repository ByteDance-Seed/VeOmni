"""Collect pytest cases, resolve launch plans, and schedule disjoint device leases."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import CancelledError, as_completed
from pathlib import Path

from scripts.ci.accelerator import ROOT
from scripts.ci.accelerator.batching import make_batches
from scripts.ci.accelerator.launch_plan import ResourcePlanError, RunnerCapabilities
from scripts.ci.accelerator.reporting import (
    BatchProgress,
    RunReporter,
    format_status,
    print_failure_log,
    selection_label,
)
from scripts.ci.accelerator.runtime.devices import child_environment
from scripts.ci.accelerator.runtime.ports import find_free_port
from scripts.ci.accelerator.runtime.workers import (
    ProcessCleanupError,
    create_workspace,
    interruptible_executor,
    run_process,
)
from scripts.ci.accelerator.scheduler import DeviceJob, run_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("unit", "e2e"), required=True)
    parser.add_argument("--backend", choices=("auto", "cuda", "npu", "cpu"), default="auto")
    parser.add_argument("--include-conditional", action="store_true")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", help="Comma-separated suite selection names")
    parser.add_argument("--pytest-arg", action="append", default=[])
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--batch-seconds", type=float, default=120)
    parser.add_argument("--batch-cases", type=int, default=64)
    args = parser.parse_args()
    if args.batch_seconds <= 0 or args.batch_cases < 1:
        parser.error("Batch duration and case count must be positive")
    output = (args.output_dir or ROOT / "outputs/accelerator-ci" / args.suite).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "inventory.json").exists():
        parser.error("Use a fresh output directory to preserve each run's evidence")
    started = time.time()
    source_paths = [
        path
        for pattern in ("scripts/ci/**/*.py", "scripts/ci/**/*.json", "tests/**/*.py", "tests/toy_config/**/*.json")
        for path in ROOT.glob(pattern)
    ]
    source_paths.extend([ROOT / "pyproject.toml", ROOT / "uv.lock"])
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    (output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    caps_path = output / "capabilities.json"
    probe_env = child_environment(RunnerCapabilities("cpu"), [], environ=env) if args.backend == "cpu" else env
    with (output / "probe.log").open("w") as log:
        run_process(
            [sys.executable, "-m", "scripts.ci.accelerator.runtime.devices", str(caps_path), args.backend],
            probe_env,
            log,
            120,
        )
    value = json.loads(caps_path.read_text())
    caps = RunnerCapabilities(value["backend"], tuple(value["devices"]))
    env["VEOMNI_CI_CAPABILITIES"] = str(caps_path)
    env["VEOMNI_CI_PORT_REGISTRY"] = str(output / "ports.json")
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(max(1, min(8, (os.cpu_count() or 8) // 8)))
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    env["VEOMNI_GPU_TEST_PARALLEL"] = "0" if args.serial else "1"
    manifest_path = Path(__file__).parent / "suites" / ("npu.json" if caps.backend == "npu" else "gpu.json")
    manifest = json.loads(manifest_path.read_text())
    selected = [
        row
        for row in manifest
        if row["suite"] == args.suite and (not row.get("conditional") or args.include_conditional)
    ]
    if args.only:
        selected = [row for row in selected if row["name"] in args.only.split(",")]
    if not selected:
        parser.error("No suite selections matched")

    def collect(selection):
        name = selection["name"]
        inventory_path = output / f"{name}.inventory.json"
        collect_env = {**env, **selection.get("env", {})}
        for key in ("CI_TIMING_EVENTS", "CI_CASE_RESULTS", "CI_CASE_PROGRESS", "VEOMNI_CI_ACTIVE_PLAN"):
            collect_env.pop(key, None)
        command = [
            sys.executable,
            "-m",
            "pytest",
            *selection["args"],
            *args.pytest_arg,
            "-p",
            "scripts.ci.accelerator.pytest_plugin",
            "--collect-only",
            "-q",
            f"--ci-inventory={inventory_path}",
        ]
        try:
            with (output / f"{name}.collection.log").open("w") as log:
                run_process(command, collect_env, log, 600, allowed_codes=(0, 5))
        except Exception as exc:
            if not isinstance(exc, CancelledError):
                print(
                    format_status(selection_label(args.suite, name), "FAILED", None, f"collection: {exc}"), flush=True
                )
                print_failure_log(output / f"{name}.collection.log", label=selection_label(args.suite, name))
            raise
        if not inventory_path.exists():
            raise ResourcePlanError(f"Collection produced no inventory: {name}")
        rows = json.loads(inventory_path.read_text())
        target = " ".join(arg for arg in selection["args"] if arg.startswith("tests/")) or name
        print(
            format_status(selection_label(args.suite, name), "COLLECT", None, f"{target} (cases={len(rows)})"),
            flush=True,
        )
        return selection, rows

    try:
        with interruptible_executor(max_workers=8, cancel_on_error=True) as (executor, scope):
            futures = {
                executor.submit(scope.run, collect, selection): index for index, selection in enumerate(selected)
            }
            inventories = [None] * len(selected)
            for future in as_completed(futures):
                inventories[futures[future]] = future.result()
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        (output / "summary.json").write_text(
            json.dumps(
                dict(suite=args.suite, started=started, ended=time.time(), stage="collection", error=repr(exc)),
                indent=2,
            )
            + "\n"
        )
        raise
    (output / "inventory.json").write_text(
        json.dumps([{**selection, "cases": rows} for selection, rows in inventories], indent=2) + "\n"
    )
    batches = make_batches(
        inventories, target_seconds=args.batch_seconds, max_cases=args.batch_cases, capacity=len(caps.devices)
    )
    (output / "batches.json").write_text(json.dumps(batches, indent=2) + "\n")
    if args.collect_only:
        return 0
    collected_at = time.time()
    finished = {}
    reporter = RunReporter(
        sum(len(rows) for _, rows in inventories),
        len(batches),
        suite=args.suite,
        inventories=inventories,
        batches=batches,
        keep_going=args.keep_going,
    )

    def execute(job, assigned):
        row = job.payload
        name = row["name"]
        workspace = output / "work" / name
        create_workspace(workspace)
        temporary = tempfile.mkdtemp(prefix="vci-", dir="/tmp")
        worker_env = child_environment(
            caps, assigned, cpu=not row["cards"], environ={**env, **row["selection"].get("env", {})}
        )
        worker_env.update(
            VEOMNI_CI_LEASE=json.dumps(assigned),
            TMPDIR=temporary,
            CACHE_DIR=str(workspace / "cache"),
            CI_TIMING_EVENTS=str(output / f"{name}.jsonl"),
            CI_LAUNCH_EVENTS=str(output / f"{name}.launches.jsonl"),
            CI_CASE_RESULTS=str(output / f"{name}.results.json"),
            CI_CASE_PROGRESS=str(output / f"{name}.progress.jsonl"),
        )
        worker_env["MASTER_ADDR"] = "127.0.0.1"
        worker_env["MASTER_PORT"] = str(find_free_port(worker_env["VEOMNI_CI_PORT_REGISTRY"]))
        batch_path = output / f"{name}.batch.json"
        batch_path.write_text(json.dumps(row, indent=2) + "\n")
        flags = [arg for arg in row["selection"]["args"] if not arg.startswith("tests/")]
        if args.keep_going:
            flags = [arg for arg in flags if arg != "-x"]
        command = [
            sys.executable,
            "-m",
            "pytest",
            *flags,
            row["cases"][0]["file"],
            *args.pytest_arg,
            "-c",
            str(ROOT / "pyproject.toml"),
            f"--rootdir={ROOT}",
            "-p",
            "scripts.ci.accelerator.pytest_plugin",
            f"--ci-batch={batch_path}",
            f"--junitxml={output / (name + '.xml')}",
        ]
        record = dict(
            name=name,
            returncode=1,
            status="failed",
            selection=row["selection"]["name"],
            cards=row["cards"],
            assigned=assigned,
            command=command,
            started=time.time(),
            planned_cases=[case["nodeid"] for case in row["cases"]],
            temporary_directory=temporary,
        )
        progress = BatchProgress(
            reporter,
            name,
            assigned,
            output / f"{name}.progress.jsonl",
            output / f"{name}.log",
            batch_file=row["cases"][0]["nodeid"].split("::", 1)[0],
            case_count=len(row["cases"]),
        )
        record["display_id"] = reporter.label(name)
        progress.start()
        try:
            with (output / f"{name}.log").open("w") as log:
                run_process(
                    command,
                    worker_env,
                    log,
                    5400,
                    cwd=workspace,
                    on_poll=progress.poll,
                    on_failure=progress.worker_failure,
                )
            result_path = output / f"{name}.results.json"
            if not result_path.is_file():
                raise ResourcePlanError(f"Worker produced no results: {result_path}")
            result = json.loads(result_path.read_text())
            if result["exitstatus"] or {case["nodeid"] for case in result["cases"]} != set(record["planned_cases"]):
                raise ResourcePlanError(f"Case execution differs from inventory: {name}")
            record.update(returncode=0, status="passed")
        except CancelledError as exc:
            progress.poll(check_abort=False)
            if progress.failures:
                record.update(returncode=1, status="failed", error=reporter.first_failure)
                with reporter.lock:
                    print_failure_log(progress.log_path, label=reporter.label(name))
            else:
                reason = f"Cancelled after {reporter.first_failure}" if reporter.first_failure else str(exc)
                record.update(returncode=130, status="cancelled", error=reason)
                progress.cancelled(reason)
            raise
        except Exception as exc:
            record.update(returncode=getattr(exc, "returncode", 1), error=repr(exc))
            if isinstance(exc, ProcessCleanupError):
                reporter.stop_event.set()
                reporter.fail(name, str(exc), assigned=assigned, target=progress.current, nodeid=progress.current)
            else:
                progress.worker_failure(exc)
            with reporter.lock:
                print_failure_log(progress.log_path, label=reporter.label(name))
            if not args.keep_going or isinstance(exc, ProcessCleanupError):
                raise
        finally:
            progress.poll(check_abort=False)
            result_path = output / f"{name}.results.json"
            if not result_path.exists():
                result_path.write_text(json.dumps(progress.partial_results(), indent=2) + "\n")
            if record.get("returncode", 1):
                retained = workspace / "temporary-files"
                shutil.move(temporary, retained)
                record["retained_temporary_directory"] = str(retained)
            else:
                shutil.rmtree(temporary)
            record.update(ended=time.time(), wall_seconds=time.time() - record["started"])
            (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
            finished[name] = record
            progress.done(record)
        return record

    jobs = [DeviceJob(row["name"], row["cards"], row) for row in batches]
    error = None
    try:
        if args.serial:
            for job in jobs:
                execute(job, list(caps.devices) if job.cards else [])
        else:
            run_jobs(
                jobs, list(caps.devices), execute, cancel_on_error=not args.keep_going, stop_event=reporter.stop_event
            )
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        error = reporter.first_failure or repr(exc)
        if not isinstance(exc, Exception):
            raise
    finally:
        error = error or reporter.first_failure
        summary = dict(
            suite=args.suite,
            backend=caps.backend,
            started=started,
            collected_at=collected_at,
            ended=time.time(),
            wall_seconds=time.time() - started,
            execution_seconds=time.time() - collected_at,
            selection_count=len(selected),
            batch_count=len(batches),
            case_count=sum(len(rows) for _, rows in inventories),
            commands=finished,
            error=error,
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        case_rows = []
        for job in jobs:
            path = output / f"{job.name}.results.json"
            if path.exists():
                for case in json.loads(path.read_text())["cases"]:
                    case_rows.append(
                        {
                            **case,
                            "batch": job.name,
                            "selection": job.payload["selection"]["name"],
                            "display_id": reporter.label(job.name, case["nodeid"]),
                        }
                    )
        (output / "cases.json").write_text(json.dumps(case_rows, indent=2) + "\n")
        with (output / "cases.csv").open("w") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("nodeid", "outcome", "wall_seconds", "batch", "selection"), extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(case_rows)
        (output / "durations.json").write_text(
            json.dumps({row["nodeid"]: row.get("wall_seconds", 0) for row in case_rows}, indent=2) + "\n"
        )
        changed = [
            name
            for name, digest in sources.items()
            if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        ]
        summary["changed_sources"] = changed
        if changed:
            error = error or "Sources changed during execution"
            summary["error"] = error
            reporter.fail("runner", "Sources changed during execution")
        reporter.summary(finished)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(error is not None or len(finished) != len(jobs) or any(row["returncode"] for row in finished.values()))
