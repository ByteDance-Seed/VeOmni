"""Collect pytest cases, resolve launch plans, and schedule disjoint device leases."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

from scripts.ci.accelerator import child_environment
from scripts.ci.device_pool import DeviceJob, run_jobs
from scripts.ci.launch_plan import ResourcePlanError, RunnerCapabilities
from scripts.ci.ports import find_free_port


ROOT = Path(__file__).resolve().parents[2]


def run_process(command, env, log, timeout, *, cwd=ROOT, allowed_codes=(0,)):
    """Track this launcher only; clean descendants before releasing its lease."""
    known = {}
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    started = time.monotonic()
    try:
        while process.poll() is None:
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    known[(child.pid, child.create_time())] = child
            except psutil.Error:
                pass
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"CI worker exceeded {timeout}s: {command}")
            time.sleep(0.2)
        if process.returncode not in allowed_codes:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        for child in known.values():
            try:
                child.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(list(known.values()), timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=5)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def make_batches(inventories, *, durations=None, target_seconds=120, max_cases=64):
    durations = durations or {}
    batches = []
    for selection, rows in inventories:
        groups = {}
        for row in rows:
            groups.setdefault((row["file"], row["cards"]), []).append(row)
        for (_, cards), cases in groups.items():
            current = []
            estimated = 0.0
            for row in cases:
                cost = max(0.01, durations.get(row["nodeid"], 60.0 if cards > 1 else 2.0))
                if current and (len(current) >= max_cases or estimated + cost > target_seconds):
                    batches.append(dict(selection=selection, cards=cards, cases=current, estimated_seconds=estimated))
                    current, estimated = [], 0.0
                current.append(row)
                estimated += cost
            if current:
                batches.append(dict(selection=selection, cards=cards, cases=current, estimated_seconds=estimated))
    # Start long smaller batches first; bounded backfilling subsequently drains
    # for full-node jobs. History affects order, never resource requests.
    batches.sort(key=lambda row: (-row["estimated_seconds"], row["cards"]))
    for index, row in enumerate(batches):
        row["name"] = f"{row['selection']['name']}-batch-{index:04d}"
    return batches


def _workspace(path):
    path.mkdir(parents=True, exist_ok=False)
    for name in ("tests", "veomni", "scripts", "tasks", "configs", "pyproject.toml"):
        (path / name).symlink_to(ROOT / name, target_is_directory=(ROOT / name).is_dir())
    (path / "cache").mkdir()


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
    parser.add_argument("--history", type=Path, help="Previous per-case duration JSON; optional")
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
        for pattern in ("scripts/ci/*.py", "scripts/ci/*.json", "tests/**/*.py", "tests/toy_config/**/*.json")
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
            [sys.executable, "-m", "scripts.ci.accelerator", str(caps_path), args.backend], probe_env, log, 120
        )
    value = json.loads(caps_path.read_text())
    caps = RunnerCapabilities(value["backend"], tuple(value["devices"]))
    env["VEOMNI_CI_CAPABILITIES"] = str(caps_path)
    env["VEOMNI_CI_PORT_REGISTRY"] = str(output / "ports.json")
    env["OMP_NUM_THREADS"] = str(max(1, min(8, (os.cpu_count() or 8) // 8)))
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    env["VEOMNI_GPU_TEST_PARALLEL"] = "0" if args.serial else "1"
    manifest_path = Path(__file__).with_name("npu_tests.json" if caps.backend == "npu" else "gpu_tests.json")
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
            "scripts.ci.pytest_plan",
            "--collect-only",
            "-q",
            f"--ci-inventory={inventory_path}",
        ]
        with (output / f"{name}.collection.log").open("w") as log:
            run_process(command, collect_env, log, 600, allowed_codes=(0, 5))
        if not inventory_path.exists():
            raise ResourcePlanError(f"Collection produced no inventory: {name}")
        rows = json.loads(inventory_path.read_text())
        print(f"COLLECT {name} cases={len(rows)}", flush=True)
        return selection, rows

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            inventories = list(executor.map(collect, selected))
    except Exception as exc:
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
    durations = json.loads(args.history.read_text()) if args.history else {}
    batches = make_batches(
        inventories, durations=durations, target_seconds=args.batch_seconds, max_cases=args.batch_cases
    )
    (output / "batches.json").write_text(json.dumps(batches, indent=2) + "\n")
    if args.collect_only:
        return 0
    collected_at = time.time()
    finished = {}

    def execute(job, assigned):
        row = job.payload
        name = row["name"]
        workspace = output / "work" / name
        _workspace(workspace)
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
            "scripts.ci.pytest_plan",
            f"--ci-batch={batch_path}",
            f"--junitxml={output / (name + '.xml')}",
        ]
        record = dict(
            name=name,
            selection=row["selection"]["name"],
            cards=row["cards"],
            assigned=assigned,
            command=command,
            started=time.time(),
            planned_cases=[case["nodeid"] for case in row["cases"]],
            temporary_directory=temporary,
        )
        print(f"START {name} devices={assigned} cases={len(row['cases'])}", flush=True)
        try:
            with (output / f"{name}.log").open("w") as log:
                run_process(command, worker_env, log, 5400, cwd=workspace)
            result = json.loads((output / f"{name}.results.json").read_text())
            if result["exitstatus"] or {case["nodeid"] for case in result["cases"]} != set(record["planned_cases"]):
                raise ResourcePlanError(f"Case execution differs from inventory: {name}")
            record["returncode"] = 0
        except Exception as exc:
            record.update(returncode=getattr(exc, "returncode", 1), error=repr(exc))
            if not args.keep_going:
                raise
        finally:
            if record.get("returncode", 1):
                retained = workspace / "temporary-files"
                shutil.move(temporary, retained)
                record["retained_temporary_directory"] = str(retained)
            else:
                shutil.rmtree(temporary)
            record.update(ended=time.time(), wall_seconds=time.time() - record["started"])
            (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
            finished[name] = record
            print(f"DONE {name} rc={record.get('returncode', 1)} {record['wall_seconds']:.1f}s", flush=True)
        return record

    jobs = [DeviceJob(row["name"], row["cards"], row) for row in batches]
    error = None
    try:
        if args.serial:
            for job in jobs:
                execute(job, list(caps.devices) if job.cards else [])
        else:
            run_jobs(jobs, list(caps.devices), execute)
    except Exception as exc:
        error = repr(exc)
    finally:
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
                    case_rows.append({**case, "batch": job.name, "selection": job.payload["selection"]["name"]})
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
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        ]
        summary["changed_sources"] = changed
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if changed:
            error = "Sources changed during execution"
    return int(error is not None or len(finished) != len(jobs) or any(row["returncode"] for row in finished.values()))


if __name__ == "__main__":
    raise SystemExit(main())
