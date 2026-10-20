"""Execute the GPU CI manifest with bounded per-file GPU reservations."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
from gpu_pool import GPUJob, run_jobs


ROOT = Path(__file__).resolve().parents[2]


def run_process(command, env, log, timeout):
    """Track only this launcher's descendants, including torchrun worker sessions."""
    known = {}
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    started = time.monotonic()
    try:
        while process.poll() is None:
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    known[(child.pid, child.create_time())] = child
            except psutil.Error:
                pass
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"GPU CI command exceeded {timeout}s: {command}")
            time.sleep(2)
        if process.returncode:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["unit", "e2e"], required=True)
    parser.add_argument("--include-conditional", action="store_true")
    parser.add_argument("--serial", action="store_true", help="Disable concurrent scheduling for diagnosis")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--keep-going", action="store_true", help="Attempt all commands when collecting timing evidence"
    )
    parser.add_argument("--pytest-arg", action="append", default=[])
    parser.add_argument("--only", help="Comma-separated manifest task names (local diagnosis)")
    args = parser.parse_args()
    output = (args.output_dir or ROOT / "outputs/gpu-ci" / args.suite).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(__file__).with_name("gpu_tests.json").read_text())
    selected = [r for r in manifest if r["suite"] == args.suite and (not r["conditional"] or args.include_conditional)]
    if args.only:
        selected = [r for r in selected if r["name"] in args.only.split(",")]
        if not selected:
            parser.error("--only selected no tasks")
    # Start audited smaller jobs first so independent files can fill the node.
    selected.sort(key=lambda r: (r["gpus"] == 8, -r.get("priority", 0)))
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(devices) != 8:
        parser.error("The full GPU CI suite requires exactly eight visible GPUs")
    jobs = [GPUJob(r["name"], 8 if args.serial else r["gpus"], r) for r in selected]
    started = time.time()

    def execute(job, assigned):
        row = job.payload
        env = os.environ.copy()
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
        # Bound concurrent CPU BLAS/JIT work as well as GPU ownership.
        env["OMP_NUM_THREADS"] = str(max(1, min(8, (os.cpu_count() or 8) // 8)))
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        env["VEOMNI_GPU_TEST_PARALLEL"] = "0" if args.serial else "1"
        env["CI_TIMING_EVENTS"] = str(output / f"{job.name}.jsonl")
        command = [sys.executable, "-m", "pytest", *row["args"]]
        if args.keep_going:
            command = [v for v in command if v != "-x"]
        command += [*args.pytest_arg, f"--junitxml={output / (job.name + '.xml')}"]
        record = dict(row, assigned=assigned, command=command, started=time.time())
        print(f"START {job.name} GPUs={env['CUDA_VISIBLE_DEVICES']}", flush=True)
        try:
            with (output / f"{job.name}.log").open("w") as log:
                run_process(command, env, log, timeout=5400)
            record["returncode"] = 0
        except Exception as exc:
            record.update(returncode=getattr(exc, "returncode", 1), error=repr(exc))
            if not args.keep_going:
                raise
        finally:
            record.update(ended=time.time(), wall_seconds=time.time() - record["started"])
            (output / f"{job.name}.json").write_text(json.dumps(record, indent=2))
            print(f"DONE {job.name} rc={record.get('returncode', 1)} {record['wall_seconds']:.1f}s", flush=True)
        return record

    results = run_jobs(jobs, devices, execute)
    summary = dict(
        suite=args.suite,
        started=started,
        ended=time.time(),
        wall_seconds=time.time() - started,
        summed_command_seconds=sum(r["wall_seconds"] for r in results.values()),
        commands=results,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return int(any(r["returncode"] for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
