"""Isolated pytest workspaces and scoped subprocess cleanup."""

import os
import signal
import subprocess
import time

import psutil

from scripts.ci.accelerator import ROOT


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

        def remember_group():
            # Reparented children can outlive the launcher between samples.
            for child in psutil.process_iter():
                try:
                    if child.pid != process.pid and os.getpgid(child.pid) == process.pid:
                        known[(child.pid, child.create_time())] = child
                except (OSError, psutil.Error):
                    pass

        def alive():
            running = []
            for child in known.values():
                try:
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        running.append(child)
                except psutil.Error:
                    pass
            return running

        for signum in (signal.SIGTERM, signal.SIGKILL):
            remember_group()
            try:
                # This group belongs to start_new_session=True, even if its
                # original leader has exited. Never sweep unrelated processes.
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
            for child in alive():
                try:
                    child.send_signal(signum)
                except psutil.Error:
                    pass
            deadline = time.monotonic() + 5
            while process.poll() is None or alive():
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            remember_group()
            if process.poll() is not None and not alive():
                break
        else:
            raise RuntimeError("CI worker cleanup could not drain its process group and tracked descendants")


def create_workspace(path):
    path.mkdir(parents=True, exist_ok=False)
    for name in ("tests", "veomni", "scripts", "tasks", "configs", "pyproject.toml"):
        (path / name).symlink_to(ROOT / name, target_is_directory=(ROOT / name).is_dir())
    (path / "cache").mkdir()
