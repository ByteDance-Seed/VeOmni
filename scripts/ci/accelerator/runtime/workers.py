"""Isolated pytest workspaces and scoped subprocess cleanup."""

import os
import signal
import subprocess
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event

import psutil

from scripts.ci.accelerator import ROOT


_CANCELLATION = ContextVar("ci_worker_cancellation", default=None)


class ProcessCleanupError(RuntimeError):
    """Owned workers survived cleanup; their device lease cannot be reused."""


class _CancellationScope:
    def __init__(self):
        self.event = Event()
        self.parent = _CANCELLATION.get()

    def check(self):
        if self.parent is not None:
            self.parent.check()
        if self.event.is_set():
            raise CancelledError("CI worker interrupted")

    def run(self, function, *args, **kwargs):
        token = _CANCELLATION.set(self)
        try:
            self.check()
            return function(*args, **kwargs)
        finally:
            _CANCELLATION.reset(token)


class _WorkerExecutor(ThreadPoolExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.futures = []

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        self.futures.append(future)
        return future


@contextmanager
def interruptible_executor(*, max_workers, cancel_on_error=False):
    """Cancel owned subprocesses before waiting for interrupted callbacks."""
    scope = _CancellationScope()
    executor = _WorkerExecutor(max_workers=max_workers)
    try:
        yield executor, scope
    except BaseException as exc:
        if cancel_on_error or isinstance(exc, ProcessCleanupError) or not isinstance(exc, Exception):
            scope.event.set()
        raise
    finally:
        interrupted = None
        while True:
            try:
                executor.shutdown(wait=False, cancel_futures=scope.event.is_set())
                pending = [future for future in executor.futures if not future.done()]
                if pending:
                    wait(pending, timeout=0.2)
                    continue
                executor.shutdown(wait=True)
                break
            except (KeyboardInterrupt, SystemExit) as exc:
                scope.event.set()
                interrupted = exc
        # Future completion proves callbacks finished cleanup. A Thread.join
        # interrupted by SIGINT can report a stopped thread before it returns.
        if interrupted is not None:
            raise interrupted


def run_process(command, env, log, timeout, *, cwd=ROOT, allowed_codes=(0,), on_poll=None, on_failure=None):
    """Track this launcher only; clean descendants before releasing its lease."""
    scope = _CANCELLATION.get()
    if scope is not None:
        scope.check()
    known = {}
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    started = time.monotonic()
    try:
        while process.poll() is None:
            if scope is not None:
                scope.check()
            if on_poll is not None:
                on_poll()
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    known[(child.pid, child.create_time())] = child
            except psutil.Error:
                pass
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"CI worker exceeded {timeout}s: {command}")
            time.sleep(0.2)
        if on_poll is not None:
            on_poll()
        if process.returncode not in allowed_codes:
            raise subprocess.CalledProcessError(process.returncode, command)
    except Exception as exc:
        if on_failure is not None and not isinstance(exc, CancelledError):
            on_failure(exc)
        raise
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
            raise ProcessCleanupError("CI worker cleanup could not drain its process group and tracked descendants")


def create_workspace(path):
    path.mkdir(parents=True, exist_ok=False)
    for name in ("tests", "veomni", "scripts", "tasks", "configs", "pyproject.toml"):
        (path / name).symlink_to(ROOT / name, target_is_directory=(ROOT / name).is_dir())
    (path / "cache").mkdir()
