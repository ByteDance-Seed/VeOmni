"""Foreground case progress and failure diagnostics for concurrent batches."""

import json
import os
import time
from collections import Counter
from concurrent.futures import CancelledError
from threading import Event, Lock

from scripts.ci.accelerator.launch_plan import ResourcePlanError


class CaseFailure(ResourcePlanError):
    """A pytest phase reported a real failure before the worker exited."""


def print_log_tail(path, *, lines=80):
    if not path.is_file():
        return
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 65536))
        tail = handle.read().decode(errors="replace").splitlines()[-lines:]
    print(f"--- log tail: {path} ---\n" + "\n".join(tail), flush=True)


class RunReporter:
    def __init__(self, total_cases, total_batches, *, keep_going=False, heartbeat_seconds=30):
        self.total_cases = total_cases
        self.total_batches = total_batches
        self.keep_going = keep_going
        self.heartbeat_seconds = heartbeat_seconds
        self.first_failure = None
        self.stop_event = Event()
        self.outcomes = {}
        self.lock = Lock()

    def fail(self, batch, message, *, detail=None):
        with self.lock:
            if self.first_failure is None:
                self.first_failure = f"{batch}: {message}"
            if not self.keep_going:
                self.stop_event.set()
            print(f"FAIL {batch}: {message}", flush=True)
            if os.environ.get("GITHUB_ACTIONS") == "true":
                escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
                print(f"::error::{escaped}", flush=True)
            if detail:
                lines = detail.splitlines()
                print("\n".join(lines[:200]), flush=True)
                if len(lines) > 200:
                    print("... full traceback is retained in the batch log/results artifact", flush=True)

    def summary(self, records):
        with self.lock:
            outcomes = dict(Counter(self.outcomes.values()))
            batches = dict(Counter(row.get("status", "failed") for row in records.values()))
            print(
                f"SUMMARY cases={len(self.outcomes)}/{self.total_cases} outcomes={outcomes} "
                f"batches={len(records)}/{self.total_batches} statuses={batches} "
                f"not_started_batches={self.total_batches - len(records)}",
                flush=True,
            )
            if self.first_failure:
                print(f"FIRST FAILURE {self.first_failure}", flush=True)


class BatchProgress:
    def __init__(self, reporter, name, assigned, progress_path, log_path):
        self.reporter = reporter
        self.name = name
        self.assigned = assigned
        self.path = progress_path
        self.log_path = log_path
        self.offset = 0
        self.buffer = b""
        self.rows = {}
        self.failures = set()
        self.worker_failed = False
        self.current = "worker startup"
        self.started = time.monotonic()
        self.heartbeat_at = self.started

    def poll(self, *, check_abort=True):
        if self.path.is_file():
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                self.buffer += handle.read()
                self.offset = handle.tell()
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if line:
                    self._event(json.loads(line))
        now = time.monotonic()
        if now - self.heartbeat_at >= self.reporter.heartbeat_seconds:
            with self.reporter.lock:
                print(
                    f"RUNNING {self.name} devices={self.assigned} elapsed={now - self.started:.0f}s "
                    f"case={self.current}",
                    flush=True,
                )
            self.heartbeat_at = now
        failure = self.reporter.first_failure
        if check_abort and failure and not self.reporter.keep_going:
            if self.failures:
                raise CaseFailure(failure)
            raise CancelledError(f"Cancelled after {failure}")

    def _event(self, event):
        nodeid = event["nodeid"]
        kind = event.get("kind", "result")
        if kind == "start":
            self.current = nodeid
            self.rows[nodeid] = dict(nodeid=nodeid, outcome="incomplete", phases=[])
            with self.reporter.lock:
                print(f"RUN [{self.name}] {nodeid} devices={self.assigned}", flush=True)
            return
        row = {key: value for key, value in event.items() if key != "kind"}
        self.rows[nodeid] = row
        if event["outcome"] == "failed" and nodeid not in self.failures:
            self.failures.add(nodeid)
            self.reporter.fail(self.name, nodeid, detail=event.get("detail"))
        key = self.name, nodeid
        with self.reporter.lock:
            already_reported = key in self.reporter.outcomes
            self.reporter.outcomes[key] = event["outcome"]
            if not already_reported:
                print(
                    f"CASE {event['outcome'].upper()} {len(self.reporter.outcomes)}/{self.reporter.total_cases} "
                    f"[{self.name}] {nodeid}",
                    flush=True,
                )
        if kind == "result":
            self.current = "between cases / worker teardown"

    def worker_failure(self, exc):
        if not self.failures and not self.worker_failed:
            self.reporter.fail(self.name, str(exc))
        self.worker_failed = True

    def partial_results(self):
        rows = []
        for value in self.rows.values():
            row = dict(value)
            if row["outcome"] == "incomplete":
                row["outcome"] = "cancelled"
            rows.append(row)
        return dict(exitstatus=1, partial=True, cases=rows)
