"""Foreground case progress and failure diagnostics for concurrent batches."""

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import CancelledError
from threading import Event, Lock

from scripts.ci.accelerator.launch_plan import ResourcePlanError


class CaseFailure(ResourcePlanError):
    """A pytest phase reported a real failure before the worker exited."""


def selection_label(suite, name):
    prefix = f"npu-{suite}-"
    if name.startswith(prefix):
        return f"{suite}-{name[len(prefix) :]}"
    if name.startswith(f"{suite}-"):
        return name
    suffix = f"-{suite}"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return f"{suite}-{name}"


def format_status(label, status, assigned, target):
    devices = "-" if assigned is None else ",".join(str(device) for device in assigned) or "cpu"
    return f"[{label}][{status}][devices={devices}] {target}"


def print_failure_log(path, *, label):
    """Stream complete failed-worker output so nested process errors stay visible."""
    if not path.is_file():
        return
    grouped = os.environ.get("GITHUB_ACTIONS") == "true"
    title = f"{label}: full worker output ({path})"
    print(f"::group::{title}" if grouped else f"--- {title} ---", flush=True)
    try:
        sys.stdout.write(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        print(f"Could not read worker log: {exc}", flush=True)
    finally:
        print("\n::endgroup::" if grouped else "", flush=True)


class RunReporter:
    def __init__(
        self,
        total_cases,
        total_batches,
        *,
        suite="unit",
        inventories=(),
        batches=(),
        keep_going=False,
        heartbeat_seconds=30,
    ):
        self.total_cases = total_cases
        self.total_batches = total_batches
        self.suite = suite
        self.labels = {batch["name"]: selection_label(suite, batch["selection"]["name"]) for batch in batches}
        collected_ids = {
            (selection["name"], row["nodeid"]): f"{selection_label(suite, selection['name'])}-case-{index:04d}"
            for selection, rows in inventories
            for index, row in enumerate(rows, 1)
        }
        self.case_ids = {
            (batch["name"], row["nodeid"]): collected_ids[batch["selection"]["name"], row["nodeid"]]
            for batch in batches
            for row in batch["cases"]
        }
        self.keep_going = keep_going
        self.heartbeat_seconds = heartbeat_seconds
        self.first_failure = None
        self.stop_event = Event()
        self.outcomes = {}
        self.lock = Lock()

    def label(self, name, nodeid=None):
        if nodeid is not None and (name, nodeid) in self.case_ids:
            return self.case_ids[name, nodeid]
        return self.labels.get(name, self.suite if name is None or name == "runner" else name)

    def status(self, batch, status, assigned, target, *, nodeid=None):
        with self.lock:
            print(format_status(self.label(batch, nodeid), status, assigned, target), flush=True)

    def _failure(self, label, message, assigned, target, status, detail):
        if self.first_failure is None:
            self.first_failure = f"{label}: {message}"
        if not self.keep_going:
            self.stop_event.set()
        print(format_status(label, status, assigned, target), flush=True)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::error::{escaped}", flush=True)
        if detail:
            lines = detail.splitlines()
            print("\n".join("    " + line for line in lines[:200]), flush=True)
            if len(lines) > 200:
                print("    ... full traceback is in the complete worker output below", flush=True)

    def fail(self, batch, message, *, assigned=None, target=None, nodeid=None, detail=None):
        with self.lock:
            target = f"{target} (error={message})" if target else message
            self._failure(self.label(batch, nodeid), message, assigned, target, "FAILED", detail)

    def result(self, batch, row, assigned):
        key = batch, row["nodeid"]
        with self.lock:
            previous = self.outcomes.get(key)
            self.outcomes[key] = row["outcome"]
            if previous == row["outcome"]:
                return
            status = f"{row['outcome'].upper()} ({len(self.outcomes)}/{self.total_cases})"
            if row["outcome"] == "failed":
                self._failure(
                    self.label(batch, row["nodeid"]), row["nodeid"], assigned, row["nodeid"], status, row.get("detail")
                )
            else:
                print(format_status(self.label(batch, row["nodeid"]), status, assigned, row["nodeid"]), flush=True)

    def summary(self, records):
        with self.lock:
            outcomes = dict(Counter(self.outcomes.values()))
            batches = dict(Counter(row.get("status", "failed") for row in records.values()))
            target = (
                f"cases={len(self.outcomes)}/{self.total_cases} outcomes={outcomes} "
                f"batches={len(records)}/{self.total_batches} statuses={batches} "
                f"not_started_batches={self.total_batches - len(records)}"
            )
            print(format_status(self.suite, "SUMMARY", None, target), flush=True)
            if self.first_failure:
                print(format_status(self.suite, "FIRST_FAILURE", None, self.first_failure), flush=True)


class BatchProgress:
    def __init__(self, reporter, name, assigned, progress_path, log_path, *, batch_file=None, case_count=0):
        self.reporter = reporter
        self.name = name
        self.assigned = assigned
        self.path = progress_path
        self.log_path = log_path
        self.batch_file = batch_file
        self.case_count = case_count
        self.offset = 0
        self.buffer = b""
        self.rows = {}
        self.failures = set()
        self.worker_failed = False
        self.current = batch_file or "worker startup"
        self.started = time.monotonic()
        self.heartbeat_at = self.started

    def start(self):
        self.reporter.status(self.name, "START", self.assigned, f"{self.batch_file} (cases={self.case_count})")

    def done(self, record):
        self.reporter.status(
            self.name,
            "DONE",
            self.assigned,
            f"{self.batch_file} (cases={self.case_count}, elapsed={record['wall_seconds']:.1f}s, rc={record['returncode']})",
        )

    def cancelled(self, reason):
        self.reporter.status(
            self.name, "CANCELLED", self.assigned, f"{self.current} (reason={reason})", nodeid=self.current
        )

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
            self.reporter.status(
                self.name,
                "RUNNING",
                self.assigned,
                f"{self.current} (elapsed={now - self.started:.0f}s)",
                nodeid=self.current,
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
        if self.batch_file is None:
            self.batch_file = nodeid.split("::", 1)[0]
        if kind == "start":
            self.current = nodeid
            self.rows[nodeid] = dict(nodeid=nodeid, outcome="incomplete", phases=[])
            self.reporter.status(self.name, "RUN", self.assigned, nodeid, nodeid=nodeid)
            return
        row = {key: value for key, value in event.items() if key != "kind"}
        self.rows[nodeid] = row
        if event["outcome"] == "failed":
            self.failures.add(nodeid)
        self.reporter.result(self.name, row, self.assigned)
        if kind == "result":
            self.current = self.batch_file

    def worker_failure(self, exc):
        if not self.failures and not self.worker_failed:
            self.reporter.fail(self.name, str(exc), assigned=self.assigned, target=self.current, nodeid=self.current)
        self.worker_failed = True

    def partial_results(self):
        rows = []
        for value in self.rows.values():
            row = dict(value)
            if row["outcome"] == "incomplete":
                row["outcome"] = "cancelled"
            rows.append(row)
        return dict(exitstatus=1, partial=True, cases=rows)
