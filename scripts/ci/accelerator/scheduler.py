"""Backend-independent scheduling with disjoint leases and bounded backfilling."""

import time
from concurrent.futures import FIRST_COMPLETED, CancelledError, wait
from dataclasses import dataclass

from scripts.ci.accelerator.runtime.workers import interruptible_executor


@dataclass(frozen=True)
class DeviceJob:
    name: str
    cards: int
    payload: object


def run_jobs(jobs, devices, run, *, max_cpu_jobs=2, drain_after=120.0, cancel_on_error=False, stop_event=None):
    """Release a lease only after its callback has cleaned up all descendants.

    A waiting head job may be bypassed until its drain deadline. Stop admitting
    accelerator work after that deadline, so running batches can drain. CPU
    work has a separate concurrency limit and does not delay device draining.
    """
    if len(set(devices)) != len(devices) or max_cpu_jobs < 1:
        raise ValueError("Expected distinct devices and positive CPU concurrency")
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("Job names must be unique")
    if any(type(job.cards) is not int or not 0 <= job.cards <= len(devices) for job in jobs):
        raise ValueError("Job device reservation exceeds the available devices")
    pending = list(jobs)
    available = list(devices)
    active = {}
    results = {}
    waiting_since = time.monotonic()
    with interruptible_executor(max_workers=max(1, len(devices) + max_cpu_jobs), cancel_on_error=cancel_on_error) as (
        executor,
        scope,
    ):

        def check_stop():
            if stop_event is not None and stop_event.is_set():
                scope.event.set()
                raise CancelledError("Scheduling cancelled after a worker failure")

        while pending or active:
            check_stop()
            head = next((job for job in pending if job.cards), None)
            draining = head is not None and time.monotonic() - waiting_since >= drain_after
            cpu_active = sum(job.cards == 0 for job, _ in active.values())
            for job in list(pending):
                check_stop()
                if job.cards == 0:
                    if cpu_active >= max_cpu_jobs:
                        continue
                    cpu_active += 1
                elif job.cards > len(available) or (draining and job is not head):
                    continue
                assigned = available[: job.cards]
                del available[: job.cards]
                pending.remove(job)
                active[executor.submit(scope.run, run, job, assigned)] = (job, assigned)
                if job is head:
                    waiting_since = time.monotonic()
                    if draining:
                        break
            if not active:
                raise RuntimeError("No runnable job in a validated resource queue")
            done, _ = wait(active, return_when=FIRST_COMPLETED, timeout=0.2 if stop_event is not None else None)
            for future in done:
                job, assigned = active.pop(future)
                results[job.name] = future.result()
                available.extend(assigned)
                available.sort(key=devices.index)
    return {job.name: results[job.name] for job in jobs}
