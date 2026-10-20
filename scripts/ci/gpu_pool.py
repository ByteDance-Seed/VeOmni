"""Run independent jobs on disjoint GPU subsets, keeping full-size jobs exclusive."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class GPUJob:
    name: str
    gpus: int
    payload: object


def run_jobs(jobs: list[GPUJob], devices: list[str], run: Callable[[GPUJob, list[str]], T]) -> dict[str, T]:
    """Reserve devices before dispatch and release them only after the job returns.

    The callback must finish cleaning up its own workers before returning. On
    failure, stop dispatching; already running callbacks finish before raising.
    Results retain input order regardless of completion order.
    """
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Expected a nonempty list of distinct GPUs")
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("Job names must be unique")
    if any(not 1 <= job.gpus <= len(devices) for job in jobs):
        raise ValueError("Job GPU reservation exceeds the available devices")
    pending = list(jobs)
    available = list(devices)
    active = {}
    results = {}
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        while pending or active:
            for job in list(pending):
                if job.gpus > len(available):
                    continue
                assigned = available[: job.gpus]
                del available[: job.gpus]
                pending.remove(job)
                active[executor.submit(run, job, assigned)] = (job, assigned)
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job, assigned = active.pop(future)
                results[job.name] = future.result()
                available.extend(assigned)
                available.sort(key=devices.index)
    return {job.name: results[job.name] for job in jobs}
