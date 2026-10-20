"""Compatibility interface for callers using the original GPU job names."""

from dataclasses import dataclass

from scripts.ci.device_pool import DeviceJob
from scripts.ci.device_pool import run_jobs as run_device_jobs


@dataclass(frozen=True)
class GPUJob:
    name: str
    gpus: int
    payload: object


def run_jobs(jobs, devices, run):
    if any(not 1 <= job.gpus <= len(devices) for job in jobs):
        raise ValueError("Job GPU reservation exceeds the available devices")
    original = {job.name: job for job in jobs}
    return run_device_jobs(
        [DeviceJob(job.name, job.gpus, job.payload) for job in jobs],
        devices,
        lambda job, assigned: run(original[job.name], assigned),
    )
