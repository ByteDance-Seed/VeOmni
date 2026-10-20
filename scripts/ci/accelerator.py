"""Device visibility adapters; the queue itself has no GPU/NPU branches."""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from scripts.ci.launch_plan import ResourcePlanError, RunnerCapabilities


def child_environment(capabilities, assigned, *, cpu=False, environ=None):
    env = dict(os.environ if environ is None else environ)
    if len(set(assigned)) != len(assigned) or not set(assigned) <= set(capabilities.devices):
        raise ResourcePlanError("Device assignment is outside the runner's owned slice")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    if cpu or capabilities.backend == "cpu":
        env.update(
            CUDA_VISIBLE_DEVICES="",
            ASCEND_VISIBLE_DEVICES="",
            ASCEND_RT_VISIBLE_DEVICES="",
            TORCH_DEVICE_BACKEND_AUTOLOAD="0",
            PYTORCH_HAL_BASED_NPU_CHECK="1",
        )
    elif not assigned:
        raise ResourcePlanError("An accelerator worker needs a nonempty device lease")
    elif capabilities.backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
    elif capabilities.backend == "npu":
        ids = [int(value) for value in assigned]
        if ids != sorted(ids):
            raise ResourcePlanError("Ascend device IDs must be in ascending order")
        env["ASCEND_VISIBLE_DEVICES"] = env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(assigned)
        env.pop("ASCEND_DEVICE_ID", None)
        env.setdefault("HCCL_HOST_SOCKET_PORT_RANGE", "auto")
        env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    return env


def probe(output, requested="auto"):
    """Probe in a short-lived process, before collection or leased workers exist."""
    from veomni.utils.device import get_device_type, get_torch_device

    backend = get_device_type()
    if requested != "auto" and requested != backend:
        raise ResourcePlanError(f"Requested {requested}, detected {backend}")
    if backend == "cpu":
        caps = RunnerCapabilities("cpu")
    else:
        device = get_torch_device()
        count = device.device_count() if device.is_available() else 0
        if count < 1:
            raise ResourcePlanError(f"No available {backend} devices")
        key = "CUDA_VISIBLE_DEVICES" if backend == "cuda" else "ASCEND_RT_VISIBLE_DEVICES"
        visible = os.environ.get(key)
        if visible is None and backend == "npu":
            visible = os.environ.get("ASCEND_VISIBLE_DEVICES")
        devices = tuple(visible.split(",")) if visible is not None else tuple(str(i) for i in range(count))
        if len(devices) != count or any(not value for value in devices):
            raise ResourcePlanError(f"Visibility {devices} disagrees with runtime device count {count}")
        caps = RunnerCapabilities(backend, devices)
    Path(output).write_text(json.dumps(asdict(caps), indent=2) + "\n")


if __name__ == "__main__":
    probe(*sys.argv[1:])
