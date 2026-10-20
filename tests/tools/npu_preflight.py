"""Fail Ascend CI early when its workspace runtime cannot use every NPU."""

from __future__ import annotations

import argparse


def check_npu_runtime(expected_devices: int = 8) -> int:
    if expected_devices < 1:
        raise ValueError("expected_devices must be positive")

    import torch
    import torch_npu  # noqa: F401 - registers the device backend

    if not torch.npu.is_available():
        raise RuntimeError("Ascend CI requires an available torch NPU runtime")
    count = torch.npu.device_count()
    if count != expected_devices:
        raise RuntimeError(f"Ascend CI requires exactly {expected_devices} visible NPUs, found {count}")
    for index in range(count):
        torch.npu.set_device(index)
        value = torch.ones(1, device=f"npu:{index}").add_(1).mul_(3)
        torch.npu.synchronize()
        if value.device.type != "npu" or value.item() != 6.0:
            raise RuntimeError(f"Ascend CI compute/readback failed on NPU {index}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-devices", type=int, default=8)
    args = parser.parse_args()
    count = check_npu_runtime(args.expected_devices)
    print(f"NPU_RUNTIME_PREFLIGHT_OK devices={count}", flush=True)
