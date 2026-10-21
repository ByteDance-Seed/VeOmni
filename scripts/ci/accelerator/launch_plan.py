"""Pure launch descriptions shared by test collection and execution."""

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping


class ResourcePlanError(RuntimeError):
    """A launch cannot run inside its declared device lease."""


@dataclass(frozen=True)
class RunnerCapabilities:
    backend: str
    devices: tuple[str, ...] = ()

    def __post_init__(self):
        if self.backend not in ("cpu", "cuda", "npu"):
            raise ResourcePlanError(f"Unsupported accelerator: {self.backend}")
        if len(set(self.devices)) != len(self.devices):
            raise ResourcePlanError("Runner devices must be distinct")
        if self.backend == "cpu" and self.devices:
            raise ResourcePlanError("CPU runners cannot own accelerator devices")


@dataclass(frozen=True)
class LaunchSpec:
    nproc: int = 1
    device: str = "accelerator"
    nnodes: int = 1

    def __post_init__(self):
        if type(self.nproc) is not int or self.nproc < 1:
            raise ResourcePlanError("Local nproc must be a positive integer")
        if self.nnodes != 1:
            raise ResourcePlanError("CI launch plans support one node")
        if self.device not in ("cpu", "accelerator"):
            raise ResourcePlanError(f"Unsupported launch device: {self.device}")

    def cards(self, capabilities):
        return self.nproc if self.device == "accelerator" and capabilities.backend != "cpu" else 0


@dataclass(frozen=True)
class CasePlan:
    """Sequential groups of concurrent launches; distributed workers stay fresh."""

    groups: tuple[tuple[LaunchSpec, ...], ...]

    def __post_init__(self):
        if not self.groups or any(not group for group in self.groups):
            raise ResourcePlanError("A case plan needs nonempty launch groups")

    @classmethod
    def sequential(cls, *launches):
        return cls(tuple((launch,) for launch in launches))

    @classmethod
    def concurrent(cls, *launches):
        return cls((tuple(launches),))

    @property
    def launches(self):
        return tuple(spec for group in self.groups for spec in group)

    def cards(self, capabilities):
        return max(sum(spec.cards(capabilities) for spec in group) for group in self.groups)

    def to_dict(self):
        return {"groups": [[asdict(spec) for spec in group] for group in self.groups]}

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(tuple(LaunchSpec(**spec) for spec in group) for group in value["groups"]))


@dataclass(frozen=True)
class PlanContext:
    params: Mapping
    capabilities: RunnerCapabilities
    instance: object = None


def from_parameter(name, *, device="accelerator"):
    """Use the very parameter passed to the launcher's nproc/world_size argument."""

    def resolve(context):
        return LaunchSpec(nproc=context.params[name], device=device)

    return resolve


def as_case_plan(value):
    return CasePlan.sequential(value) if isinstance(value, LaunchSpec) else value


@lru_cache
def _read_capabilities(path):
    value = json.loads(Path(path).read_text())
    return RunnerCapabilities(value["backend"], tuple(value["devices"]))


def runner_capabilities():
    """Return immutable owned capacity, independently of the current free subset."""
    path = os.environ.get("VEOMNI_CI_CAPABILITIES")
    if path:
        return _read_capabilities(path)
    from veomni.utils.device import get_device_type, get_torch_device

    backend = get_device_type()
    if backend == "cpu":
        return RunnerCapabilities("cpu")
    device = get_torch_device()
    count = device.device_count() if device.is_available() else 0
    return RunnerCapabilities(backend, tuple(str(i) for i in range(count)))


def runner_device_count():
    """Capacity for eligibility and launch configuration, never for device indexing."""
    return len(runner_capabilities().devices)


def default_plan(capabilities):
    return CasePlan.sequential(LaunchSpec(device="cpu" if capabilities.backend == "cpu" else "accelerator"))
