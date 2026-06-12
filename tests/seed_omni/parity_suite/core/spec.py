"""Declarative case schema for SeedOmni parity tests."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def expand_env(value: Any) -> Any:
    """Expand `${VAR}` placeholders in string values.

    Missing variables expand to an empty string so env-gated cases skip with a
    clear missing-input reason instead of leaking unresolved placeholders into
    file paths.
    """

    if not isinstance(value, str):
        return value
    return _ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), ""), value)


@dataclass(frozen=True)
class BackendSpec:
    type: str = "transformers"
    model: str | None = None
    trust_remote_code: bool = False
    local_transformers: str | bool | None = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> BackendSpec:
        values = dict(data or {})
        known = {
            "type": values.pop("type", "transformers"),
            "model": expand_env(values.pop("model", None)),
            "trust_remote_code": bool(values.pop("trust_remote_code", False)),
            "local_transformers": values.pop("local_transformers", "auto"),
        }
        return cls(**known, extra=values)


@dataclass(frozen=True)
class V2ModelSpec:
    model_root: str | None = None
    config_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> V2ModelSpec:
        values = dict(data or {})
        known = {
            "model_root": expand_env(values.pop("model_root", None)),
            "config_dir": expand_env(values.pop("config_dir", None)),
        }
        return cls(**known, extra=values)


@dataclass(frozen=True)
class EnvSpec:
    prefix: str = ""
    enable: str | None = None
    requires_cuda: bool = False
    min_cuda_devices: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EnvSpec:
        values = dict(data or {})
        return cls(
            prefix=str(values.get("prefix", "")),
            enable=values.get("enable"),
            requires_cuda=bool(values.get("requires_cuda", False)),
            min_cuda_devices=int(values.get("min_cuda_devices", 0) or 0),
        )

    def name(self, suffix: str | None) -> str | None:
        if not suffix:
            return None
        return suffix if suffix.startswith(self.prefix) else f"{self.prefix}{suffix}"

    def value(self, suffix: str | None) -> str | None:
        name = self.name(suffix)
        return None if name is None else os.environ.get(name)

    def flag_enabled(self) -> bool:
        if not self.enable:
            return True
        value = self.value(self.enable)
        return value is not None and value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CaseSpec:
    model_name: str
    id: str
    domain: str
    level: str
    case: dict[str, Any]
    source_path: Path
    reference_backend: BackendSpec
    v2_model: V2ModelSpec
    env: EnvSpec
    adapter: str
    probes_module: str | None
    captures_module: str | None
    probes: tuple[str, ...] = ()
    graph: str | None = None
    fixture: str | None = None
    capture: str | None = None
    reference: str | None = None
    category: str = "reference_parity"

    @property
    def node_id(self) -> str:
        return f"{self.model_name}.{self.domain}.{self.level}.{self.id}"

    @property
    def fixture_path(self) -> str | None:
        if not self.fixture:
            return None
        env_value = self.env.value(self.fixture)
        if env_value:
            return env_value
        expanded = expand_env(self.fixture)
        if expanded != self.fixture or Path(expanded).exists():
            return expanded
        return None

    @property
    def requires_fixture(self) -> bool:
        return bool(self.case.get("requires_fixture", False))

    @property
    def requires_v2_model(self) -> bool:
        return self.level in {"module", "graph", "trainer", "fsdp"} or self.case.get("requires_v2_model", False)

    @property
    def requires_reference_model(self) -> bool:
        return (
            self.fixture_path is None
            and self.reference not in {"v2_graph", "none"}
            and self.case.get("requires_reference_model", True)
        )

    def required_env_missing(self) -> list[str]:
        missing: list[str] = []
        if self.requires_fixture and self.fixture and not self.fixture_path:
            missing.append(self.env.name(self.fixture) or self.fixture)
        if self.requires_v2_model and not self.v2_model.model_root:
            missing.append(self.env.name("SPLIT_MODEL_ROOT") or "SPLIT_MODEL_ROOT")
        if self.requires_v2_model and self.v2_model.model_root and not Path(self.v2_model.model_root).exists():
            missing.append(f"{self.env.name('SPLIT_MODEL_ROOT') or 'SPLIT_MODEL_ROOT'} (path does not exist)")
        if self.fixture_path is None and self.requires_reference_model and not self.capture:
            missing.append("capture")
        if (
            self.requires_reference_model
            and self.reference_backend.type == "transformers"
            and not self.reference_backend.model
        ):
            missing.append(self.env.name("REFERENCE_MODEL") or "REFERENCE_MODEL")
        if (
            self.requires_reference_model
            and self.reference_backend.type == "transformers"
            and self.reference_backend.model
            and not Path(self.reference_backend.model).exists()
        ):
            missing.append(f"{self.env.name('REFERENCE_MODEL') or 'REFERENCE_MODEL'} (path does not exist)")
        for suffix in self.case.get("requires_env", []) or []:
            if not self.env.value(str(suffix)):
                missing.append(self.env.name(str(suffix)) or str(suffix))
        return missing

    def static_skip_reason(self) -> str | None:
        if not self.env.flag_enabled():
            enable_name = self.env.name(self.env.enable) or str(self.env.enable)
            return f"Set {enable_name}=1 to run {self.node_id}."
        missing = self.required_env_missing()
        if missing:
            return f"Missing required env/config for {self.node_id}: {', '.join(missing)}"
        return None
