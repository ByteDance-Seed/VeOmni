"""Structured reports for generated parity cases."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParityReport:
    case_id: str
    category: str
    all_pass: bool
    probes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "all_pass": self.all_pass,
            "metadata": self.metadata,
            "probes": self.probes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def __str__(self) -> str:
        return self.to_json()
