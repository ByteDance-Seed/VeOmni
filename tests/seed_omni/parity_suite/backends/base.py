"""Backend interface for reference model runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tests.seed_omni.parity_suite.core.spec import BackendSpec, CaseSpec


@dataclass
class LoadedBackend:
    model: Any
    tokenizer: Any | None = None
    processor: Any | None = None


class ReferenceBackend:
    def __init__(self, spec: BackendSpec, case: CaseSpec, *, device: str | torch.device = "cpu") -> None:
        self.spec = spec
        self.case = case
        self.device = torch.device(device)

    def load(self) -> LoadedBackend:
        raise NotImplementedError
