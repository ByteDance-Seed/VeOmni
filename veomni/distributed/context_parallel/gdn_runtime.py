# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public runtime identity and collective counters for GDN context parallelism.

The evidence surface is deliberately independent of launch scripts, environment
markers, and filesystem paths.  A caller may retain the observer exposed by a
GDN layer and serialize :meth:`GdnCpRuntimeObserver.snapshot` with its normal
logging or metrics stack.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from .gdn_lossless import GdnLosslessRuntimePlan


GdnCpImplementation = Literal["state_passing_lossless", "kcp"]


class GdnCpOperation(str, Enum):
    """Collective operations whose liveness is part of the public contract."""

    OWNERSHIP_A2A = "ownership_a2a"
    STATE_P2P_RECV = "state_p2p_recv"
    STATE_P2P_SEND = "state_p2p_send"
    HALO_P2P_RECV = "halo_p2p_recv"
    HALO_P2P_SEND = "halo_p2p_send"
    KCP_AFFINE_READY = "kcp_affine_readiness"
    KCP_AFFINE_AG = "kcp_affine_all_gather"


class GdnCpPhase(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass(frozen=True)
class GdnCpRuntimeIdentity:
    """Immutable identity for one GDN layer bound to one CP rank."""

    implementation: GdnCpImplementation
    ownership_plan_hash: str
    cp_size: int
    cp_rank: int
    layout: str = "lossless_sparse_packed"
    affine_backend: str | None = None
    native_chunk_size: int = 64

    def __post_init__(self) -> None:
        if self.implementation not in ("state_passing_lossless", "kcp"):
            raise ValueError(f"unsupported GDN CP implementation {self.implementation!r}")
        if self.cp_size < 1 or not 0 <= self.cp_rank < self.cp_size:
            raise ValueError(f"invalid CP identity: size={self.cp_size}, rank={self.cp_rank}")
        if self.implementation == "kcp" and self.affine_backend != "ttx_bc8_m1":
            raise ValueError("KCP runtime identity requires affine_backend='ttx_bc8_m1'")
        if self.implementation == "state_passing_lossless" and self.affine_backend is not None:
            raise ValueError("state_passing_lossless does not have an affine backend")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GdnCpEventCount:
    operation: str
    phase: str
    enter: int
    exit: int
    error: int


@dataclass(frozen=True)
class GdnCpRuntimeSnapshot:
    """Point-in-time, JSON-compatible evidence from one observer."""

    identity: GdnCpRuntimeIdentity
    observed_cp_ranks: tuple[int, ...]
    events: tuple[GdnCpEventCount, ...]

    @property
    def balanced(self) -> bool:
        return all(event.enter == event.exit and event.error == 0 for event in self.events)

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.as_dict(),
            "observed_cp_ranks": list(self.observed_cp_ranks),
            "events": [asdict(event) for event in self.events],
            "balanced": self.balanced,
        }


class GdnCpRuntimeObserver:
    """Small thread-safe counter owned by a live GDN layer.

    The observer does not emit logs by itself.  This keeps the model runtime
    free of environment-specific marker formats while still giving generic
    collectors a typed, stable API.
    """

    def __init__(self, identity: GdnCpRuntimeIdentity) -> None:
        self.identity = identity
        self._counts: Counter[tuple[str, str, str]] = Counter()
        self._ranks = {identity.cp_rank}
        self._lock = Lock()

    def enter(
        self,
        operation: GdnCpOperation,
        phase: GdnCpPhase,
        *,
        peer_rank: int | None = None,
    ) -> None:
        self._record(operation, phase, "enter", peer_rank=peer_rank)

    def exit(
        self,
        operation: GdnCpOperation,
        phase: GdnCpPhase,
        *,
        peer_rank: int | None = None,
    ) -> None:
        self._record(operation, phase, "exit", peer_rank=peer_rank)

    def error(
        self,
        operation: GdnCpOperation,
        phase: GdnCpPhase,
        *,
        peer_rank: int | None = None,
    ) -> None:
        self._record(operation, phase, "error", peer_rank=peer_rank)

    def _record(
        self,
        operation: GdnCpOperation,
        phase: GdnCpPhase,
        state: str,
        *,
        peer_rank: int | None,
    ) -> None:
        if peer_rank is not None and not 0 <= peer_rank < self.identity.cp_size:
            raise ValueError(f"peer rank {peer_rank} is outside CP size {self.identity.cp_size}")
        with self._lock:
            self._counts[(operation.value, phase.value, state)] += 1
            if peer_rank is not None:
                self._ranks.add(peer_rank)

    def observe_cp_ranks(self, ranks: Iterable[int]) -> None:
        """Record ranks proven to participate in a collective topology."""
        normalized = {int(rank) for rank in ranks}
        if any(rank < 0 or rank >= self.identity.cp_size for rank in normalized):
            raise ValueError(f"observed ranks {sorted(normalized)} are outside CP size {self.identity.cp_size}")
        with self._lock:
            self._ranks.update(normalized)

    def snapshot(self) -> GdnCpRuntimeSnapshot:
        with self._lock:
            keys = sorted({(operation, phase) for operation, phase, _ in self._counts})
            events = tuple(
                GdnCpEventCount(
                    operation=operation,
                    phase=phase,
                    enter=self._counts[(operation, phase, "enter")],
                    exit=self._counts[(operation, phase, "exit")],
                    error=self._counts[(operation, phase, "error")],
                )
                for operation, phase in keys
            )
            ranks = tuple(sorted(self._ranks))
        return GdnCpRuntimeSnapshot(identity=self.identity, observed_cp_ranks=ranks, events=events)


def make_gdn_cp_runtime_observer(
    implementation: GdnCpImplementation,
    *,
    plan: GdnLosslessRuntimePlan,
) -> GdnCpRuntimeObserver:
    """Create an observer bound to the validated ownership plan."""
    identity = GdnCpRuntimeIdentity(
        implementation=implementation,
        ownership_plan_hash=plan.plan_hash,
        cp_size=plan.cp_size,
        cp_rank=plan.cp_rank,
        affine_backend="ttx_bc8_m1" if implementation == "kcp" else None,
    )
    return GdnCpRuntimeObserver(identity)


__all__ = [
    "GdnCpEventCount",
    "GdnCpImplementation",
    "GdnCpOperation",
    "GdnCpPhase",
    "GdnCpRuntimeIdentity",
    "GdnCpRuntimeObserver",
    "GdnCpRuntimeSnapshot",
    "make_gdn_cp_runtime_observer",
]
