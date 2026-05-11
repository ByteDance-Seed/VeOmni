"""
OmniGraph: derives the module execution order for the training DAG.

A *connection* is a named, atomic edge definition stored in ``OmniConfig.connections``.
Two forms are supported:

  ``{module: X}``
      Execute module X.  No data is explicitly routed; X reads directly from
      the global raw-batch context (all previous module outputs are also
      available).

  ``{from: A, output: k, to: B, as: m}``
      Take key ``k`` from module A's output dict and pass it to module B as
      kwarg ``m``, then execute B.

``OmniGraph`` is given the full connections pool plus the *active* list for the
current training run (``training_graph.connections``).  It builds a DAG,
performs a topological sort, and exposes:

  - ``execution_order`` — list of module names in dependency order.
  - ``collect_inputs(module_name, module_outputs, raw_batch)`` — merge raw
    batch + connection-routed tensors for a given module.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ConnectionDef:
    """Parsed representation of a single named connection."""

    name: str
    # {module: X} form
    module: Optional[str] = None
    # {from: A, output: k, to: B, as: m} form
    from_: Optional[str] = None
    output_key: Optional[str] = None
    to: Optional[str] = None
    as_: Optional[str] = None

    @classmethod
    def parse(cls, name: str, spec: Dict) -> "ConnectionDef":
        if "module" in spec:
            return cls(name=name, module=spec["module"])
        elif "from" in spec:
            return cls(
                name=name,
                from_=spec["from"],
                output_key=spec.get("output"),
                to=spec["to"],
                as_=spec.get("as"),
            )
        else:
            raise ValueError(
                f"Connection '{name}' must have either 'module' or 'from'/'to' keys. Got: {spec}"
            )

    @property
    def target_module(self) -> Optional[str]:
        """The module that will be *executed* by this connection."""
        return self.module or self.to

    @property
    def source_module(self) -> Optional[str]:
        """The module whose output is consumed by this connection (if any)."""
        return self.from_


class OmniGraph:
    """Training DAG built from the active subset of named connections.

    Parameters
    ----------
    connections:
        Full pool of named connection definitions (``OmniConfig.connections``).
    training_connections:
        Ordered list of connection names that are active for this training run
        (``OmniConfig.training_graph.connections``).

    Topological sort
    ----------------
    Two modules are ordered when there exists a data-transfer connection
    ``{from: A, to: B}`` between them — A must execute before B.
    ``{module: X}`` connections contribute X to the module set but carry no
    ordering constraint beyond what other connections impose.
    """

    def __init__(self, connections: Dict[str, Dict], training_connections: List[str]):
        self._pool: Dict[str, ConnectionDef] = {
            n: ConnectionDef.parse(n, spec) for n, spec in connections.items()
        }
        self._active: List[ConnectionDef] = [self._pool[n] for n in training_connections]
        self._execution_order: List[str] = self._topological_sort()

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def execution_order(self) -> List[str]:
        """Module names in dependency-safe execution order."""
        return list(self._execution_order)

    def collect_inputs(
        self,
        module_name: str,
        module_outputs: Dict[str, Dict[str, Any]],
        raw_batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the ``**kwargs`` dict for *module_name*'s forward call.

        Priority (last wins):
          1. raw_batch (globally transparent to every module).
          2. Connection-routed values from earlier modules' outputs.

        Parameters
        ----------
        module_name:
            The module about to be executed.
        module_outputs:
            ``{module_name: output_dict}`` for all already-executed modules.
        raw_batch:
            The original training batch dict.
        """
        kwargs: Dict[str, Any] = dict(raw_batch)
        for conn in self._active:
            if conn.to == module_name and conn.from_ is not None:
                src_out = module_outputs.get(conn.from_)
                if src_out is None:
                    continue
                val = src_out.get(conn.output_key)
                if val is not None and conn.as_ is not None:
                    kwargs[conn.as_] = val
        return kwargs

    def active_connections(self) -> List[ConnectionDef]:
        return list(self._active)

    # ── internal ──────────────────────────────────────────────────────────────

    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm over the module dependency graph."""
        # Collect all modules mentioned in active connections
        all_modules: set = set()
        for conn in self._active:
            if conn.module:
                all_modules.add(conn.module)
            if conn.from_:
                all_modules.add(conn.from_)
            if conn.to:
                all_modules.add(conn.to)

        # Build dependency map: module → set of modules it depends on
        deps: Dict[str, set] = defaultdict(set)
        for conn in self._active:
            if conn.from_ and conn.to:
                deps[conn.to].add(conn.from_)

        in_degree: Dict[str, int] = {m: len(deps[m]) for m in all_modules}
        queue: deque = deque(sorted(m for m in all_modules if in_degree[m] == 0))
        order: List[str] = []

        while queue:
            m = queue.popleft()
            order.append(m)
            for other in sorted(all_modules):
                if m in deps[other]:
                    deps[other].discard(m)
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        if len(order) != len(all_modules):
            cycle = all_modules - set(order)
            raise ValueError(
                f"Circular dependency detected in training_graph connections. "
                f"Involved modules: {cycle}"
            )

        return order
