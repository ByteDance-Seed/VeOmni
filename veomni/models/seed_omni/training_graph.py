"""
TrainingGraph: DAG view over the ``nodes`` / ``edges`` pools for training.

Schema
------
The ``training_graph`` block of ``OmniConfig`` is a *single list of edge
names*::

    training_graph:
      edges:
        - vision_to_ar
        - vae_enc_to_ar
        - tok_enc_to_ar
        - ar_to_tok_decode
        - ar_to_vq_decode
        - vq_token_to_decode
        - tok_decode_sink     # leaf node → end
        - vq_decode_sink      # leaf node → end

Active nodes are *derived* from the endpoints of those edges (excluding the
virtual :data:`~.graph.END` keyword).  The pool may contain inference-only
items — they are simply ignored when computing the training subset.

Every node (real, non-``end``) MUST appear on at least one edge — either as a
``from`` (data producer), as a ``to`` (data consumer), or as a leaf with
``to: end``.  This guarantees no orphans and a single, well-formed forward
queue.

Each active node executes **exactly once** per forward pass.  Multiple edges
into the same node fan in (their routed values merge into one kwargs dict).
Multiple edges out of the same node fan out (downstream nodes share the same
single output dict).

Single-loss protocol
--------------------
Each module's :meth:`OmniModule.forward` returns at most one ``_loss`` key
(scalar, already token-mean-reduced across all micro-batches).  ``OmniModel``
sums them — see ``modeling_omni.py``.

Exposed surface
---------------
* ``execution_order``                   — node names in topological order.
* ``active_nodes()`` / ``active_edges()`` — typed definitions, in topo / pool
                                            order respectively.
* ``sources`` / ``sinks``               — nodes with no incoming / no
                                            non-``end`` outgoing active edge.
* ``module_of(node)`` / ``method_of(node)`` — accessors.
* ``collect_inputs(node, outputs, raw_batch)`` — assemble forward kwargs.
* ``to_mermaid(...)``                    — render the active DAG (``end`` is
                                            drawn as a dashed terminal node).

See also
--------
``graph.py``           — NodeDef / EdgeDef / END shared pool types.
``generation_graph.py`` — FSM view driven by ``OmniConfig.generation_graph``.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from .graph import END, EdgeDef, NodeDef, is_end


class TrainingGraph:
    """Active training DAG over named graph nodes and edges.

    See module docstring for the full schema.
    """

    def __init__(
        self,
        nodes: Dict[str, Dict],
        edges: Dict[str, Dict],
        training_edges: List[str],
    ):
        if not training_edges:
            raise ValueError(
                "TrainingGraph requires a non-empty `training_edges` list. "
                "Even a single-node graph must use `to: end` to make the node visible."
            )

        nodes = nodes or {}
        edges = edges or {}

        # Name-collision guard: nodes & edges share a single namespace
        # (FSM body looks up names in both — clashes silently misroute).
        overlap = set(nodes) & set(edges)
        if overlap:
            raise ValueError(
                f"`nodes` and `edges` pools share name(s): {sorted(overlap)}. "
                "Each name must be unique across both pools."
            )
        if END in nodes or END in edges:
            raise ValueError(f"The reserved keyword '{END}' cannot appear as a node or edge name.")

        self._node_pool: Dict[str, NodeDef] = {n: NodeDef.parse(n, spec) for n, spec in nodes.items()}
        self._edge_pool: Dict[str, EdgeDef] = {n: EdgeDef.parse(n, spec) for n, spec in edges.items()}

        # Resolve training_edges
        missing_edges = [n for n in training_edges if n not in self._edge_pool]
        if missing_edges:
            raise KeyError(
                f"training_edges references undefined edge name(s): {missing_edges}. "
                f"Known edges: {sorted(self._edge_pool)}"
            )

        seen_edges: set = set()
        for ename in training_edges:
            if ename in seen_edges:
                raise ValueError(f"Duplicate edge name in training_edges: '{ename}'.")
            seen_edges.add(ename)

        # Index module → nodes for alias resolution.
        self._nodes_by_module: Dict[str, List[NodeDef]] = defaultdict(list)
        for n in self._node_pool.values():
            self._nodes_by_module[n.module].append(n)

        # Resolve edges & derive active node set
        active_node_names: List[str] = []  # in first-appearance order
        active_node_set: set = set()
        resolved_edges: List[EdgeDef] = []
        for ename in training_edges:
            raw = self._edge_pool[ename]
            from_ = self._resolve_endpoint(raw.from_, edge_name=raw.name, side="from", allow_end=False)
            to = self._resolve_endpoint(raw.to, edge_name=raw.name, side="to", allow_end=True)
            resolved_edges.append(
                EdgeDef(
                    name=raw.name,
                    from_=from_,
                    to=to,
                    output_key=raw.output_key,
                    as_=raw.as_,
                )
            )
            for endpoint in (from_, to):
                if not is_end(endpoint) and endpoint not in active_node_set:
                    active_node_set.add(endpoint)
                    active_node_names.append(endpoint)

        if not active_node_names:
            raise ValueError("training_edges yielded zero real (non-`end`) nodes — every edge points to `end`.")

        self._active_nodes: List[NodeDef] = [self._node_pool[n] for n in active_node_names]
        self._node_by_name: Dict[str, NodeDef] = {n.name: n for n in self._active_nodes}
        self._active_edges: List[EdgeDef] = resolved_edges

        # Sanity: every active node has *some* outgoing edge (forbidding orphans
        # is the whole point of the `to: end` keyword).  By construction every
        # active node appears on at least one edge endpoint, so this is always
        # satisfied — but assert defensively.
        producers = {e.from_ for e in self._active_edges}
        consumers = {e.to for e in self._active_edges if not is_end(e.to)}
        touched = producers | consumers
        for n in active_node_names:
            assert n in touched, f"Internal: derived node '{n}' missing from any edge endpoint."

        self._execution_order: List[str] = self._topological_sort()

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def execution_order(self) -> List[str]:
        """Active node names in dependency-safe execution order. Excludes ``end``."""
        return list(self._execution_order)

    @property
    def sources(self) -> List[str]:
        """Active nodes with no incoming active edge — they only see ``raw_batch``."""
        targets = {e.to for e in self._active_edges if not is_end(e.to)}
        return [n for n in self._execution_order if n not in targets]

    @property
    def sinks(self) -> List[str]:
        """Active nodes whose only outgoing edges go to the virtual ``end``."""
        producers_to_real = {e.from_ for e in self._active_edges if not is_end(e.to)}
        return [n for n in self._execution_order if n not in producers_to_real]

    def active_nodes(self) -> List[NodeDef]:
        """Active node definitions in topological order."""
        return [self._node_by_name[n] for n in self._execution_order]

    def active_edges(self) -> List[EdgeDef]:
        """Active edges in declaration order (``end``-targeted edges included)."""
        return list(self._active_edges)

    def module_of(self, node: str) -> str:
        n = self._node_by_name.get(node)
        if n is None:
            raise KeyError(f"'{node}' is not an active node. Active: {sorted(self._node_by_name)}.")
        return n.module

    def method_of(self, node: str) -> str:
        n = self._node_by_name.get(node)
        if n is None:
            raise KeyError(f"'{node}' is not an active node. Active: {sorted(self._node_by_name)}.")
        return n.method

    def collect_inputs(
        self,
        node: str,
        outputs: Dict[str, Dict[str, Any]],
        raw_batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the kwargs dict for ``node``'s forward call.

        Priority (last wins):

        1. ``raw_batch`` — globally transparent to every node.
        2. Edge-routed values from already-executed predecessor nodes.

        ``outputs`` is keyed by **node name**.
        """
        kwargs: Dict[str, Any] = dict(raw_batch)
        for e in self._active_edges:
            if e.to != node:
                continue
            src_out = outputs.get(e.from_)
            if src_out is None:
                continue
            val = src_out.get(e.output_key)
            if val is not None and e.as_ is not None:
                kwargs[e.as_] = val
        return kwargs

    # ── pool accessors (used by FSM) ─────────────────────────────────────────

    @property
    def node_pool(self) -> Dict[str, NodeDef]:
        return dict(self._node_pool)

    @property
    def edge_pool(self) -> Dict[str, EdgeDef]:
        return dict(self._edge_pool)

    # ── visualization ────────────────────────────────────────────────────────

    def to_mermaid(
        self,
        show_io: bool = True,
        title: Optional[str] = None,
    ) -> str:
        """Render the active training DAG as Mermaid flowchart syntax.

        Each graph node is labelled ``<node_name><br/><module>.<method>`` so
        multiple nodes of the same module are visually distinct.  Edges with
        ``to: end`` render as dashed arrows into a single ``end`` terminal.

        Parameters
        ----------
        show_io:
            When True (default), draws a dashed ``raw_batch`` pseudo-node
            feeding every source node, plus a dashed ``losses`` pseudo-node
            collecting from every active node (each module may emit at most
            one ``_loss`` scalar).
        title:
            Optional Mermaid ``title:`` directive.
        """
        lines: List[str] = []
        if title:
            lines += ["---", f"title: {title}", "---"]
        lines.append("flowchart TD")

        sources = set(self.sources)
        sinks = set(self.sinks)

        if show_io:
            lines.append("    raw_batch[(raw batch)]:::io")
            lines.append("    losses[(losses)]:::io")

        for n in self.active_nodes():
            label = f"{n.name}<br/><i>{n.module}.{n.method}</i>"
            cls = (
                "both"
                if n.name in sources and n.name in sinks
                else "source"
                if n.name in sources
                else "sink"
                if n.name in sinks
                else "middle"
            )
            lines.append(f'    {n.name}["{label}"]:::{cls}')

        # Virtual end sink (only drawn when at least one edge targets it).
        has_end = any(is_end(e.to) for e in self._active_edges)
        if has_end:
            lines.append('    end_sink(("end")):::end_sink')

        if show_io:
            for n in sorted(sources):
                lines.append(f"    raw_batch -.-> {n}")

        for e in self._active_edges:
            label = self._edge_label(e)
            arrow = f"-->|{label}|" if label else "-->"
            target = "end_sink" if is_end(e.to) else e.to
            lines.append(f"    {e.from_} {arrow} {target}")

        if show_io:
            for n in self._execution_order:
                lines.append(f"    {n} -.-> losses")

        lines += [
            "    classDef source fill:#dff,stroke:#06c,stroke-width:2px",
            "    classDef sink fill:#fdd,stroke:#c06,stroke-width:2px",
            "    classDef both fill:#efe,stroke:#693,stroke-width:2px",
            "    classDef middle fill:#fff,stroke:#666",
            "    classDef io fill:#eef,stroke:#669,stroke-dasharray:3 3",
            "    classDef end_sink fill:#eee,stroke:#333,stroke-width:1px,stroke-dasharray:3 3",
        ]
        return "\n".join(lines)

    # ── internal ─────────────────────────────────────────────────────────────

    def _resolve_endpoint(
        self,
        name: Optional[str],
        edge_name: str,
        side: str,
        *,
        allow_end: bool,
    ) -> str:
        """Resolve an edge endpoint string to an active node name.

        Accepts:
        * an exact node name from the pool;
        * a *module name* when that module has exactly one declared node;
        * the reserved keyword :data:`~.graph.END` when ``allow_end`` is True.
        """
        if name is None:
            raise ValueError(f"Edge '{edge_name}': missing `{side}`.")
        if name == END:
            if not allow_end:
                raise ValueError(
                    f"Edge '{edge_name}': `{side}: {END}` is forbidden — the virtual sink may only appear on `to`."
                )
            return END
        if name in self._node_pool:
            return name
        candidates = self._nodes_by_module.get(name, [])
        if len(candidates) == 1:
            return candidates[0].name
        if len(candidates) > 1:
            names = sorted(c.name for c in candidates)
            raise ValueError(
                f"Edge '{edge_name}': `{side}: {name}` is ambiguous — module '{name}' "
                f"has multiple declared nodes: {names}. Use a node name."
            )
        raise KeyError(
            f"Edge '{edge_name}': `{side}: {name}` references neither a declared node "
            f"nor a uniquely-declared module. Declared nodes: {sorted(self._node_pool)}."
        )

    @staticmethod
    def _edge_label(e: EdgeDef) -> str:
        if e.output_key and e.as_ and e.output_key != e.as_:
            return f'"{e.output_key} → {e.as_}"'
        if e.output_key:
            return f'"{e.output_key}"'
        return ""

    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm over the active-node dependency graph (excluding ``end``)."""
        nodes: List[str] = [n.name for n in self._active_nodes]
        deps: Dict[str, set] = defaultdict(set)
        for e in self._active_edges:
            if is_end(e.to):
                continue
            deps[e.to].add(e.from_)

        in_degree: Dict[str, int] = {n: len(deps[n]) for n in nodes}
        # Sort by user-declared order (active_nodes preserves first-appearance).
        order_index = {n: i for i, n in enumerate(nodes)}
        queue: deque = deque(sorted((n for n in nodes if in_degree[n] == 0), key=order_index.get))
        order: List[str] = []

        while queue:
            n = queue.popleft()
            order.append(n)
            # Process in declaration order for stability.
            for other in sorted(nodes, key=order_index.get):
                if n in deps[other]:
                    deps[other].discard(n)
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        if len(order) != len(nodes):
            cycle = sorted(set(nodes) - set(order))
            raise ValueError(
                f"Circular dependency in active nodes: {cycle}. "
                "The same module may back multiple nodes, but the active dependency graph must be acyclic."
            )

        return order
