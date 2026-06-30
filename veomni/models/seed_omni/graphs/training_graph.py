"""
TrainingGraph: DAG view over a flat list of edges for training.

Schema
------
The ``training_graph`` block of ``OmniConfig`` is a *single list of edges*;
each endpoint is a self-describing ``module[.method]`` string (a bare module
defaults to ``.forward``)::

    training_graph:
      - {from: janus_siglip,             to: janus_llama}
      - {from: janus_vqvae.encode,       to: janus_llama}
      - {from: janus_text_encoder.encode, to: janus_llama}
      - {from: janus_llama,              to: janus_text_encoder.decode}
      - {from: janus_llama,              to: janus_vqvae.decode}
      - {from: janus_text_encoder.decode, to: end}   # leaf → end
      - {from: janus_vqvae.decode,       to: end}    # leaf → end

Active nodes are *derived* from the endpoints of those edges (excluding the
virtual :data:`~.graph.END` keyword); a node's identity is its canonical
``"<module>.<method>"`` form.

Every node (real, non-``end``) MUST appear on at least one edge — either as a
``from`` (data producer), as a ``to`` (data consumer), or as a leaf with
``to: end``.  This guarantees no orphans and a single, well-formed forward
queue.

Each active node executes **exactly once** per forward pass.  Edges declare
**topology only** (execution order) — there is no per-node input routing: every
node receives the same shared ``batch`` and all cross-node state flows through
the single ``conversation_list`` carrier, mutated/replaced in place as it goes.

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
* ``reset()`` / ``is_done()`` / ``current_node_name`` — cursor lifecycle
                                            (mirror of the generation FSM).
* ``step(modules, batch, ...)``         — run the node at the cursor (one
                                            forward); ``maybe_transition()``
                                            advances to the next node.
* ``to_mermaid(...)``                    — render the active DAG (``end`` is
                                            drawn as a dashed terminal node).

See also
--------
``graph.py``           — NodeDef / EdgeDef / END shared types.
``generation_graph.py`` — FSM view driven by ``OmniConfig.generation_graph``.
"""

from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Dict, List, Optional

from .dispatch import call_graph_endpoint, unwrap_graph_module
from .graph import EdgeDef, NodeDef, is_end
from .profiling import GraphProfiler


def _mermaid_id(name: str) -> str:
    """Sanitise a canonical ``module.method`` node name into a Mermaid-safe id."""
    return name.replace(".", "_").replace("-", "_")


class TrainingGraph:
    """Active training DAG derived from a flat list of edges.

    See module docstring for the full schema.
    """

    def __init__(
        self,
        edges: List[Dict],
        *,
        default_method: str = "forward",
    ):
        if not edges:
            raise ValueError(
                "TrainingGraph requires a non-empty `edges` list. "
                "Even a single-node graph must use `to: end` to make the node visible."
            )

        # Parse edges; endpoints are self-describing `module[.method]` strings.
        resolved_edges: List[EdgeDef] = []
        seen_edges: set = set()
        for spec in edges:
            edge = EdgeDef.parse(spec, default_method=default_method)
            key = (edge.from_, edge.to)
            if key in seen_edges:
                raise ValueError(f"Duplicate edge in training_graph: '{edge.name}'.")
            seen_edges.add(key)
            resolved_edges.append(edge)

        # Derive the active node set (canonical names) in first-appearance order.
        active_node_names: List[str] = []
        self._node_by_name: Dict[str, NodeDef] = {}
        for edge in resolved_edges:
            for node in (edge.from_node, edge.to_node):
                if node is None or node.name in self._node_by_name:
                    continue
                self._node_by_name[node.name] = node
                active_node_names.append(node.name)

        if not active_node_names:
            raise ValueError("training_graph yielded zero real (non-`end`) nodes — every edge points to `end`.")

        self._active_nodes: List[NodeDef] = [self._node_by_name[n] for n in active_node_names]
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

        # Runtime cursor into :attr:`_execution_order` — the training analogue of
        # the FSM's ``_current`` state. ``step`` runs the node at the cursor;
        # ``maybe_transition`` advances it. Reset before each forward pass.
        self._cursor: int = 0

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

    # ── Lifecycle (mirror of GenerationGraph) ───────────────────────────────────

    def reset(self) -> None:
        """Re-point the cursor at the first node for a fresh forward pass.

        Training analogue of :meth:`GenerationGraph.reset` (FSM → initial state).
        """
        self._cursor = 0

    def is_done(self) -> bool:
        """True once every node in :attr:`execution_order` has been stepped.

        Training analogue of :meth:`GenerationGraph.is_done` (FSM at ``done``).
        """
        return self._cursor >= len(self._execution_order)

    @property
    def current_node_name(self) -> str:
        """The node the next :meth:`step` will run (the cursor position).

        Analogue of :attr:`GenerationGraph.current_state_name`.
        """
        if self.is_done():
            raise RuntimeError("TrainingGraph.current_node_name: cursor past the last node (graph is done).")
        return self._execution_order[self._cursor]

    # ── Step & Transition (mirror of GenerationGraph) ───────────────────────────

    def step(
        self,
        modules: Dict[str, Any],
        batch: Dict[str, Any],
        *,
        profiler: Optional[GraphProfiler] = None,
        scope_fn: Optional[Callable[[str], ContextManager]] = None,
    ) -> Dict[str, Any]:
        """Run the node at the cursor — one forward (mirror of ``GenerationGraph.step``).

        Like the FSM step, this is self-contained: it resolves the module, scopes
        its :class:`ParallelState` (via ``scope_fn`` — vocab-parallel ``emb`` /
        MoE EP groups), runs ``pre_forward`` → method → ``post_forward``, and
        feeds the optional per-module trace meter. Training and inference both
        call the graph endpoint through the **wrapped** module so DDP/FSDP hooks
        fire. Non-``forward`` node methods are dispatched by temporarily
        pointing ``raw.forward`` at the target method so they still run through
        ``__call__``. ``raw`` (the unwrapped :class:`ModuleMixin`) owns the
        hooks; FSDP2 is in-place (``raw is wrapped``) while DDP wraps
        (``raw = wrapped.module``).

        Edges are pure topology — no per-node input routing. Every node receives
        the same shared ``batch``; cross-node state flows through the single
        ``conversation_list`` carrier, so the node's return dict (only ever
        ``conversation_list`` and/or ``_loss``) is merged back into ``batch`` for
        downstream nodes. Loss collection lives in :meth:`OmniModel.forward`
        (``_collect_training_loss``), mirroring ``_collect_generated``.

        Returns the (mutated) ``batch``.
        """
        node_name = self.current_node_name
        node = self._node_by_name[node_name]
        method = node.method

        wrapped = modules.get(node.module)
        if wrapped is None:
            raise KeyError(
                f"TrainingGraph.step: module '{node.module}' (node '{node_name}') missing "
                f"from modules dict. Provided: {sorted(modules)}."
            )
        raw = unwrap_graph_module(wrapped, module_name=node.module)

        module_context = scope_fn(node.module) if scope_fn is not None else nullcontext()
        profile_context = profiler.node(f"forward:{node_name}") if profiler is not None else nullcontext()
        with module_context, profile_context:
            kwargs = raw.pre_forward(method=method, **batch)

            # Opt-in trace meter (only modules multi-inheriting a TraceMixin have
            # ``trace_add``); token lengths read straight from the real inputs.
            if hasattr(raw, "trace_add"):
                raw.trace_add(method, kwargs)

            out = call_graph_endpoint(wrapped, raw, method=method, kwargs=kwargs)
            out = raw.post_forward(method=method, **out)

        batch.update(out)

        return batch

    def maybe_transition(self, *, profiler: Optional[GraphProfiler] = None) -> bool:
        """Advance the cursor to the next node (mirror of ``GenerationGraph.maybe_transition``).

        Training's "transition" is unconditional: a static topological pass just
        moves to the next node. Returns ``True`` while nodes remain, ``False``
        once the cursor steps past the last one (``is_done()``).
        """
        self._cursor += 1
        moved = not self.is_done()
        if profiler is not None and moved:
            profiler.record(f"transition: -> {self.current_node_name}")
        return moved

    # ── visualization ────────────────────────────────────────────────────────

    def to_mermaid(
        self,
        title: Optional[str] = None,
    ) -> str:
        """Render the active training DAG as Mermaid flowchart syntax.

        Each graph node is labelled ``<node_name><br/><module>.<method>`` so
        multiple nodes of the same module are visually distinct.  Edges with
        ``to: end`` render as dashed arrows into a single ``end`` terminal.

        A dashed ``data`` pseudo-node fans out to every source node to mark
        inputs that come from the shared batch dict (``raw_batch`` at runtime).

        Layout
        ------
        Direction is ``LR`` (left → right) and active nodes are bucketed into
        invisible per-rank subgraphs by topological depth.  This forces the
        renderer to lay out encoders / backbone / heads as parallel columns
        rather than zig-zagging across the canvas.  The ELK renderer
        (``defaultRenderer: elk``) is requested up-front so edges route
        orthogonally with mostly straight segments — closer to what one
        would draw by hand for a multi-modal training pipeline.

        Single-loss protocol
        --------------------
        Each module collects its own scalar ``_loss`` (token-mean over its
        own micro-batches); ``OmniModel`` sums them.  There is no central
        ``losses`` collector, so the diagram only carries real data-flow
        edges — no dashed fan-in to a pseudo loss node.

        Parameters
        ----------
        title:
            Optional Mermaid ``title:`` directive.
        """
        lines: List[str] = []
        if title:
            lines += ["---", f"title: {title}", "---"]
        # Request the ELK renderer for orthogonal, mostly-straight edge routing.
        lines.append("%%{init: {'flowchart': {'defaultRenderer': 'elk'}}}%%")
        lines.append("flowchart LR")

        sources = set(self.sources)
        sinks = set(self.sinks)

        # Topological depth per active node — the rank used for column banding.
        # Sources sit at depth 0; every other node is one beyond its deepest
        # active predecessor.  Edges into the virtual `end` don't count.
        depth: Dict[str, int] = {}
        active_predecessors: Dict[str, List[str]] = defaultdict(list)
        for e in self._active_edges:
            if not is_end(e.to):
                active_predecessors[e.to].append(e.from_)
        for n in self._execution_order:
            preds = active_predecessors.get(n, [])
            depth[n] = 0 if not preds else 1 + max(depth[p] for p in preds)

        rank_to_nodes: Dict[int, List[str]] = defaultdict(list)
        for n in self._execution_order:
            rank_to_nodes[depth[n]].append(n)
        ranks = sorted(rank_to_nodes)

        if sources:
            lines.append("    data[(data)]:::io")

        # One invisible subgraph per topological rank — the renderer keeps each
        # rank as a vertical stack, and the columns line up left-to-right.
        for r in ranks:
            lines.append(f"    subgraph col{r} [ ]")
            lines.append("        direction TB")
            for n_name in rank_to_nodes[r]:
                n = self._node_by_name[n_name]
                label = f"<i>{n.module}.{n.method}</i>"
                cls = (
                    "both"
                    if n.name in sources and n.name in sinks
                    else "source"
                    if n.name in sources
                    else "sink"
                    if n.name in sinks
                    else "middle"
                )
                lines.append(f'        {_mermaid_id(n.name)}["{label}"]:::{cls}')
            lines.append("    end")

        has_end = any(is_end(e.to) for e in self._active_edges)
        if has_end:
            lines.append('    end_sink(("end")):::end_sink')

        if sources:
            for n in sorted(sources):
                lines.append(f"    data -.-> {_mermaid_id(n)}")

        for e in self._active_edges:
            label = self._edge_label(e)
            arrow = f"-->|{label}|" if label else "-->"
            target = "end_sink" if is_end(e.to) else _mermaid_id(e.to)
            lines.append(f"    {_mermaid_id(e.from_)} {arrow} {target}")

        # Hide the rank-banding subgraph borders — they only constrain layout.
        for r in ranks:
            lines.append(f"    style col{r} fill:transparent,stroke:none")

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

    @staticmethod
    def _edge_label(e: EdgeDef) -> str:
        """Render a body edge label (topology-only — no field routing)."""
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
