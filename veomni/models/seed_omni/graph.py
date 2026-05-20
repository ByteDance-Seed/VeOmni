"""
Shared pool data types for the SeedOmni V2 graph.

`OmniConfig` declares two parallel top-level pools:

- ``nodes``  — graph nodes (a.k.a. call-sites).  Each maps a node name to one
  ``module.method`` pair::

      {module: <name>}              # runs <name>.forward(**kwargs)
      {module: <name>.<method>}     # runs <name>.<method>(**kwargs)

  The same underlying ``nn.Module`` may appear under multiple node names — for
  example a VQ module with both an ``encode`` node (pixels → embeds) and a
  ``decode`` node (LLM hidden → CE loss / token_id → embed).  These are
  independent graph nodes that happen to share weights.

- ``edges`` — graph edges (data dependencies).  Each routes one output key
  from a source node into one kwarg of a destination node::

      {from: A, output: k, to: B, as: m}

  ``from``/``to`` reference node names; for convenience a *module name* is
  accepted when that module has exactly one active node (alias shorthand).

Reserved sink keyword
---------------------
The string ``"end"`` (exposed as the :data:`END` constant) is a virtual sink:
``to: end`` declares that a node's output flows nowhere — it just makes the
node visible to the active subset.  Every node MUST appear on at least one
edge (no orphans); leaf nodes use ``to: end`` to satisfy that invariant.

The ``end`` keyword is reserved — it cannot appear as a real node or edge
name in either pool, and it cannot appear in the ``from`` field of any edge.

Two execution views consume these pools (see ``training_graph.py`` and
``generation_graph.py``):

- ``TrainingGraph``  — DAG view.  Active nodes are derived from the endpoints
  of ``OmniConfig.training_graph.edges`` (excluding the virtual ``end`` node).
  Topological sort over those nodes produces the forward execution order.
- ``GenerationGraph`` — FSM view.  Each ``state.body`` is also a list of edge
  names; the unique nodes appearing as endpoints (excluding ``end``) execute
  once per FSM step in declaration order.

See ``design.md`` §1 for the full schema.
"""

from dataclasses import dataclass
from typing import Dict, Optional


END: str = "end"
"""Reserved virtual sink node name.

Used as ``to: end`` on edges whose source node has no real successor.  Treated
as a sentinel in graph construction; never appears in execution orders.
"""


def is_end(name: Optional[str]) -> bool:
    """Return True iff *name* refers to the virtual ``end`` sink."""
    return name == END


@dataclass
class NodeDef:
    """Parsed graph node — one ``module.method`` call-site."""

    name: str
    module: str
    method: str = "forward"

    @classmethod
    def parse(cls, name: str, spec: Dict) -> "NodeDef":
        if name == END:
            raise ValueError(f"Node name '{END}' is reserved as the virtual sink and cannot appear in `nodes`.")
        if not isinstance(spec, dict) or "module" not in spec:
            raise ValueError(f"Node '{name}': missing required `module` field. Got: {spec!r}")

        if "from" in spec or "to" in spec:
            raise ValueError(
                f"Node '{name}' is a graph node but contains edge fields (`from`/`to`). "
                "Move it into the `edges` pool, or remove the edge fields. "
                f"Got: {spec!r}"
            )

        mod_spec = spec["module"]
        explicit_method = spec.get("method")
        if "." in mod_spec:
            if explicit_method is not None:
                raise ValueError(
                    f"Node '{name}': cannot specify both dotted form `module: {mod_spec}` "
                    f"and `method: {explicit_method}`."
                )
            module, method = mod_spec.split(".", 1)
        else:
            module = mod_spec
            method = explicit_method or "forward"

        if not module or not method:
            raise ValueError(f"Node '{name}': module/method must be non-empty (got {spec!r}).")

        return cls(name=name, module=module, method=method)


@dataclass
class EdgeDef:
    """Parsed graph edge — routes ``output_key`` of ``from_`` into ``to``'s ``as_`` kwarg.

    ``to`` may be the reserved keyword :data:`END` (``"end"``) to mark the edge
    as a virtual sink.  ``from`` must always reference a real node name (or an
    unambiguous module alias resolved at TrainingGraph build time).
    """

    name: str
    from_: str
    to: str
    output_key: Optional[str] = None
    as_: Optional[str] = None

    @classmethod
    def parse(cls, name: str, spec: Dict) -> "EdgeDef":
        if name == END:
            raise ValueError(f"Edge name '{END}' is reserved and cannot appear in `edges`.")
        if not isinstance(spec, dict):
            raise ValueError(f"Edge '{name}': spec must be a dict. Got: {spec!r}")

        if "module" in spec or "method" in spec:
            raise ValueError(
                f"Edge '{name}' is a graph edge but contains node fields (`module`/`method`). "
                "Move it into the `nodes` pool, or remove the node fields. "
                f"Got: {spec!r}"
            )

        if "from" not in spec or "to" not in spec:
            raise ValueError(f"Edge '{name}' must declare both `from` and `to`. Got: {spec!r}")

        from_ = spec["from"]
        to = spec["to"]
        if from_ == END:
            raise ValueError(f"Edge '{name}': `from: {END}` is forbidden. The virtual sink may only appear on `to`.")

        return cls(
            name=name,
            from_=from_,
            to=to,
            output_key=spec.get("output"),
            as_=spec.get("as"),
        )

    def is_sink(self) -> bool:
        """True iff this edge terminates at the virtual ``end`` sink."""
        return self.to == END
