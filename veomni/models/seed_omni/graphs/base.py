"""
Shared data types for the SeedOmni V2 graph.

``OmniConfig`` no longer declares separate ``nodes`` / ``edges`` pools.  Both
the training DAG (``training_graph``) and the inference FSM
(``generation_graph``) are written as plain lists of **edges**, and each edge
endpoint is a self-describing ``module[.method]`` string::

    {from: janus_siglip,       to: janus_llama}        # bare → default method
    {from: janus_vqvae.encode, to: janus_llama}        # explicit method
    {from: janus_text_encoder.decode, to: end}         # leaf → virtual sink

An endpoint string therefore *is* the node — there is no indirection through a
named pool.  A node's identity is its canonical ``"<module>.<method>"`` form
(see :meth:`NodeDef.from_endpoint`):

- a **bare** endpoint (no ``.method``) takes the view's *default method* —
  ``forward`` for the training DAG, ``generate`` for the inference FSM;
- a **dotted** endpoint (``module.method``) uses that method verbatim in both
  views.

The same underlying ``nn.Module`` may appear under several methods — e.g. a VQ
codec with both ``janus_vqvae.encode`` (pixels → embeds) and
``janus_vqvae.decode`` (LLM hidden → CE loss).  These are independent nodes
that happen to share weights.

Data flows through the shared ``conversation_list`` carrier (training) or the
FSM ``ctx`` dict (inference) — modules read/write keys directly on those shared
objects; edges declare execution-order / topology only and do **not** route
individual tensor fields.

Reserved sink keyword
---------------------
The string ``"end"`` (exposed as the :data:`END` constant) is a virtual sink:
``to: end`` declares that a node's output flows nowhere — it just makes the
node visible to the active subset.  Every node MUST appear on at least one
edge (no orphans); leaf nodes use ``to: end`` to satisfy that invariant.

``end`` is reserved — it may only appear in an edge's ``to`` field, never as a
``from`` endpoint or as a real module/method name.

Two execution views consume these edge lists (see ``training_graph.py`` and
``generation_graph.py``):

- ``TrainingGraph``  — DAG view over ``OmniConfig.training_graph`` (a flat
  list of edges).  Active nodes are derived from the endpoints (excluding the
  virtual ``end`` node); a topological sort produces the forward order.
- ``GenerationGraph`` — FSM view.  Each ``state.body`` is itself a list of
  inline edge dicts; the unique nodes appearing as endpoints (excluding
  ``end``) execute once per FSM step in declaration order.

See the :class:`~veomni.models.seed_omni.graphs.training_graph.TrainingGraph` and
:class:`~veomni.models.seed_omni.graphs.generation_graph.GenerationGraph` module
docstrings for the full schema.
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
    """Parsed graph node — one ``module.method`` call-site.

    Constructed from an edge endpoint string (:meth:`from_endpoint`).  ``name``
    is the canonical ``"<module>.<method>"`` identity used to de-duplicate the
    same call-site across edges.
    """

    name: str
    module: str
    method: str = "forward"

    @classmethod
    def from_endpoint(cls, endpoint: str, *, default_method: str) -> "NodeDef":
        """Parse a ``module[.method]`` endpoint string into a :class:`NodeDef`.

        A bare ``module`` takes *default_method* (``forward`` for the training
        DAG, ``generate`` for the inference FSM); a dotted ``module.method``
        uses that method verbatim.
        """
        if endpoint == END:
            raise ValueError(f"'{END}' is the virtual sink, not a node — it may only appear in an edge's `to` field.")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError(f"Edge endpoint must be a non-empty 'module[.method]' string. Got: {endpoint!r}")

        if "." in endpoint:
            module, method = endpoint.split(".", 1)
        else:
            module, method = endpoint, default_method

        if not module or not method:
            raise ValueError(f"Edge endpoint '{endpoint}': module/method must be non-empty.")

        return cls(name=f"{module}.{method}", module=module, method=method)


@dataclass
class EdgeDef:
    """Parsed graph edge — declares ``from_`` must execute before ``to``.

    Both endpoints are parsed from ``module[.method]`` strings into
    :class:`NodeDef`; ``from_`` / ``to`` hold their canonical names.  ``to`` may
    be the reserved keyword :data:`END` (``"end"``), in which case ``to_node``
    is ``None`` and the edge is a virtual sink.

    Data is **not** routed through edges — training modules share the
    ``conversation_list`` carrier; inference modules merge outputs into ``ctx``.
    """

    from_: str
    to: str
    from_node: NodeDef
    to_node: Optional[NodeDef] = None

    @classmethod
    def parse(cls, spec: Dict, *, default_method: str) -> "EdgeDef":
        """Parse a ``{from, to}`` edge dict; endpoints resolve via *default_method*."""
        if not isinstance(spec, dict):
            raise ValueError(f"Edge spec must be a `{{from, to}}` dict. Got: {spec!r}")
        if "module" in spec or "method" in spec:
            raise ValueError(
                f"Edge spec must not contain node fields (`module`/`method`) — write endpoints "
                f"as `module[.method]` strings in `from`/`to`. Got: {spec!r}"
            )
        if "from" not in spec or "to" not in spec:
            raise ValueError(f"Edge must declare both `from` and `to`. Got: {spec!r}")

        from_ep = spec["from"]
        to_ep = spec["to"]
        if from_ep == END:
            raise ValueError(f"`from: {END}` is forbidden — the virtual sink may only appear on `to`.")

        from_node = NodeDef.from_endpoint(from_ep, default_method=default_method)
        if to_ep == END:
            return cls(from_=from_node.name, to=END, from_node=from_node, to_node=None)
        to_node = NodeDef.from_endpoint(to_ep, default_method=default_method)
        return cls(from_=from_node.name, to=to_node.name, from_node=from_node, to_node=to_node)

    @property
    def name(self) -> str:
        """Synthetic diagnostic name (``"<from> -> <to>"``); edges are anonymous."""
        return f"{self.from_} -> {self.to}"

    def is_sink(self) -> bool:
        """True iff this edge terminates at the virtual ``end`` sink."""
        return self.to == END
