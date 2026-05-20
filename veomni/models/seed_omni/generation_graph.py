"""
GenerationGraph: FSM view over the ``nodes`` / ``edges`` pools for inference.

The FSM drives multi-modal generation by cycling through *named states*.
Each state specifies:

  body
      An ordered list of **edge** names from the pool.  At each FSM step:

      1. Every unique node appearing as ``from`` or ``to`` of an edge in
         ``body`` (excluding the virtual :data:`~.graph.END` sink) is
         executed exactly once, in **first-appearance order**.
      2. Edges drive data routing: when consuming ``ctx`` for node X, every
         edge in ``body`` with ``to: X`` injects ``ctx[edge.output] →
         ctx[edge.as]`` from the producing node's freshly-emitted output.
      3. Each node's output is also merged into ``ctx`` so that subsequent
         steps (and subsequent state-iterations within the same state) see
         it.

      An edge with ``to: end`` is purely declarative — it pins the producing
      node into the active set without routing anywhere.

Method dispatch
---------------
A node whose YAML-declared method is ``forward`` dispatches to the module's
``generate_step`` (the conventional inference entry).  Nodes with an explicit
method name (e.g. ``vq_decoder.decode``) dispatch to that method as-is.  This
keeps training and inference consuming the same pool while letting modules
expose specialised inference routines.

  token_length
      How many auto-regressive iterations to spend in this state:

        ``{type: fixed, value: N}``        — always N iterations.
        ``{type: variable}``               — run until a transition fires.
        ``{type: from_request, key: K}``   — read N from the generation
                                             request dict.
        ``{type: from_generated_text, key: K}``
                                           — parse N from a previously
                                             generated text field in ctx.

  transitions
      Ordered list of ``{condition: ..., next_state: S}`` items checked after
      every iteration.  First matching condition wins.  Supported conditions:

        ``{type: token_match, token_id: T}``
            Fires when ``context["last_token_id"] == T``.

        ``{type: steps_complete}``
            Fires when the current state has run its allotted iterations.

        ``{type: always}``
            Unconditional — useful as a fallback / terminal transition.

Usage
-----
  >>> fsm = GenerationGraph(config["generation_graph"], config["nodes"], config["edges"])
  >>> fsm.reset(request={"max_new_tokens": 512})
  >>> ctx = {"input_ids": ..., "attention_mask": ...}
  >>> while not fsm.is_done():
  ...     ctx = fsm.step(modules, ctx)
  ...     fsm.maybe_transition(ctx)

See also
--------
``graph.py``           — NodeDef / EdgeDef / END shared pool types.
``training_graph.py``  — DAG view driven by ``OmniConfig.training_graph``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .graph import END, EdgeDef, NodeDef, is_end


# ── Condition helpers ─────────────────────────────────────────────────────────


@dataclass
class _Condition:
    type: str
    token_id: Optional[int] = None

    def check(self, context: Dict[str, Any], steps_done: int, total_steps: Optional[int]) -> bool:
        if self.type == "token_match":
            return context.get("last_token_id") == self.token_id
        if self.type == "steps_complete":
            return total_steps is not None and steps_done >= total_steps
        if self.type == "always":
            return True
        return False

    def describe(self) -> str:
        if self.type == "token_match":
            return f"token_match({self.token_id})"
        return self.type


@dataclass
class _Transition:
    condition: _Condition
    next_state: str


# ── TokenLength spec ──────────────────────────────────────────────────────────


@dataclass
class _TokenLength:
    type: str
    value: Optional[int] = None
    key: Optional[str] = None

    def resolve(self, request: Dict[str, Any], context: Dict[str, Any]) -> Optional[int]:
        """Return the number of iterations for this state, or ``None`` for variable."""
        if self.type == "fixed":
            return self.value
        if self.type == "from_request":
            return int(request.get(self.key, 0)) if self.key else None
        if self.type == "from_generated_text":
            val = context.get(self.key)
            return int(val) if val is not None else None
        return None  # variable

    def describe(self) -> str:
        if self.type == "fixed":
            return f"fixed={self.value}"
        if self.type == "from_request":
            return f"from_request[{self.key}]"
        if self.type == "from_generated_text":
            return f"from_generated_text[{self.key}]"
        return "variable"


# ── State ─────────────────────────────────────────────────────────────────────


class _State:
    """Parsed FSM state.

    ``body`` is a list of :class:`EdgeDef` (resolved against the pool).  The
    *node sequence* — the unique nodes appearing as ``from``/``to`` endpoints
    in declaration order, excluding ``end`` — is precomputed for stable
    iteration.
    """

    def __init__(
        self,
        name: str,
        spec: Dict,
        node_pool: Dict[str, NodeDef],
        edge_pool: Dict[str, EdgeDef],
    ):
        self.name = name
        body_names: List[str] = list(spec.get("body", []))

        body: List[EdgeDef] = []
        for n in body_names:
            if n in node_pool:
                raise ValueError(
                    f"State '{name}' body item '{n}' is a graph node — only edge names are "
                    "allowed in `body` (the active node set is derived from edge endpoints). "
                    f"Add an edge to/from '{n}' (use `to: end` if it is a leaf)."
                )
            if n not in edge_pool:
                raise KeyError(
                    f"State '{name}' body item '{n}' is not a known edge name. Known edges: {sorted(edge_pool)}."
                )
            body.append(edge_pool[n])
        self.body: List[EdgeDef] = body

        # Derive node sequence: unique nodes by first appearance, skipping `end`.
        seen: set = set()
        sequence: List[str] = []
        for e in body:
            for endpoint in (e.from_, e.to):
                if is_end(endpoint) or endpoint in seen:
                    continue
                if endpoint not in node_pool:
                    raise KeyError(
                        f"State '{name}': edge '{e.name}' references unknown node "
                        f"'{endpoint}'. Declared nodes: {sorted(node_pool)}."
                    )
                seen.add(endpoint)
                sequence.append(endpoint)
        self.node_sequence: List[str] = sequence

        self.token_length = _TokenLength(**spec.get("token_length", {"type": "variable"}))
        self.transitions: List[_Transition] = [
            _Transition(
                condition=_Condition(**t["condition"]),
                next_state=t["next_state"],
            )
            for t in spec.get("transitions", [])
        ]


# ── GenerationGraph ───────────────────────────────────────────────────────────


class GenerationGraph:
    """FSM view over the nodes / edges pools that drives multi-modal inference.

    Parameters
    ----------
    fsm_config:
        The ``generation_graph`` section of ``OmniConfig``.  Must have:
        ``initial`` (str) and ``states`` (dict of state specs).
    nodes / edges:
        Raw pool dicts from ``OmniConfig`` (``{name: spec}``).  The FSM parses
        them into :class:`NodeDef` / :class:`EdgeDef` and resolves each
        ``body`` entry to an :class:`EdgeDef`.
    """

    def __init__(
        self,
        fsm_config: Dict,
        nodes: Dict[str, Any],
        edges: Optional[Dict[str, Any]] = None,
    ):
        node_pool = {n: NodeDef.parse(n, v) if not isinstance(v, NodeDef) else v for n, v in (nodes or {}).items()}
        edge_pool = {n: EdgeDef.parse(n, v) if not isinstance(v, EdgeDef) else v for n, v in (edges or {}).items()}

        overlap = set(node_pool) & set(edge_pool)
        if overlap:
            raise ValueError(
                f"`nodes` and `edges` pools share name(s): {sorted(overlap)}. "
                "Each name must be unique across both pools."
            )

        self._node_pool: Dict[str, NodeDef] = node_pool
        self._edge_pool: Dict[str, EdgeDef] = edge_pool

        self._initial: str = fsm_config["initial"]
        self._states: Dict[str, _State] = {
            name: _State(name, spec, node_pool, edge_pool) for name, spec in fsm_config["states"].items()
        }
        if self._initial not in self._states:
            raise KeyError(
                f"GenerationGraph initial state '{self._initial}' not in declared states {sorted(self._states)}."
            )
        for name, state in self._states.items():
            for trans in state.transitions:
                if trans.next_state not in self._states:
                    raise KeyError(
                        f"State '{name}' transitions to undeclared state "
                        f"'{trans.next_state}' (known states: {sorted(self._states)})."
                    )
        self._done_sentinel: Optional[str] = fsm_config.get("done_state")
        if self._done_sentinel is not None and self._done_sentinel not in self._states:
            raise KeyError(
                f"GenerationGraph done_state '{self._done_sentinel}' not in declared states {sorted(self._states)}."
            )

        # Runtime state — reset before each generate call.
        self._current: str = self._initial
        self._steps_in_state: int = 0
        self._total_steps: Optional[int] = None
        self._request: Dict[str, Any] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, request: Optional[Dict[str, Any]] = None) -> None:
        """Reset FSM to the initial state for a new generation request."""
        self._current = self._initial
        self._steps_in_state = 0
        self._request = request or {}
        self._total_steps = self._current_state.token_length.resolve(self._request, {})

    def is_done(self) -> bool:
        """Return True when the FSM has reached the configured terminal state."""
        return self._done_sentinel is not None and self._current == self._done_sentinel

    # ── Step & Transition ─────────────────────────────────────────────────────

    def step(
        self,
        modules: Dict[str, Any],
        context: Dict[str, Any],
        *,
        trace: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute one iteration of the current state.

        Per the design doc (§5 "FSM 一步执行规则"):

          1. Walk ``state.body`` edges in declaration order.
          2. The first time we encounter an edge endpoint (``from_`` then
             ``to``), execute that node — passing the **current ctx** as
             kwargs.  Merge its returned dict back into ``ctx`` so all
             subsequent edges and subsequent iterations observe it.
          3. After (or in lieu of) executing the ``from_`` node, apply the
             edge routing ``ctx[output] → ctx[as]`` so the ``to`` node sees
             a properly-renamed kwarg.
          4. ``end`` is a virtual sink and is never executed.
          5. The same node never re-executes within one step — repeat
             encounters only contribute routing.

        Method dispatch: a node whose declared method is ``forward``
        invokes ``module.generate_step``; any other method name dispatches
        as-is (e.g. ``vqvae.decode``).  This keeps the same node usable for
        both the training DAG (``forward``/explicit method) and the
        inference FSM (``generate_step``/explicit method) without YAML
        duplication.

        Parameters
        ----------
        modules:
            ``{module_name: module}`` dict.  Any object exposing the
            requested method works — :class:`OmniModule` mixin instances,
            plain modules in tests, or FSDP-unwrapped raw modules.
        context:
            Mutable generation context (input_ids, attention_mask, kv
            cache, previously generated tokens, ...).  This call returns
            a *new* dict; the input is not mutated.
        trace:
            Optional list to which the FSM appends one
            ``"<state>:<node>"`` token per executed node — handy for
            print-driven flow tests.

        Returns
        -------
        Updated context dict.
        """
        ctx = dict(context)
        state = self._current_state
        executed: set = set()

        def _run(node_name: str) -> None:
            if is_end(node_name) or node_name in executed:
                return
            node = self._node_pool[node_name]
            module = modules.get(node.module)
            if module is None:
                raise KeyError(
                    f"FSM step: module '{node.module}' (node '{node_name}') missing "
                    f"from modules dict. Provided: {sorted(modules)}."
                )
            method_name = "generate_step" if node.method == "forward" else node.method
            method_fn: Optional[Callable] = getattr(module, method_name, None)
            if method_fn is None:
                raise AttributeError(f"FSM node '{node_name}': {type(module).__name__} has no method '{method_name}'.")
            if trace is not None:
                trace.append(f"{state.name}:{node_name}")
            out = method_fn(**ctx)
            if not isinstance(out, dict):
                raise TypeError(f"FSM node '{node_name}'.{method_name} must return a dict; got {type(out).__name__}.")
            ctx.update(out)
            executed.add(node_name)

        for edge in state.body:
            _run(edge.from_)
            if edge.output_key is not None and edge.as_ is not None and edge.output_key in ctx:
                ctx[edge.as_] = ctx[edge.output_key]
            _run(edge.to)

        self._steps_in_state += 1
        return ctx

    def maybe_transition(self, context: Dict[str, Any], *, trace: Optional[List[str]] = None) -> bool:
        """Check transitions for the current state.

        Returns True if a transition fired (state changed).
        """
        state = self._current_state
        for trans in state.transitions:
            if trans.condition.check(context, self._steps_in_state, self._total_steps):
                if trace is not None:
                    trace.append(f"transition: {state.name} -> {trans.next_state} [{trans.condition.describe()}]")
                self._transition_to(trans.next_state, context)
                return True
        return False

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def initial_state(self) -> str:
        return self._initial

    @property
    def state_names(self) -> List[str]:
        return list(self._states)

    @property
    def current_state_name(self) -> str:
        return self._current

    @property
    def steps_in_current_state(self) -> int:
        return self._steps_in_state

    def state_node_sequence(self, state_name: str) -> List[str]:
        """Return the derived node-execution sequence for a given state."""
        return list(self._states[state_name].node_sequence)

    # ── Visualization ─────────────────────────────────────────────────────────

    def to_mermaid(self, title: Optional[str] = None) -> str:
        """Render the FSM as a Mermaid ``stateDiagram-v2``.

        Each state is annotated with its derived node sequence and its
        ``token_length`` policy.  Transitions are labelled with the condition;
        the initial state is connected from ``[*]`` and the configured
        ``done_state`` (if any) flows to ``[*]``.
        """
        lines: List[str] = []
        if title:
            lines += ["---", f"title: {title}", "---"]
        lines.append("stateDiagram-v2")

        lines.append(f"    [*] --> {self._initial}")

        for name, state in self._states.items():
            seq_str = " → ".join(state.node_sequence) if state.node_sequence else "(empty)"
            tl_str = state.token_length.describe()
            label = f"{name}<br/>nodes: [{seq_str}]<br/>token_length: {tl_str}"
            lines.append(f"    {name} : {label}")

        for name, state in self._states.items():
            for trans in state.transitions:
                lines.append(f"    {name} --> {trans.next_state} : {trans.condition.describe()}")

        if self._done_sentinel and self._done_sentinel in self._states:
            lines.append(f"    {self._done_sentinel} --> [*]")

        lines += [
            "    classDef initial fill:#dff,stroke:#06c,stroke-width:2px",
            "    classDef terminal fill:#eee,stroke:#666,stroke-width:1px,stroke-dasharray:3 3",
            f"    class {self._initial} initial",
        ]
        if self._done_sentinel and self._done_sentinel in self._states:
            lines.append(f"    class {self._done_sentinel} terminal")

        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────────

    @property
    def _current_state(self) -> _State:
        return self._states[self._current]

    def _transition_to(self, next_state: str, context: Dict[str, Any]) -> None:
        self._current = next_state
        self._steps_in_state = 0
        self._total_steps = self._current_state.token_length.resolve(self._request, context)


__all__ = ["GenerationGraph", "END"]
