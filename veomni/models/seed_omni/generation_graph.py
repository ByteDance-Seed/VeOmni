"""
GenerationGraph: FSM view over the ``nodes`` / ``edges`` pools for inference.

The FSM drives multi-modal generation by cycling through *named states*.
Each state specifies:

  body
      An ordered list of **edge** names from the pool.  Per FSM step the
      runtime walks the body in **topological order**:

      1. Pre-compute, per node ``X`` appearing in body, the count of
         body edges with ``to: X`` (its in-body fan-in).
      2. Walk the body edges in declaration order.  For each edge ``e``:

         a. Ensure ``e.from_`` is executed.  If it has any unprocessed
            in-body fan-in, that's a body-ordering bug — error.
         b. Decrement ``e.to``'s pending fan-in.  When it hits zero
            (i.e. all body edges into ``e.to`` have been processed),
            execute ``e.to``.

      Each executed node merges its return dict into ``ctx`` directly
      (``ctx.update(out)``).  Edges declare execution order only — they
      do not route individual fields.  Early FSM steps use
      ``conversation_list``; later AR steps use ``input_ids`` /
      ``past_key_values`` / ``hidden_states`` as modules write them.

      This rule generalises "first-encounter execution":

        * For purely linear bodies (e.g. ``text_ar`` =
          ``tok_enc → llama → tok_dec``) it executes every node exactly
          once, in declaration order.
        * For multi-source nodes (e.g. understanding/I2T where
          ``janus_llama`` consumes both ``inputs_embeds`` and
          ``und_image_embeds``) the backbone executes only after the
          last in-body fan-in edge has fired — both keys are already
          in ``ctx`` from upstream ``ctx.update(out)`` calls.
        * For self-feedback bodies (``image_vq`` =
          ``ar_to_vae_dec, vae_dec_to_ar``) ``vae_decode`` writes
          ``embed`` (or ``inputs_embeds``) into ``ctx``; the next
          ``janus_llama`` step reads it directly — no edge renaming.

      An edge with ``to: end`` is purely declarative — it pins the producing
      node into the active set without routing anywhere.

Method dispatch
---------------
Bare ``{module: X}`` nodes parse as ``forward`` for training; the FSM maps
that default to ``generate`` at runtime.  Any other method (``encode``,
``decode``, ``emit_image_start``, …) is taken from YAML as-is.

  transitions
      Ordered list of ``{condition: ..., next_state: S}`` items checked after
      every iteration of the state body.  First matching condition wins.

      A state has **no iteration-count budget**: its body runs once and then
      keeps iterating until one of its transitions fires.  Modules — not the
      FSM — decide when a state ends, either by raising a signal (the AR
      loop case) or implicitly after a single pass (the bridge/leaf case,
      via a ``default`` transition).  Supported conditions:

        ``{type: module_signal, key: K}``
            Fires when ``context["module_signal"] == K``.  Modules write a
            one-shot string signal into ``ctx["module_signal"]`` from inside
            ``generate_step`` / ``decode`` — e.g. ``JanusTextEncoder.decode``
            sets ``"start_image_gen"`` / ``"text_done"`` after sampling; a VQ
            decoder sets ``"image_complete"`` on the final patch.  The
            framework **auto-clears** ``ctx["module_signal"]`` once the
            transition fires.  This is how an AR loop state (e.g. ``image_vq``)
            keeps iterating until its module says "done".  The FSM never
            inspects raw token ids — vocabulary semantics stay inside the
            module.

        ``{type: default}``
            The catch-all (switch-``default``) branch — matches
            unconditionally.  Because transitions are evaluated in order and
            the first match wins, a ``default`` is the lowest-priority
            **fallback**: it fires only when none of the conditions listed
            *before* it matched.  It MUST therefore be the last transition in
            a state (the FSM rejects a ``default`` that isn't, since any
            transition after it would be dead code).  Two uses:

            * sole transition on a deterministic single-pass bridge / leaf
              state (prompt encode, ``<boi>`` / ``<eoi>`` emit) — run the
              body once, then advance;
            * the else-branch after one or more ``module_signal`` checks
              (e.g. "sampled a normal text token → keep decoding").

Usage
-----
  >>> fsm = GenerationGraph(config["generation_graph"], config["nodes"], config["edges"])
  >>> fsm.reset()
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

from .graph import (
    EdgeDef,
    NodeDef,
    is_end,
)


# Reserved name for the framework-injected terminal state.  Every FSM
# automatically gains a ``done`` state with empty body and no transitions —
# users must NOT declare it in their YAML.  All
# transitions whose ``next_state`` is ``"done"`` therefore land on this
# built-in state, and ``OmniModel.generate`` calls each active module's
# :meth:`OmniModule.finalize` hook once the FSM enters it.
DONE_STATE_NAME: str = "done"

# Single ctx slot for module-driven FSM transitions.  Modules set
# ``ctx[FSM_SIGNAL_KEY] = "<signal_name>"``; YAML ``module_signal.key``
# matches that string value (not a separate boolean flag per signal).
FSM_SIGNAL_KEY: str = "module_signal"


def _fsm_dispatch_method(node: NodeDef) -> str:
    """Resolve the module method invoked for one FSM node execution.

    Training defaults bare nodes to ``forward`` (see :class:`NodeDef`); inference
    defaults the same nodes to ``generate``.  Explicit YAML methods are unchanged.
    """
    return "generate" if node.method == "forward" else node.method


# ── Condition helpers ─────────────────────────────────────────────────────────


_KNOWN_CONDITION_TYPES = frozenset({"module_signal", "default"})


@dataclass
class _Condition:
    type: str
    key: Optional[str] = None  # required by `module_signal`, forbidden otherwise

    def __post_init__(self) -> None:
        # Catch malformed YAML at FSM build time, not at first transition
        # check (which would otherwise silently never fire).
        if self.type not in _KNOWN_CONDITION_TYPES:
            raise ValueError(f"Unknown FSM condition type '{self.type}'. Supported: {sorted(_KNOWN_CONDITION_TYPES)}.")
        if self.type == "module_signal" and not self.key:
            raise ValueError("Condition `module_signal` requires a non-empty `key`.")
        if self.type != "module_signal" and self.key is not None:
            raise ValueError(f"Condition `{self.type}` does not accept `key`.")

    def check(self, context: Dict[str, Any]) -> bool:
        if self.type == "module_signal":
            return context.get(FSM_SIGNAL_KEY) == self.key
        if self.type == "default":
            return True
        return False

    def describe(self) -> str:
        if self.type == "module_signal":
            return f"module_signal({self.key})"
        return self.type


@dataclass
class _Transition:
    condition: _Condition
    next_state: str


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

        self.transitions: List[_Transition] = [
            _Transition(
                condition=_Condition(**t["condition"]),
                next_state=t["next_state"],
            )
            for t in spec.get("transitions", [])
        ]

        # A `default` condition matches unconditionally, so with first-match
        # ordering it is the lowest-priority fallback — anything after it is
        # dead code.  Reject that at build time so the priority is explicit.
        for i, trans in enumerate(self.transitions[:-1]):
            if trans.condition.type == "default":
                raise ValueError(
                    f"State '{name}': a `default` transition must be last (it fires "
                    f"unconditionally, so the {len(self.transitions) - i - 1} transition(s) after "
                    f"it would never run). Move `default` to the end of `{name}.transitions`."
                )


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

        # `done` is reserved — auto-injected below.  Users must NOT redeclare
        # it; doing so silently lets a custom body/transitions override the
        # framework's terminal semantics, which is exactly the kind of magic
        # we are trying to avoid.
        if DONE_STATE_NAME in fsm_config["states"]:
            raise ValueError(
                f"State name '{DONE_STATE_NAME}' is reserved and auto-injected by the framework. "
                f"Remove the explicit `{DONE_STATE_NAME}:` block from your generation_graph YAML — "
                f"transitions targeting `next_state: {DONE_STATE_NAME}` will land on the built-in "
                f"terminal state, which then triggers each active module's `finalize` hook."
            )
        # The pre-existing `done_state` config knob is gone — the framework
        # always uses `DONE_STATE_NAME`.  If the user still has it lying
        # around, reject loudly so they migrate cleanly.
        if "done_state" in fsm_config:
            raise ValueError(
                "`generation_graph.done_state` is no longer configurable — the terminal state "
                f"is hardcoded to '{DONE_STATE_NAME}'. Remove the `done_state:` line from your "
                "generation_graph YAML."
            )

        self._initial: str = fsm_config["initial"]
        self._states: Dict[str, _State] = {
            name: _State(name, spec, node_pool, edge_pool) for name, spec in fsm_config["states"].items()
        }

        # Inject the built-in terminal state.  Empty body, no outgoing
        # transitions: the FSM "rests" here and the orchestrator picks up the
        # post-processing baton via finalize hooks.
        self._states[DONE_STATE_NAME] = _State(
            DONE_STATE_NAME,
            {"body": [], "transitions": []},
            node_pool,
            edge_pool,
        )

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
        self._done_sentinel: str = DONE_STATE_NAME

        # Runtime state — reset before each generate call.
        self._current: str = self._initial

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset FSM to the initial state for a new generation request."""
        self._current = self._initial

    def is_done(self) -> bool:
        """Return True when the FSM has reached the framework-injected terminal state."""
        return self._current == self._done_sentinel

    # ── Step & Transition ─────────────────────────────────────────────────────

    def step(
        self,
        modules: Dict[str, Any],
        ctx: Dict[str, Any],
        trace: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one iteration of the current state.

        Algorithm (topological body execution, see module-doc §"body"):

        1. Compute ``pending[X]`` = number of body edges with ``to: X``
           for every node ``X`` in the body's node sequence.  Nodes with
           ``pending == 0`` are body sources — they execute on first
           sight as ``edge.from_``.
        2. Walk ``state.body`` edges in declaration order.  For each
           edge ``e``:

           a. If ``e.from_`` hasn't been executed yet, execute it now.
              (If it still has unprocessed in-body fan-in, that's a
              body-ordering bug — raise.)
           c. Decrement ``pending[e.to]``.  When it reaches zero (and
              ``e.to`` is not ``end`` and not yet executed), execute it.

        ``end`` is a virtual sink and is never executed; an edge with
        ``to: end`` only pins its ``from_`` node into the active set.
        The same node never re-executes within one step.

        Method dispatch: bare nodes (YAML default ``forward``) invoke
        ``module.generate``; explicit YAML methods dispatch as-is.

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
        state = self._current_state
        executed: set = set()
        pending_step_ids: Any = None

        # Per-node first appearance as `from_` in body — distinguishes
        # **feed-forward** edges (with `to: X` *before* X's first `from_`
        # position; these must complete before X runs) from
        # **post-execution feedback** edges (after; they only update ctx
        # for the next iteration / state).  Nodes that never appear as
        # `from_` in body get ``len(body)`` so every incoming edge
        # counts as feed-forward — this is the "to-only sink" case
        # (e.g. ``janus_llama`` in ``image_vq_start`` whose body is
        # ``[emit_start_to_janus_llama, janus_llama_sink]`` — the sink edge
        # triggers ``janus_llama``).
        first_from_idx: Dict[str, int] = {}
        for i, e in enumerate(state.body):
            if not is_end(e.from_) and e.from_ not in first_from_idx:
                first_from_idx[e.from_] = i

        # Feed-forward fan-in count per node (only edges before the
        # node's first `from_` appearance).  Used to gate execution and
        # to detect body-order bugs (a node about to run as `from_`
        # whose pending > 0 means an upstream feed-forward edge appears
        # later than expected).
        pending: Dict[str, int] = dict.fromkeys(state.node_sequence, 0)
        for i, e in enumerate(state.body):
            if is_end(e.to):
                continue
            fi = first_from_idx.get(e.to, len(state.body))
            if i < fi:
                pending[e.to] += 1

        def _run(node_name: str) -> None:
            nonlocal pending_step_ids
            if is_end(node_name) or node_name in executed:
                return
            if pending.get(node_name, 0) > 0:
                raise RuntimeError(
                    f"FSM step (state '{state.name}'): node '{node_name}' is being "
                    f"executed before all of its feed-forward in-body inputs have "
                    f"been routed (pending={pending[node_name]}). Re-order the body "
                    f"so every edge feeding '{node_name}' precedes its first "
                    f"appearance as a source."
                )
            node = self._node_pool[node_name]
            module = modules.get(node.module)
            if module is None:
                raise KeyError(
                    f"FSM step: module '{node.module}' (node '{node_name}') missing "
                    f"from modules dict. Provided: {sorted(modules)}."
                )
            method_name = _fsm_dispatch_method(node)
            method_fn: Optional[Callable] = getattr(module, method_name, None)
            if method_fn is None:
                raise AttributeError(f"FSM node '{node_name}': {type(module).__name__} has no method '{method_name}'.")
            if trace is not None:
                trace.append(f"[State|{state.name}] {node_name}: {node.module}.{method_name}")
            out = method_fn(**ctx, generation_kwargs=generation_kwargs)
            if not isinstance(out, dict):
                raise TypeError(f"FSM node '{node_name}'.{method_name} must return a dict; got {type(out).__name__}.")
            ctx.update(out)
            executed.add(node_name)

        for i, edge in enumerate(state.body):
            # 1. Execute the source node (idempotent for repeated `from_`).
            _run(edge.from_)

            # A terminating ``module_signal`` (e.g. ``text_done`` on ``</s>``)
            # means no further nodes in this body should run — transition is
            # evaluated in :meth:`maybe_transition` after the step returns.
            if FSM_SIGNAL_KEY in ctx:
                break

            # 2. Decrement the destination's feed-forward pending count;
            #    if the node has no later appearance as a source (it's a
            #    body sink), trigger it now once all its inputs are in.
            if not is_end(edge.to):
                fi = first_from_idx.get(edge.to, len(state.body))
                if i < fi:
                    pending[edge.to] -= 1

        return ctx

    def maybe_transition(self, context: Dict[str, Any], *, trace: Optional[List[str]] = None) -> bool:
        """Check transitions for the current state.

        Returns True if a transition fired (state changed).

        For ``module_signal`` transitions ``context["module_signal"]`` is
        popped after logging the trace and before the state switch.
        """
        state = self._current_state
        for trans in state.transitions:
            if trans.condition.check(context):
                if trace is not None:
                    trace.append(f"transition: {state.name} -> {trans.next_state} [{trans.condition.describe()}]")
                if trans.condition.type == "module_signal":
                    context.pop(FSM_SIGNAL_KEY, None)
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

    def state_node_sequence(self, state_name: str) -> List[str]:
        """Return the derived node-execution sequence for a given state."""
        return list(self._states[state_name].node_sequence)

    # ── Visualization ─────────────────────────────────────────────────────────

    def to_mermaid(self, title: Optional[str] = None) -> str:
        """Render the FSM as a Mermaid ``flowchart LR`` with body subgraphs.

        Visual conventions
        ------------------
        Each non-``done`` state is rendered as a labelled subgraph whose
        interior is a mini-flow over the body's topology edges (``to: end`` sink
        edges are filtered out since they don't carry data — they only
        pin a node into the body).  The body's node names inside the
        subgraph are namespaced as ``<state>__<node>`` so the same node
        can appear in multiple states without ID collisions.

        A dashed (unlabelled) self-loop on each subgraph marks that the
        state body iterates until one of its transitions fires — there is no
        static iteration count to display (modules decide when to leave).

        State transitions are thick arrows (``==>``) carrying the firing
        condition (e.g. ``module_signal(start_image_gen)``, ``default``).
        The line weight + simpler label distinguishes them visually from
        the intra-body data edges.

        A small ``▶`` node marks FSM entry; a small ``⏹`` terminal absorbs
        every transition that targets the built-in ``done`` state.  The
        ``done`` state itself is NOT drawn — its body is empty by
        construction (framework-injected, not user-declared), and
        rendering it would just add a redundant box.

        Layout uses ``flowchart LR`` + the ELK renderer so major FSM states
        read left-to-right. State bodies use ``direction TB`` so multi-step
        states stay readable instead of stretching into a single wide strip.
        """
        lines: List[str] = []
        if title:
            lines += ["---", f"title: {title}", "---"]
        lines.append("%%{init: {'flowchart': {'defaultRenderer': 'elk'}}}%%")
        lines.append("flowchart LR")

        done_name = self._done_sentinel

        # ── Entry / terminal markers (small circles) ──────────────────────────
        lines.append('    fsm_start(("▶")):::fsm_start')
        has_done_target = done_name is not None and any(
            trans.next_state == done_name for state in self._states.values() for trans in state.transitions
        )
        if has_done_target:
            lines.append('    fsm_done(("⏹")):::fsm_terminal')

        # ── Body subgraphs (skip the done state — empty body, no value) ───────
        drawn: List[str] = []
        for name, state in self._states.items():
            if name == done_name:
                continue
            drawn.append(name)
            lines.append(f"    subgraph state_{name} [{name}]")
            lines.append("        direction TB")
            for n_name in state.node_sequence:
                n = self._node_pool[n_name]
                node_label = f"{n.name}<br/><i>{n.module}.{n.method}</i>"
                lines.append(f'        {name}__{n.name}["{node_label}"]:::body_node')
            for e in state.body:
                if is_end(e.to):
                    # `to: end` sinks are declarative pins — they don't carry
                    # data, so they don't appear inside the body's mini-flow.
                    continue
                lines.append(f"        {name}__{e.from_} --> {name}__{e.to}")
            lines.append("    end")

        # ── Self-loops marking that a state body iterates until a transition ──
        for name in drawn:
            lines.append(f"    state_{name} -.-> state_{name}")

        # ── Entry edge ────────────────────────────────────────────────────────
        if self._initial in self._states and self._initial != done_name:
            lines.append(f"    fsm_start ==> state_{self._initial}")

        # ── State transitions (thick ==> arrows with quoted condition labels) ─
        for name in drawn:
            for trans in self._states[name].transitions:
                if trans.next_state == done_name:
                    target = "fsm_done"
                else:
                    target = f"state_{trans.next_state}"
                cond = trans.condition.describe()
                lines.append(f'    state_{name} ==>|"{cond}"| {target}')

        # ── Class definitions / styling ──────────────────────────────────────
        lines += [
            "    classDef body_node fill:#fff,stroke:#666",
            "    classDef fsm_start fill:#dff,stroke:#06c,stroke-width:2px",
            "    classDef fsm_terminal fill:#eee,stroke:#333,stroke-width:1px,stroke-dasharray:3 3",
        ]
        if self._initial in self._states and self._initial != done_name:
            # Highlight the initial state's subgraph background to mirror the
            # training graph's source-node colouring (light blue accent).
            lines.append(f"    style state_{self._initial} fill:#eef,stroke:#06c,stroke-width:2px")

        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────────

    @property
    def _current_state(self) -> _State:
        return self._states[self._current]

    def _transition_to(self, next_state: str, context: Dict[str, Any]) -> None:
        self._current = next_state


__all__ = ["GenerationGraph"]
