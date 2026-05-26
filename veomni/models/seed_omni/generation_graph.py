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
         b. Apply edge routing **permissively**: ``ctx[e.output] →
            ctx[e.as_]`` only if ``e.output`` is currently in ``ctx``.
            When the source node returned ``{}`` (e.g. SigLIP with no
            ``pixel_values`` on a text-only inference prompt) the
            routing silently skips — the destination still executes.
         c. Decrement ``e.to``'s pending fan-in.  When it hits zero
            (i.e. all body edges into ``e.to`` have been processed),
            execute ``e.to``.

      This rule generalises "first-encounter execution":

        * For purely linear bodies (e.g. ``text_ar`` =
          ``tok_enc → llama → tok_dec``) it executes every node exactly
          once, in declaration order.
        * For multi-source nodes (e.g. understanding/I2T where
          ``janus_llama`` consumes both ``inputs_embeds`` and
          ``und_image_embeds``) the backbone executes only after the
          last incoming routing edge has fired — both inputs are
          present in ``ctx`` first.
        * For self-feedback bodies (``image_vq`` =
          ``ar_to_vae_dec, vae_dec_to_ar``) the second edge re-routes
          ``vae_decode``'s output back into ``inputs_embeds`` for the
          next iteration's ``janus_llama`` call.

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

        ``{type: steps_complete}``
            Fires when the current state has run its allotted iterations.
            Useless on a ``token_length: variable`` state (steps_done never
            reaches ``None``).

        ``{type: module_signal, key: K}``  (alias: ``ctx_flag``)
            Fires when ``context["module_signal"] == K``.  Modules write a
            one-shot string signal into ``ctx["module_signal"]`` from inside
            ``generate_step`` / ``decode`` — e.g. ``JanusTextEncoder.decode``
            sets ``"start_image_gen"`` / ``"text_done"`` after sampling; a VQ
            decoder sets ``"image_complete"`` on the final patch.  The
            framework **auto-clears** ``ctx["module_signal"]`` once the
            transition fires.  Pair with ``token_length: variable``.  The FSM
            never inspects raw token ids — vocabulary semantics stay inside
            the module.

        ``{type: token_match, token_id: T}``  *(deprecated)*
            Legacy: fires when the last ``input_ids`` token equals ``T``.  Prefer
            ``module_signal`` so token ids are not duplicated in YAML.

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

from .graph import END, EdgeDef, NodeDef, is_end, scalar_token_id


# Reserved name for the framework-injected terminal state.  Every FSM
# automatically gains a ``done`` state with empty body, no transitions, and
# zero-length token budget — users must NOT declare it in their YAML.  All
# transitions whose ``next_state`` is ``"done"`` therefore land on this
# built-in state, and ``OmniModel.generate`` calls each active module's
# :meth:`OmniModule.finalize` hook once the FSM enters it.
DONE_STATE_NAME: str = "done"

# Single ctx slot for module-driven FSM transitions.  Modules set
# ``ctx[FSM_SIGNAL_KEY] = "<signal_name>"``; YAML ``module_signal.key``
# matches that string value (not a separate boolean flag per signal).
FSM_SIGNAL_KEY: str = "module_signal"


# ── Condition helpers ─────────────────────────────────────────────────────────


_KNOWN_CONDITION_TYPES = frozenset({"token_match", "steps_complete", "module_signal", "ctx_flag", "always"})
_MODULE_SIGNAL_TYPES = frozenset({"module_signal", "ctx_flag"})


@dataclass
class _Condition:
    type: str
    token_id: Optional[int] = None
    key: Optional[str] = None  # used by module_signal / ctx_flag

    def __post_init__(self) -> None:
        # Catch malformed YAML at FSM build time, not at first transition
        # check (which would otherwise silently never fire).
        if self.type not in _KNOWN_CONDITION_TYPES:
            raise ValueError(f"Unknown FSM condition type '{self.type}'. Supported: {sorted(_KNOWN_CONDITION_TYPES)}.")
        if self.type == "token_match" and self.token_id is None:
            raise ValueError("Condition `token_match` requires `token_id`.")
        if self.type in _MODULE_SIGNAL_TYPES and not self.key:
            raise ValueError(f"Condition `{self.type}` requires a non-empty `key`.")
        if self.type != "token_match" and self.token_id is not None:
            raise ValueError(f"Condition `{self.type}` does not accept `token_id`.")
        if self.type not in _MODULE_SIGNAL_TYPES and self.key is not None:
            raise ValueError(f"Condition `{self.type}` does not accept `key`.")

    def check(self, context: Dict[str, Any], steps_done: int, total_steps: Optional[int]) -> bool:
        if self.type == "token_match":
            return scalar_token_id(context.get("input_ids")) == self.token_id
        if self.type == "steps_complete":
            return total_steps is not None and steps_done >= total_steps
        if self.type in _MODULE_SIGNAL_TYPES:
            return context.get(FSM_SIGNAL_KEY) == self.key
        if self.type == "always":
            return True
        return False

    def describe(self) -> str:
        if self.type == "token_match":
            return f"token_match({self.token_id})"
        if self.type in _MODULE_SIGNAL_TYPES:
            return f"module_signal({self.key})"
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

        # Inject the built-in terminal state.  Empty body, zero-length token
        # budget, no outgoing transitions: the FSM "rests" here and the
        # orchestrator picks up the post-processing baton via finalize hooks.
        self._states[DONE_STATE_NAME] = _State(
            DONE_STATE_NAME,
            {"body": [], "token_length": {"type": "fixed", "value": 0}, "transitions": []},
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
        """Return True when the FSM has reached the framework-injected terminal state."""
        return self._current == self._done_sentinel

    # ── Step & Transition ─────────────────────────────────────────────────────

    def step(
        self,
        modules: Dict[str, Any],
        context: Dict[str, Any],
        *,
        trace: Optional[List[str]] = None,
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
           b. Apply permissive routing: ``ctx[e.output] → ctx[e.as_]``
              **only if** ``e.output`` is in ``ctx``.  An absent key
              means the source returned ``{}`` (e.g. SigLIP with no
              ``pixel_values``); the routing silently skips and the
              destination still executes when its other inputs land.
           c. Decrement ``pending[e.to]``.  When it reaches zero (and
              ``e.to`` is not ``end`` and not yet executed), execute it.

        ``end`` is a virtual sink and is never executed; an edge with
        ``to: end`` only pins its ``from_`` node into the active set.
        The same node never re-executes within one step.

        Method dispatch: a node whose declared method is ``forward``
        invokes ``module.generate_step``; any other method name dispatches
        as-is (e.g. ``vqvae.decode``, ``text_encoder.emit_image_start``).
        This keeps the same node usable for both the training DAG
        (``forward``/explicit method) and the inference FSM
        (``generate_step``/explicit method) without YAML duplication.

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

        # Per-node first appearance as `from_` in body — distinguishes
        # **feed-forward** edges (with `to: X` *before* X's first `from_`
        # position; these must complete before X runs) from
        # **post-execution feedback** edges (after; they only update ctx
        # for the next iteration / state).  Nodes that never appear as
        # `from_` in body get ``len(body)`` so every incoming edge
        # counts as feed-forward — this is the "to-only sink" case
        # (e.g. ``run_ar`` in ``image_vq_start`` whose body is
        # ``[emit_start_to_ar, ar_run_sink]`` — ``ar_run_sink`` triggers
        # ``run_ar``).
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

        for i, edge in enumerate(state.body):
            # 1. Execute the source node (idempotent for repeated `from_`).
            _run(edge.from_)

            # 2. Permissive routing — skip if the source returned no such
            #    key (or returned None for it; downstream treats both as
            #    absent).  This is the "no input → empty dict" inference
            #    fast-path: e.g. SigLIP with no `pixel_values` returns
            #    ``{}``; the routing edge silently drops; the destination
            #    still executes when its other inputs land.
            if (
                edge.output_key is not None
                and edge.as_ is not None
                and edge.output_key in ctx
                and ctx[edge.output_key] is not None
            ):
                ctx[edge.as_] = ctx[edge.output_key]

            # 3. Decrement the destination's feed-forward pending count;
            #    if the node has no later appearance as a source (it's a
            #    body sink), trigger it now once all its inputs are in.
            if not is_end(edge.to):
                fi = first_from_idx.get(edge.to, len(state.body))
                if i < fi:
                    pending[edge.to] -= 1

        self._steps_in_state += 1
        return ctx

    def maybe_transition(self, context: Dict[str, Any], *, trace: Optional[List[str]] = None) -> bool:
        """Check transitions for the current state.

        Returns True if a transition fired (state changed).

        For ``module_signal`` / ``ctx_flag`` transitions ``context["module_signal"]``
        is popped after logging the trace and before the state switch.
        """
        state = self._current_state
        for trans in state.transitions:
            if trans.condition.check(context, self._steps_in_state, self._total_steps):
                if trace is not None:
                    trace.append(f"transition: {state.name} -> {trans.next_state} [{trans.condition.describe()}]")
                if trans.condition.type in _MODULE_SIGNAL_TYPES:
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

    @property
    def steps_in_current_state(self) -> int:
        return self._steps_in_state

    def state_node_sequence(self, state_name: str) -> List[str]:
        """Return the derived node-execution sequence for a given state."""
        return list(self._states[state_name].node_sequence)

    # ── Visualization ─────────────────────────────────────────────────────────

    def to_mermaid(self, title: Optional[str] = None) -> str:
        """Render the FSM as a Mermaid ``flowchart LR`` with body subgraphs.

        Visual conventions
        ------------------
        Each non-``done`` state is rendered as a labelled subgraph whose
        interior is a mini-flow over the body's data edges (same
        ``output → as`` labels as the training graph; ``to: end`` sink
        edges are filtered out since they don't carry data — they only
        pin a node into the body).  The body's node names inside the
        subgraph are namespaced as ``<state>__<node>`` so the same node
        can appear in multiple states without ID collisions.

        A dashed self-loop on each subgraph carries the iteration count:

        * ``token_length: fixed: N``  → label ``×N``
        * ``variable`` / ``from_request[K]`` / ``from_generated_text[K]``
          → no label.  These have no static upper bound, matching
          "如果没有次数限制那就不写数字".

        State transitions are thick arrows (``==>``) carrying the firing
        condition (e.g. ``module_signal(start_image_gen)``, ``steps_complete``).
        The line weight + simpler label distinguishes them visually from
        the intra-body data edges.

        A small ``▶`` node marks FSM entry; a small ``⏹`` terminal absorbs
        every transition that targets the built-in ``done`` state.  The
        ``done`` state itself is NOT drawn — its body is empty by
        construction (framework-injected, not user-declared), and
        rendering it would just add a redundant box.

        Layout uses ``flowchart LR`` + the ELK renderer to match the
        training graph's left-to-right column-banded look.
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
            lines.append("        direction LR")
            for n_name in state.node_sequence:
                n = self._node_pool[n_name]
                node_label = f"{n.name}<br/><i>{n.module}.{n.method}</i>"
                lines.append(f'        {name}__{n.name}["{node_label}"]:::body_node')
            for e in state.body:
                if is_end(e.to):
                    # `to: end` sinks are declarative pins — they don't carry
                    # data, so they don't appear inside the body's mini-flow.
                    continue
                edge_label = self._edge_label(e)
                arrow = f"-->|{edge_label}|" if edge_label else "-->"
                lines.append(f"        {name}__{e.from_} {arrow} {name}__{e.to}")
            lines.append("    end")

        # ── Self-loops carrying the per-state iteration count ─────────────────
        for name in drawn:
            loop_label = self._loop_label(self._states[name].token_length)
            if loop_label:
                lines.append(f'    state_{name} -.->|"{loop_label}"| state_{name}')
            else:
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

    @staticmethod
    def _edge_label(e: EdgeDef) -> str:
        """Render a body edge's data-routing label — same shape as the training graph's."""
        if e.output_key and e.as_ and e.output_key != e.as_:
            return f'"{e.output_key} → {e.as_}"'
        if e.output_key:
            return f'"{e.output_key}"'
        return ""

    @staticmethod
    def _loop_label(tl: _TokenLength) -> str:
        """Self-loop label encoding the iteration count.

        Only ``fixed: N`` produces a static count (``×N``).  Variable and
        runtime-resolved kinds (``from_request`` / ``from_generated_text``)
        return an empty string — the loop arrow is drawn unlabelled.
        """
        if tl.type == "fixed" and tl.value is not None:
            return f"×{tl.value}"
        return ""

    # ── Internal ──────────────────────────────────────────────────────────────

    @property
    def _current_state(self) -> _State:
        return self._states[self._current]

    def _transition_to(self, next_state: str, context: Dict[str, Any]) -> None:
        self._current = next_state
        self._steps_in_state = 0
        self._total_steps = self._current_state.token_length.resolve(self._request, context)


__all__ = ["GenerationGraph", "END"]
