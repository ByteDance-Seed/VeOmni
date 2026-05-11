"""
GenerationStateMachine (FSM) for OmniModel inference.

The FSM drives multi-modal generation by cycling through *named states*.
Each state specifies:

  body
      An ordered list of named connections to execute per step.  Each
      connection either runs a module (``{module: X}``) or transfers an
      output and runs a module (``{from: A, output: k, to: B, as: m}``).

  token_length
      How many auto-regressive steps to spend in this state:

        ``{type: fixed, value: N}``        — always N steps.
        ``{type: variable}``               — run until a transition fires.
        ``{type: from_request, key: K}``   — read N from the generation request dict.
        ``{type: from_generated_text, key: K}``
                                           — parse N from a previously generated
                                             text field in the context.

  transitions
      Ordered list of ``{condition: ..., next_state: S}`` items checked after
      every step.  First matching condition wins.  Supported conditions:

        ``{type: token_match, token_id: T}``
            Fires when ``context["last_token_id"] == T``.

        ``{type: steps_complete}``
            Fires when the current state has run its allotted steps.

        ``{type: always}``
            Unconditional — always fires (useful as a fallback / terminal
            transition).

Usage
-----
  >>> fsm = GenerationStateMachine(config["generation_states"], config["connections"])
  >>> fsm.reset(request={"max_new_tokens": 512})
  >>> context = {"input_ids": ..., "attention_mask": ...}
  >>> while not fsm.is_done():
  ...     context = fsm.step(modules, context)
  ...     fsm.maybe_transition(context)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .graph import ConnectionDef


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
        """Return the number of steps for this state, or ``None`` for variable."""
        if self.type == "fixed":
            return self.value
        if self.type == "from_request":
            return int(request.get(self.key, 0)) if self.key else None
        if self.type == "from_generated_text":
            val = context.get(self.key)
            return int(val) if val is not None else None
        # variable
        return None


# ── State ─────────────────────────────────────────────────────────────────────


class _State:
    def __init__(self, name: str, spec: Dict, conn_pool: Dict[str, ConnectionDef]):
        self.name = name
        self.body_names: List[str] = spec["body"]
        self.connections: List[ConnectionDef] = [conn_pool[n] for n in self.body_names]
        self.token_length = _TokenLength(**spec.get("token_length", {"type": "variable"}))
        self.transitions: List[_Transition] = [
            _Transition(
                condition=_Condition(**t["condition"]),
                next_state=t["next_state"],
            )
            for t in spec.get("transitions", [])
        ]


# ── GenerationStateMachine ────────────────────────────────────────────────────


class GenerationStateMachine:
    """Finite state machine that drives multi-modal inference.

    Parameters
    ----------
    fsm_config:
        The ``generation_states`` section of ``OmniConfig``.  Must have:
        ``initial`` (str) and ``states`` (dict of state specs).
    connections:
        Full pool of ``ConnectionDef`` objects from ``OmniGraph`` (or parsed
        directly from ``OmniConfig.connections``).
    """

    def __init__(self, fsm_config: Dict, connections: Dict[str, Any]):
        # Normalise connections to ConnectionDef
        conn_pool: Dict[str, ConnectionDef] = {}
        for n, v in connections.items():
            if isinstance(v, ConnectionDef):
                conn_pool[n] = v
            else:
                conn_pool[n] = ConnectionDef.parse(n, v)

        self._initial: str = fsm_config["initial"]
        self._states: Dict[str, _State] = {
            name: _State(name, spec, conn_pool)
            for name, spec in fsm_config["states"].items()
        }
        self._done_sentinel: Optional[str] = fsm_config.get("done_state")

        # Runtime state (reset before each generate call)
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
        """Return True when the FSM has reached a terminal condition."""
        if self._done_sentinel and self._current == self._done_sentinel:
            return True
        # If no explicit done_state, check if there are no transitions left
        # (caller is responsible for setting a done_state or using max_new_tokens)
        return False

    # ── Step & Transition ─────────────────────────────────────────────────────

    def step(self, modules: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step of the current state's body.

        Each connection in ``body`` is executed in order.  Connection outputs
        are merged into ``context`` before the next connection runs.

        Parameters
        ----------
        modules:
            ``{module_name: OmniModule}`` dict from OmniModel.
        context:
            Mutable generation context (input_ids, attention_mask, kv cache,
            previously generated tokens, etc.).

        Returns
        -------
        Updated context dict.
        """
        ctx = dict(context)
        for conn in self._current_state.connections:
            if conn.module is not None:
                # {module: X} — execute X with full context
                mod = modules[conn.module]
                out = mod.generate_step(**ctx)
                ctx.update(out)
            else:
                # {from: A, output: k, to: B, as: m} — route then execute B
                val = ctx.get(conn.output_key)
                if val is not None and conn.as_ is not None:
                    ctx[conn.as_] = val
                if conn.to is not None:
                    mod = modules[conn.to]
                    out = mod.generate_step(**ctx)
                    ctx.update(out)

        self._steps_in_state += 1
        return ctx

    def maybe_transition(self, context: Dict[str, Any]) -> bool:
        """Check transitions for the current state.

        Returns True if a transition fired (state changed).
        """
        state = self._current_state
        for trans in state.transitions:
            if trans.condition.check(context, self._steps_in_state, self._total_steps):
                self._transition_to(trans.next_state, context)
                return True
        return False

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def current_state_name(self) -> str:
        return self._current

    @property
    def steps_in_current_state(self) -> int:
        return self._steps_in_state

    # ── Internal ──────────────────────────────────────────────────────────────

    @property
    def _current_state(self) -> _State:
        return self._states[self._current]

    def _transition_to(self, next_state: str, context: Dict[str, Any]) -> None:
        self._current = next_state
        self._steps_in_state = 0
        self._total_steps = self._current_state.token_length.resolve(self._request, context)
