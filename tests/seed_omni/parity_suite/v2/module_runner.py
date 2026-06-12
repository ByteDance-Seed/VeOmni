"""Module-level runner protocol for parity adapters.

The generic suite can plan target graph nodes, but model adapters own the
semantic input factories for isolated module execution.
"""

from __future__ import annotations

from typing import Any


def run_module_method(module: Any, method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    fn = module if method == "forward" else getattr(module, method)
    output = fn(**kwargs)
    if not isinstance(output, dict):
        return {"output": output}
    return output
