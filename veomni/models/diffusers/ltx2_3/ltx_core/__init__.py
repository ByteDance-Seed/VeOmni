"""Copied from https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core

Upstream LTX imports ``ltx_core`` as a top-level package. Bind this tree
onto ``sys.path``.
"""

import sys
from pathlib import Path


_LTX_CORE_DIR = Path(__file__).resolve().parent
_LTX_CORE_PARENT = str(_LTX_CORE_DIR.parent)


def _bind_this_tree() -> None:
    if sys.path[:1] != [_LTX_CORE_PARENT]:
        if _LTX_CORE_PARENT in sys.path:
            sys.path.remove(_LTX_CORE_PARENT)
        sys.path.insert(0, _LTX_CORE_PARENT)

    existing = sys.modules.get("ltx_core")
    if existing is None:
        return
    existing_file = getattr(existing, "__file__", None)
    if existing_file is not None and Path(existing_file).resolve() == _LTX_CORE_DIR / "__init__.py":
        return
    stale = [name for name in list(sys.modules) if name == "ltx_core" or name.startswith("ltx_core.")]
    for name in stale:
        del sys.modules[name]


_bind_this_tree()
