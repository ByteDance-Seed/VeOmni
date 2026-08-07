# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Make ``import deepspec`` work, from an install or a local checkout.

DeepSpec is normally installed via VeOmni's ``deepspec`` extra (a git-pinned
dependency). When it is not installed — e.g. local DeepSpec development — this
falls back to putting a checkout on ``sys.path``. Resolution order:

0. An already-importable ``deepspec`` (the installed extra) — used as-is.
1. ``$DEEPSPEC_PATH`` (explicit; may point at the repo root or its parent).
2. A sibling ``DeepSpec/`` directory next to the VeOmni repo root, or a
   ``DeepSpec/`` checkout inside the repo root.

The check is memoised so repeated calls (registration + trainer + data) are
cheap and idempotent.
"""

import os
import sys
from functools import lru_cache
from typing import List, Optional


DEEPSPEC_ENV_VAR = "DEEPSPEC_PATH"

# Marker file that identifies a DeepSpec repo root: ``<root>/deepspec/__init__.py``.
_PACKAGE_DIR_NAME = "deepspec"


def _looks_like_deepspec_root(path: str) -> bool:
    """True if ``path`` contains an importable ``deepspec`` package."""
    return os.path.isfile(os.path.join(path, _PACKAGE_DIR_NAME, "__init__.py"))


def _candidate_roots() -> List[str]:
    candidates: List[str] = []

    env_path = os.environ.get(DEEPSPEC_ENV_VAR)
    if env_path:
        env_path = os.path.abspath(os.path.expanduser(env_path))
        # Accept either the repo root itself or its parent directory.
        candidates.append(env_path)
        candidates.append(os.path.join(env_path, "DeepSpec"))

    # ``veomni/integrations/deepspec/deepspec_path.py`` -> VeOmni repo root is
    # four levels up; DeepSpec is expected as its sibling.
    veomni_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    workspace_root = os.path.dirname(veomni_repo_root)
    candidates.append(os.path.join(workspace_root, "DeepSpec"))
    candidates.append(os.path.join(veomni_repo_root, "DeepSpec"))

    return candidates


def _resolve_root() -> Optional[str]:
    for candidate in _candidate_roots():
        if candidate and _looks_like_deepspec_root(candidate):
            return candidate
    return None


@lru_cache(maxsize=1)
def ensure_deepspec_importable() -> str:
    """Make ``import deepspec`` work; return the resolved DeepSpec repo root.

    Raises a clear ``ImportError`` if no DeepSpec checkout can be found, telling
    the user to set ``DEEPSPEC_PATH``.
    """
    # Already importable (installed, or path set by a previous call / caller).
    try:
        import deepspec  # noqa: F401

        return os.path.abspath(os.path.dirname(os.path.dirname(deepspec.__file__)))
    except ImportError:
        pass

    root = _resolve_root()
    if root is None:
        searched = "\n  ".join(_candidate_roots())
        raise ImportError(
            "Could not locate a DeepSpec checkout. The DeepSpec draft-model "
            "bridge needs the DeepSpec repository importable as `deepspec`.\n"
            f"Set the `{DEEPSPEC_ENV_VAR}` environment variable to the DeepSpec "
            "repo root, or place a `DeepSpec/` checkout next to the VeOmni repo.\n"
            f"Searched:\n  {searched}"
        )

    if root not in sys.path:
        sys.path.insert(0, root)

    # Validate the import actually works now (fail loud, fail early).
    import deepspec  # noqa: F401

    return root


__all__ = [
    "DEEPSPEC_ENV_VAR",
    "ensure_deepspec_importable",
]
