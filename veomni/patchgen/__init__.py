# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Backward-compat shim for the in-tree ``veomni.patchgen.*`` import path.

The real implementation lives in the sibling ``patchgen`` package
(``patchgen-pkg/``). This directory exists so static analysers, IDEs, and
``mypy`` / ``pyright`` see a real package — each submodule is a one-line
forwarding file that re-exports the corresponding ``patchgen.<sub>``
surface. The 22 in-tree ``*_patch_gen_config.py`` files still use
``from veomni.patchgen.patch_spec import PatchConfig`` and keep working
unchanged.

Caveat: after ``import veomni.patchgen.run_codegen``, the attribute
``veomni.patchgen.run_codegen`` resolves to the *submodule*, not the
function — Python's import machinery does
``setattr(parent_pkg, sub_name, submodule)`` on submodule load and there
is no clean way to prevent that. Prefer ``from veomni.patchgen.run_codegen
import run_codegen`` or just use the ``patchgen`` console script.

Use the ``patchgen`` console script (``patchgen --check`` /
``patchgen <module>``) for the actual CLI — see ``docs/design/patchgen.md``.
"""

import patchgen as _patchgen
from patchgen import *


__all__ = list(_patchgen.__all__)

del _patchgen
