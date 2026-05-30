# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Backward-compat shim. The real package is now ``patchgen``.

Re-exports the public surface so ``from veomni.patchgen import PatchConfig``
keeps working, and registers each submodule under ``veomni.patchgen.<sub>``
in :data:`sys.modules` so the very common in-tree pattern
``from veomni.patchgen.patch_spec import PatchConfig`` (used by every
``*_patch_gen_config.py`` file) also keeps working without a per-submodule
forwarding file. Python's import machinery checks ``sys.modules`` before
attempting parent-package ``__path__`` resolution, so pre-populating the
keys is sufficient for ordinary ``import`` statements even though this
shim is a single ``.py`` file rather than a package.

The shim does **not** support ``python -m veomni.patchgen.<sub>`` — runpy
validates the loader's owning module name and refuses to execute a
submodule whose loader belongs to the ``patchgen.<sub>`` namespace. Use
the ``patchgen`` console script (``patchgen --check`` / ``patchgen <module>``)
instead, which is what the Makefile, CI, and ``*_patch_gen_config.py``
docstrings now reference.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

import patchgen as _patchgen
from patchgen import *  # noqa: F401,F403


__all__ = list(_patchgen.__all__)

# Make `from veomni.patchgen.<sub> import X` resolve to patchgen.<sub>.X.
# Python's import machinery checks sys.modules before falling back to the
# parent package's __path__ — pre-populating these keys is enough to make
# submodule imports work even though this shim is a single .py module.
#
# Skip the `globals()[_sub] = _mod` step for names that `patchgen` already
# re-exports as public callables (e.g. `run_codegen` is both a function
# and the submodule that defines it). Without this guard the assignment
# would overwrite the function binding from `from patchgen import *`,
# silently turning `veomni.patchgen.run_codegen(...)` into a module —
# the same trap that `patchgen/__init__.py` works around with PEP 562.
_PATCHGEN_PUBLIC = set(_patchgen.__all__)
for _sub in ("patch_spec", "codegen", "run_codegen", "check_patchgen"):
    _mod = _importlib.import_module(f"patchgen.{_sub}")
    _sys.modules[f"{__name__}.{_sub}"] = _mod
    if _sub not in _PATCHGEN_PUBLIC:
        globals()[_sub] = _mod

del _sub, _mod, _sys, _importlib, _patchgen, _PATCHGEN_PUBLIC
