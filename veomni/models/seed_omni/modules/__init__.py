"""Per-family OmniModule implementations.

Status: pending migration to the SeedOmni V2 mixin contract.

The previous V1.5 code under ``janus/`` and ``text/`` subclasses
``OmniModule(ABC)`` with a now-removed ``_build_nn_module`` lifecycle.  V2
makes ``OmniModule`` a *mixin* (see :mod:`veomni.models.seed_omni.module`),
so those classes need to be re-derived against the real
``transformers.JanusModel`` / ``LlamaModel`` etc.  Until that migration
lands, this package exports nothing — keeping the V2 graph runtime
importable in a clean environment.

Tests that need stand-in OmniModules subclass :class:`OmniModule` directly;
see ``tests/seed_omni/print_modules.py`` for the canonical pattern.
"""

# No re-exports yet.  Importing the submodules below would pull in code
# that targets the V1.5 abstract-base-class API and would fail at import
# time in a V2 environment.  The migration is tracked separately.

__all__: list = []
