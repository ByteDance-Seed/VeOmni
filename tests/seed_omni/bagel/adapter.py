"""BAGEL adapter shim for the generated SeedOmni parity suite."""

from __future__ import annotations

from tests.seed_omni.parity_suite.core.adapters import ReferenceV2ProbeAdapter


class BagelParityAdapter(ReferenceV2ProbeAdapter):
    """BAGEL's adapter currently uses the suite default probe comparator.

    Keep this shim so Bagel can add model-specific behavior later without
    changing `cases.yaml`.
    """
