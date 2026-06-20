"""Reference oracle factory."""

from __future__ import annotations

from typing import Any

from tests.seed_omni.parity_suite.core import ParityCase
from tests.seed_omni.parity_suite.reference.contract import ReferenceOracle


# Public factory ---------------------------------------------------------------


def build_reference_oracle(case: ParityCase, driver: Any) -> ReferenceOracle:
    """Build the configured reference oracle for one parity case."""

    del driver
    oracle = case.recipe.reference.get("oracle")
    if oracle is None:
        raise ValueError(f"Recipe {case.model.name}.{case.recipe.id} must declare reference.oracle.")
    oracle_name = str(oracle)
    if oracle_name == "hf_model":
        from tests.seed_omni.parity_suite.reference.oracles.hf_model import HfModelReferenceOracle

        return HfModelReferenceOracle(case=case)
    prefix = "hf_module."
    if oracle_name.startswith(prefix):
        from tests.seed_omni.parity_suite.reference.oracles.hf_module import HfModuleReferenceOracle

        name = oracle_name.removeprefix(prefix)
        if not name:
            raise ValueError(f"Recipe {case.model.name}.{case.recipe.id} has empty hf_module oracle name.")
        return HfModuleReferenceOracle(case=case, name=name)
    raise ValueError(
        f"Unsupported reference oracle {oracle_name!r} for {case.model.name}.{case.recipe.id}; "
        "expected 'hf_model' or 'hf_module.<name>'."
    )


__all__ = ["build_reference_oracle"]
