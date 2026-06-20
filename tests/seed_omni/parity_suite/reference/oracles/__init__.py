"""Reference oracle backends for parity-suite execution."""

from .factory import build_reference_oracle
from .hf_model import HfModelReferenceOracle, HfModelSubject
from .hf_module import HfModuleReferenceOracle, HfModuleSubject


__all__ = [
    "HfModelReferenceOracle",
    "HfModuleReferenceOracle",
    "HfModuleSubject",
    "HfModelSubject",
    "build_reference_oracle",
]
