"""Self-contained BAGEL transformers-compatible test reference.

This package is intentionally local to tests. It does not import
`bagel-official` and is not the SeedOmni V2 implementation.
"""

from tests.seed_omni.bagel.transformers.bagel import BagelOfficialReference, load_vendored_model
from tests.seed_omni.bagel.transformers.vendor.modeling.autoencoder import AutoEncoderParams
from tests.seed_omni.bagel.transformers.vendor.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
    SiglipVisionConfig,
    SiglipVisionModel,
)


__all__ = [
    "AutoEncoderParams",
    "Bagel",
    "BagelConfig",
    "BagelOfficialReference",
    "load_vendored_model",
    "Qwen2Config",
    "Qwen2ForCausalLM",
    "Qwen2Model",
    "SiglipVisionConfig",
    "SiglipVisionModel",
]
