"""Self-contained BAGEL transformers-compatible test reference.

This package is intentionally local to tests. It does not import
`bagel-official` and is not the SeedOmni V2 implementation.
"""

from tests.seed_omni.bagel.transformers.configuration_bagel import BagelReferenceConfig
from tests.seed_omni.bagel.transformers.modeling_bagel import BagelReferenceForCausalLM
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
from tests.seed_omni.bagel.transformers.wrapper import BagelOfficialReferenceWrapper


__all__ = [
    "AutoEncoderParams",
    "Bagel",
    "BagelConfig",
    "BagelOfficialReferenceWrapper",
    "BagelReferenceConfig",
    "BagelReferenceForCausalLM",
    "Qwen2Config",
    "Qwen2ForCausalLM",
    "Qwen2Model",
    "SiglipVisionConfig",
    "SiglipVisionModel",
    "register_for_auto_classes",
]


def register_for_auto_classes() -> None:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    AutoConfig.register(BagelConfig.model_type, BagelConfig, exist_ok=True)
    AutoModel.register(BagelConfig, Bagel, exist_ok=True)
    AutoConfig.register(Qwen2Config.model_type, Qwen2Config, exist_ok=True)
    AutoModel.register(Qwen2Config, Qwen2Model, exist_ok=True)
    AutoModelForCausalLM.register(Qwen2Config, Qwen2ForCausalLM, exist_ok=True)
    AutoConfig.register(BagelReferenceConfig.model_type, BagelReferenceConfig, exist_ok=True)
    AutoModel.register(BagelReferenceConfig, BagelReferenceForCausalLM, exist_ok=True)
    AutoModelForCausalLM.register(BagelReferenceConfig, BagelReferenceForCausalLM, exist_ok=True)
