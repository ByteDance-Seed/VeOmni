from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from .configuration_seed import SeedConfig
from .modeling_seed import (
    SeedForCausalLM,
    SeedForSequenceClassification,
    SeedForTokenClassification,
    SeedModel,
)


AutoConfig.register("seed", SeedConfig)
AutoModel.register(SeedConfig, SeedModel)
AutoModelForCausalLM.register(SeedConfig, SeedForCausalLM)
AutoModelForSequenceClassification.register(SeedConfig, SeedForSequenceClassification)
AutoModelForTokenClassification.register(SeedConfig, SeedForTokenClassification)
