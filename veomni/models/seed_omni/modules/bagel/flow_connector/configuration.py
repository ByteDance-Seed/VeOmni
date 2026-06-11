"""Configuration for BAGEL's flow connector module."""

from transformers import PretrainedConfig


class BagelFlowConnectorConfig(PretrainedConfig):
    model_type = "bagel_flow_connector"

    def __init__(
        self,
        hidden_size: int = 3584,
        patch_latent_dim: int = 64,
        max_latent_size: int = 64,
        timestep_frequency_embedding_size: int = 256,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.patch_latent_dim = patch_latent_dim
        self.max_latent_size = max_latent_size
        self.timestep_frequency_embedding_size = timestep_frequency_embedding_size
        super().__init__(**kwargs)
