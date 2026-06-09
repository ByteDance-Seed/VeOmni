"""Configuration for BAGEL's flow connector module."""

from transformers import PretrainedConfig


class BagelFlowConnectorConfig(PretrainedConfig):
    model_type = "bagel_flow_connector"

    def __init__(
        self,
        hidden_size: int = 3584,
        vit_hidden_size: int = 1152,
        patch_latent_dim: int = 64,
        max_latent_size: int = 64,
        vit_max_num_patch_per_side: int = 70,
        connector_act: str = "gelu_pytorch_tanh",
        timestep_frequency_embedding_size: int = 256,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.vit_hidden_size = vit_hidden_size
        self.patch_latent_dim = patch_latent_dim
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.timestep_frequency_embedding_size = timestep_frequency_embedding_size
        super().__init__(**kwargs)
