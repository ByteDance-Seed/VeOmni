"""Configuration for BAGEL's VAE/LLM flow connector."""

from transformers import PretrainedConfig


class BagelFlowConnectorConfig(PretrainedConfig):
    """BAGEL latent-token connector config."""

    model_type = "bagel_flow_connector"

    def __init__(
        self,
        hidden_size: int = 3584,
        z_channels: int = 16,
        latent_patch_size: int = 2,
        patch_latent_dim: int | None = None,
        max_latent_size: int = 64,
        timestep_frequency_embedding_size: int = 256,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.z_channels = z_channels
        self.latent_patch_size = latent_patch_size
        self.patch_latent_dim = (
            latent_patch_size * latent_patch_size * z_channels if patch_latent_dim is None else patch_latent_dim
        )
        self.max_latent_size = max_latent_size
        self.timestep_frequency_embedding_size = timestep_frequency_embedding_size
        super().__init__(**kwargs)


__all__ = ["BagelFlowConnectorConfig"]
