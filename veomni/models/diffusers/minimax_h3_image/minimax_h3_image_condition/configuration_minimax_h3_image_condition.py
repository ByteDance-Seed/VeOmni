"""Configuration for the MiniMax-H3-Image condition model."""

from typing import Optional

from veomni.models.configuration_base import VeOmniPretrainedConfig


class MiniMaxH3ImageConditionModelConfig(VeOmniPretrainedConfig):
    model_type = "MiniMaxH3ImageConditionModel"

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        text_encoder_subfolder: str = "text_encoder",
        vae_subfolder: str = "vae_diffusers",
        processor_subfolder: str = "processor",
        text_encoder_num_retained_layers: int = 50,
        max_sequence_length: int = 4096,
        skip_encoder_load: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_path = base_model_path
        self.text_encoder_subfolder = text_encoder_subfolder
        self.vae_subfolder = vae_subfolder
        self.processor_subfolder = processor_subfolder
        self.text_encoder_num_retained_layers = text_encoder_num_retained_layers
        self.max_sequence_length = max_sequence_length
        self.skip_encoder_load = skip_encoder_load
