"""Configuration for the MiniMax-H3-Image condition model."""

from typing import Optional

from veomni.models.configuration_base import VeOmniPretrainedConfig

from ..pdd import DEFAULT_SIGMAS, validate_pdd_sigmas


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
        height: int = 1024,
        width: int = 1024,
        seed: int = 42,
        pdd_sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
        num_pdd_heads: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if any(not isinstance(size, int) or size <= 0 or size % 32 for size in (height, width)):
            raise ValueError("MiniMax-H3-Image training height and width must be positive multiples of 32.")
        self.base_model_path = base_model_path
        self.text_encoder_subfolder = text_encoder_subfolder
        self.vae_subfolder = vae_subfolder
        self.processor_subfolder = processor_subfolder
        self.text_encoder_num_retained_layers = text_encoder_num_retained_layers
        self.max_sequence_length = max_sequence_length
        self.skip_encoder_load = skip_encoder_load
        self.height = height
        self.width = width
        self.seed = seed
        self.pdd_sigmas = validate_pdd_sigmas(pdd_sigmas, num_pdd_heads)
        self.num_pdd_heads = len(self.pdd_sigmas) - 1

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = dict(config_dict)
        if "pdd_sigmas" in kwargs and "num_pdd_heads" not in kwargs:
            config_dict.pop("num_pdd_heads", None)
        for name in ("height", "width", "pdd_sigmas", "num_pdd_heads"):
            if name in kwargs:
                config_dict[name] = kwargs.pop(name)
        return super().from_dict(config_dict, **kwargs)
