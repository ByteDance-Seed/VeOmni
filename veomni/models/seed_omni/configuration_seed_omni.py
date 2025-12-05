from copy import deepcopy
from typing import Any, Dict, Literal, Optional

from transformers import AutoConfig, PretrainedConfig

from ..loader import MODEL_CONFIG_REGISTRY


def _init_config(config_dict: Optional[Dict[str, Any]]) -> Optional["PretrainedConfig"]:
    """
    Initialize model config from config_dict. If config_dict is None, return a PretrainedConfig.
    """
    if config_dict is None:
        return PretrainedConfig()

    config_dict = deepcopy(config_dict)
    model_type = config_dict.pop("model_type")
    if model_type == "":
        return PretrainedConfig()
    return MODEL_CONFIG_REGISTRY[model_type]()(**config_dict)


class SeedOmniEncoderConfig(PretrainedConfig):
    model_type = "seed_omni_encoder"
    sub_configs = {
        "image_config": AutoConfig,
        "video_config": AutoConfig,
        "audio_config": AutoConfig,
        "text_config": AutoConfig,
    }

    def __init__(
        self,
        image_config: Optional[Dict[str, Any]] = None,
        video_config: Optional[Dict[str, Any]] = None,
        audio_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        encode_input: bool = True,
        encode_output: bool = False,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.image_config = _init_config(image_config)
        self.video_config = _init_config(video_config)
        self.audio_config = _init_config(audio_config)
        self.text_config = _init_config(text_config)
        self.encode_input = encode_input
        self.encode_output = encode_output
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class SeedOmniDecoderConfig(PretrainedConfig):
    model_type = "seed_omni_decoder"
    sub_configs = {
        "image_config": AutoConfig,
        "video_config": AutoConfig,
        "audio_config": AutoConfig,
    }

    def __init__(
        self,
        image_config: Optional[Dict[str, Any]] = None,
        video_config: Optional[Dict[str, Any]] = None,
        audio_config: Optional[Dict[str, Any]] = None,
        encode_input: bool = False,
        encode_output: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.image_config = _init_config(image_config)
        self.video_config = _init_config(video_config)
        self.audio_config = _init_config(audio_config)
        self.encode_input = encode_input
        self.encode_output = encode_output
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class SeedOmniConfig(PretrainedConfig):
    model_type = "seed_omni"
    sub_configs = {"encoder_config": AutoConfig, "foundation_config": AutoConfig, "decoder_config": AutoConfig}

    def __init__(
        self,
        encoder_config: Dict[Literal["image_config"], Dict[str, Any]] = {},
        foundation_config: Optional[Dict[str, Any]] = None,
        decoder_config: Dict[Literal["image_config"], Dict[str, Any]] = {},
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.encoder_config = SeedOmniEncoderConfig(**encoder_config)
        self.decoder_config = SeedOmniDecoderConfig(**decoder_config)
        self.foundation_config = _init_config(foundation_config)
        self.initializer_range = initializer_range
        super().__init__(architectures=kwargs.pop("architectures", "SeedOmniForCausalLM"), **kwargs)

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.foundation_config
