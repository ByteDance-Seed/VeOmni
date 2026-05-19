from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniVisionEncoderConfig

from ..base import BaseEncoderConfigMixin


class Qwen25OmniVisionModelConfig(BaseEncoderConfigMixin, Qwen2_5OmniVisionEncoderConfig):
    model_type = "qwen2_5_omni_vision_model"

    def __init__(
        self,
        return_hidden_states=False,
        train_origin_projector=False,
        **kwargs,
    ):
        self.return_hidden_states = return_hidden_states
        self.train_origin_projector = train_origin_projector
        super().__init__(**kwargs)
