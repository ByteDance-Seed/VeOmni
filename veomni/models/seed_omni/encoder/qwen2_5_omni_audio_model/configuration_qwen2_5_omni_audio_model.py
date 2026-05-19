from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig

from ..base import BaseEncoderConfigMixin


class Qwen25OmniAudioModelConfig(BaseEncoderConfigMixin, Qwen2_5OmniAudioEncoderConfig):
    model_type = "qwen2_5_omni_audio_model"

    def __init__(self, train_origin_projector=False, **kwargs):
        super().__init__(**kwargs)
        self.train_origin_projector = train_origin_projector
