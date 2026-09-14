from typing import Optional

from transformers import PretrainedConfig


class LTXVideoConditionModelConfig(PretrainedConfig):
    model_type = "LTXVideoConditionModel"

    def __init__(
        self,
        base_model_path: str = "",
        gemma_model_path: Optional[str] = None,
        tokenizer_subfolder: str = "",
        text_encoder_subfolder: str = "",
        vae_subfolder: str = "",
        scheduler_subfolder: str = "",
        max_sequence_length: int = 256,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        video_max_size: int = 512,
        with_audio: bool = False,
        first_frame_conditioning_p: float = 0.5,
        timestep_sampling_mode: str = "shifted_logit_normal",
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.gemma_model_path = gemma_model_path
        self.tokenizer_subfolder = tokenizer_subfolder
        self.text_encoder_subfolder = text_encoder_subfolder
        self.vae_subfolder = vae_subfolder
        self.scheduler_subfolder = scheduler_subfolder
        self.max_sequence_length = max_sequence_length
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.video_max_size = video_max_size
        self.with_audio = with_audio
        self.first_frame_conditioning_p = first_frame_conditioning_p
        self.timestep_sampling_mode = timestep_sampling_mode
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path,
        **kwargs,
    ):
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        config_dict["base_model_path"] = pretrained_model_name_or_path
        return config_dict, kwargs
