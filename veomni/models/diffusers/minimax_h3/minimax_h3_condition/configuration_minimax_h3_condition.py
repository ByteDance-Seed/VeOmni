from typing import Optional

from transformers import PretrainedConfig


class MiniMaxH3ConditionModelConfig(PretrainedConfig):
    model_type = "MiniMaxH3ConditionModel"

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        text_encoder_subfolder: str = "text_encoder",
        video_vae_subfolder: str = "video_vae",
        audio_vae_subfolder: str = "audio_vae",
        processor_subfolder: str = "processor",
        num_train_timesteps: int = 1000,
        sigma_shift_video: float = 12.0,
        sigma_shift_audio: float = 3.0,
        use_keyframe_condition: bool = True,
        keyframe_indices: list = None,
        imgvid_cond_noise_aug: float = 0.999,
        audio_cond_noise_aug: float = 1.0,
        skip_encoder_load: bool = False,
        video_max_frames: int = 73,
        video_max_resolution: int = 848,
        text_encoder_num_retained_layers: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_path = base_model_path
        self.text_encoder_subfolder = text_encoder_subfolder
        self.video_vae_subfolder = video_vae_subfolder
        self.audio_vae_subfolder = audio_vae_subfolder
        self.processor_subfolder = processor_subfolder
        self.num_train_timesteps = num_train_timesteps
        self.sigma_shift_video = sigma_shift_video
        self.sigma_shift_audio = sigma_shift_audio
        self.use_keyframe_condition = use_keyframe_condition
        self.keyframe_indices = keyframe_indices if keyframe_indices is not None else [0, -1]
        self.imgvid_cond_noise_aug = imgvid_cond_noise_aug
        self.audio_cond_noise_aug = audio_cond_noise_aug
        self.skip_encoder_load = skip_encoder_load
        self.video_max_frames = video_max_frames
        self.video_max_resolution = video_max_resolution
        self.text_encoder_num_retained_layers = text_encoder_num_retained_layers
