import math
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
        training_cfg_scale: float = 1.0,
        training_cfg_schedule: str = "constant",
        training_cfg_curvature_power: float = 2.0,
        cfg_unconditional_mode: str = "per_sample",
        cfg_unconditional_path: Optional[str] = None,
        video_sigma_bucket_weights: Optional[list[float]] = None,
        **kwargs,
    ):
        if "training_cfg_loss_mode" in kwargs:
            raise ValueError("Use training_cfg_curvature_power instead of the removed training_cfg_loss_mode.")
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
        if not math.isfinite(training_cfg_scale) or training_cfg_scale < 1:
            raise ValueError("training_cfg_scale must be finite and >= 1.")
        if training_cfg_schedule not in ("constant", "sigma"):
            raise ValueError("training_cfg_schedule must be constant or sigma.")
        if not math.isfinite(training_cfg_curvature_power) or not 0 <= training_cfg_curvature_power <= 2:
            raise ValueError("training_cfg_curvature_power must be finite and in [0, 2].")
        if cfg_unconditional_mode not in ("per_sample", "shared_empty"):
            raise ValueError("cfg_unconditional_mode must be per_sample or shared_empty.")
        if training_cfg_scale > 1 and cfg_unconditional_mode == "shared_empty" and not cfg_unconditional_path:
            raise ValueError("shared_empty CFG requires cfg_unconditional_path.")
        weights = list(video_sigma_bucket_weights) if video_sigma_bucket_weights is not None else [1.0] * 5
        if len(weights) != 5 or any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError("video_sigma_bucket_weights requires five finite nonnegative weights, not all zero.")
        self.training_cfg_scale = training_cfg_scale
        self.training_cfg_schedule = training_cfg_schedule
        self.training_cfg_curvature_power = training_cfg_curvature_power
        self.cfg_unconditional_mode = cfg_unconditional_mode
        self.cfg_unconditional_path = cfg_unconditional_path
        self.video_sigma_bucket_weights = weights
