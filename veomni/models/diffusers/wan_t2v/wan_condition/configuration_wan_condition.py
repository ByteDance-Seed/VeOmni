import json
import os
from typing import Optional, Tuple

from transformers import PretrainedConfig


class WanTransformer3DConditionModelConfig(PretrainedConfig):
    model_type = "WanTransformer3DConditionModel"

    def __init__(
        self,
        base_model_path: str = "",
        tokenizer_subfolder: str = "tokenizer",
        text_encoder_subfolder: str = "text_encoder",
        vae_subfolder: str = "vae",
        scheduler_subfolder: str = "scheduler",
        max_sequence_length: int = 512,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        do_classifier_free_guidance: bool = False,
        cfg_negative_prompt: str = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
            "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
            "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
            "fused fingers, still picture, messy background, three legs, many people in the background, "
            "walking backwards"
        ),
        cfg_negative_prob: float = 0.1,
        video_max_size: int = 480,
        seed: Optional[int] = 42,
        expand_timesteps: bool = False,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.tokenizer_subfolder = tokenizer_subfolder
        self.text_encoder_subfolder = text_encoder_subfolder
        self.vae_subfolder = vae_subfolder
        self.scheduler_subfolder = scheduler_subfolder
        self.max_sequence_length = max_sequence_length
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.cfg_negative_prompt = cfg_negative_prompt
        self.cfg_negative_prob = cfg_negative_prob
        self.video_max_size = video_max_size
        self.seed = seed
        self.expand_timesteps = expand_timesteps
        self.patch_size = patch_size
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path,
        **kwargs,
    ):
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        config_dict["base_model_path"] = pretrained_model_name_or_path

        model_index_path = os.path.join(pretrained_model_name_or_path, "model_index.json")
        if os.path.isfile(model_index_path):
            with open(model_index_path, "r", encoding="utf-8") as f:
                model_index = json.load(f)
            if "expand_timesteps" in model_index and "expand_timesteps" not in config_dict:
                config_dict["expand_timesteps"] = model_index["expand_timesteps"]

        transformer_config_path = os.path.join(pretrained_model_name_or_path, "transformer", "config.json")
        if os.path.isfile(transformer_config_path) and "patch_size" not in config_dict:
            with open(transformer_config_path, "r", encoding="utf-8") as f:
                transformer_config = json.load(f)
            if "patch_size" in transformer_config:
                config_dict["patch_size"] = transformer_config["patch_size"]
                
        return config_dict, kwargs