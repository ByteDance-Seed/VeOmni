from typing import Optional

from transformers import PretrainedConfig


class QwenImageConditionModelConfig(PretrainedConfig):
    model_type = "QwenImageConditionModel"

    def __init__(
        self,
        base_model_path: str = "",
        tokenizer_subfolder: str = "tokenizer",
        text_encoder_subfolder: str = "text_encoder",
        vae_subfolder: str = "vae",
        scheduler_subfolder: str = "scheduler",
        max_sequence_length: int = 512,
        num_train_timesteps: int = 1000,
        height: int = 1024,
        width: int = 1024,
        training_recipe: str = "veomni",
        image_resize_mode: str = "resize",
        prompt_template_encode: str = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
            "relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        ),
        prompt_template_encode_start_idx: int = 34,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        if training_recipe not in ("veomni", "diffsynth"):
            raise ValueError("training_recipe must be 'veomni' or 'diffsynth'.")
        if image_resize_mode not in ("resize", "center_crop"):
            raise ValueError("image_resize_mode must be 'resize' or 'center_crop'.")
        if height <= 0 or width <= 0 or max_sequence_length <= 0 or num_train_timesteps < 2:
            raise ValueError("Image dimensions and sequence length must be positive; timesteps must be at least 2.")
        self.base_model_path = base_model_path
        self.tokenizer_subfolder = tokenizer_subfolder
        self.text_encoder_subfolder = text_encoder_subfolder
        self.vae_subfolder = vae_subfolder
        self.scheduler_subfolder = scheduler_subfolder
        self.max_sequence_length = max_sequence_length
        self.num_train_timesteps = num_train_timesteps
        self.height = height
        self.width = width
        self.training_recipe = training_recipe
        self.image_resize_mode = image_resize_mode
        self.prompt_template_encode = prompt_template_encode
        self.prompt_template_encode_start_idx = prompt_template_encode_start_idx
        self.guidance_scale = guidance_scale
        self.seed = seed
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path,
        **kwargs,
    ):
        try:
            config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        except Exception:
            config_dict = {}
        config_dict["base_model_path"] = pretrained_model_name_or_path
        return config_dict, kwargs
