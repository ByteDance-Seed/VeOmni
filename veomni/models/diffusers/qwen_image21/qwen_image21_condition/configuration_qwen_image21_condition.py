from transformers import PretrainedConfig


class QwenImage21ConditionModelConfig(PretrainedConfig):
    model_type = "QwenImage21ConditionModel"

    def __init__(
        self,
        base_model_path: str = "",
        processor_subfolder: str = "processor",
        text_encoder_subfolder: str = "text_encoder",
        vae_subfolder: str = "vae",
        scheduler_subfolder: str = "scheduler",
        height: int = 512,
        width: int = 512,
        seed: int | None = 42,
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.processor_subfolder = processor_subfolder
        self.text_encoder_subfolder = text_encoder_subfolder
        self.vae_subfolder = vae_subfolder
        self.scheduler_subfolder = scheduler_subfolder
        self.height = height
        self.width = width
        self.seed = seed
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        try:
            config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        except Exception:
            config_dict = {}
        config_dict["base_model_path"] = pretrained_model_name_or_path
        return config_dict, kwargs
