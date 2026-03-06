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
        video_max_size: int = 480,
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
        self.video_max_size = video_max_size
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
