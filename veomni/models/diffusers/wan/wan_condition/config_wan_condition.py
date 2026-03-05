from transformers import PretrainedConfig


class WanConditionConfig(PretrainedConfig):
    model_type = "wan_condition"

    def __init__(
        self,
        base_model_path: str = "",
        tokenizer_subfolder: str = "tokenizer",
        text_encoder_subfolder: str = "text_encoder",
        vae_subfolder: str = "vae",
        scheduler_subfolder: str = "scheduler",
        max_sequence_length: int = 226,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        do_classifier_free_guidance: bool = False,
        load_components: bool = True,
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
        self.load_components = load_components
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        **kwargs,
    ):
        base_model_path = pretrained_model_name_or_path
        config_dict = {"base_model_path": base_model_path}
        return cls.from_dict(config_dict)
