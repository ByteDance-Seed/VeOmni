import os

from transformers import PretrainedConfig
from transformers.utils import CONFIG_NAME, cached_file


_CONFIG_LOADER_KWARGS = {
    "cache_dir",
    "force_download",
    "proxies",
    "token",
    "local_files_only",
    "revision",
    "trust_remote_code",
    "subfolder",
    "_from_pipeline",
    "_from_auto",
    "_commit_hash",
    "_configuration_file",
}


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
        background_color: tuple[int, int, int] = (255, 255, 255),
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
        self.background_color = background_color
        self.seed = seed
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        if kwargs.get("gguf_file") is not None or os.path.isfile(pretrained_model_name_or_path):
            config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        else:
            config_file = cached_file(
                pretrained_model_name_or_path,
                kwargs.get("_configuration_file", CONFIG_NAME),
                cache_dir=kwargs.get("cache_dir"),
                force_download=kwargs.get("force_download", False),
                proxies=kwargs.get("proxies"),
                token=kwargs.get("token"),
                local_files_only=kwargs.get("local_files_only", False),
                revision=kwargs.get("revision"),
                subfolder=kwargs.get("subfolder", ""),
                _commit_hash=kwargs.get("_commit_hash"),
                _raise_exceptions_for_gated_repo=True,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=True,
            )
            if config_file is None:
                config_dict = {}
                kwargs = {key: value for key, value in kwargs.items() if key not in _CONFIG_LOADER_KWARGS}
            else:
                config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        config_dict["base_model_path"] = pretrained_model_name_or_path
        return config_dict, kwargs
