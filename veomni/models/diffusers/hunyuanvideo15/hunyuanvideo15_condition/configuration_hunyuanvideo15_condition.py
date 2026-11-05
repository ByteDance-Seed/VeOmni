import torch

from ....configuration_base import VeOmniPretrainedConfig


def _normalize_condition_dtype(dtype):
    # HF only serializes torch.dtype values automatically for its own dtype field.
    return str(dtype).removeprefix("torch.") if isinstance(dtype, torch.dtype) else dtype


class HunyuanVideo15ConditionConfig(VeOmniPretrainedConfig):
    model_type = "HunyuanVideo15ConditionModel"

    def __init__(
        self,
        base_model_path="",
        seed=42,
        generator_device=None,
        train_timestep_shift=3.0,
        num_train_timesteps=1000,
        max_text_length=1000,
        byt5_max_length=256,
        condition_dtype="bfloat16",
        height=480,
        width=832,
        num_frames=33,
        **kwargs,
    ):
        # Accept older condition configs without setting HF's construction dtype.
        legacy_dtype = kwargs.pop("dtype", None)
        super().__init__(**kwargs)
        self.condition_dtype = _normalize_condition_dtype(legacy_dtype or condition_dtype)
        self.base_model_path = base_model_path
        self.seed = seed
        self.generator_device = generator_device
        self.train_timestep_shift = train_timestep_shift
        self.num_train_timesteps = num_train_timesteps
        self.max_text_length = max_text_length
        self.byt5_max_length = byt5_max_length
        self.height, self.width, self.num_frames = height, width, num_frames
        if height % 16 or width % 16 or num_frames % 4 != 1:
            raise ValueError("VAE requires spatial multiples of 16 and 4n+1 frames")
        if train_timestep_shift <= 0 or num_train_timesteps <= 0:
            raise ValueError("Time shift and timestep scale must be positive")

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # HF reapplies overrides after __init__, so translate the old output-dtype
        # key before that pass can set the frozen encoders' construction dtype.
        legacy_dtype = kwargs.pop("dtype", None)
        if legacy_dtype is not None:
            kwargs.setdefault("condition_dtype", legacy_dtype)
        if "condition_dtype" in kwargs:
            kwargs["condition_dtype"] = _normalize_condition_dtype(kwargs["condition_dtype"])
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        config, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        # The native root config describes a pipeline, not a Transformers model.
        if config.get("model_type") != cls.model_type:
            config = {"base_model_path": str(pretrained_model_name_or_path)}
        return config, kwargs
