from copy import deepcopy

from transformers import PretrainedConfig


NATIVE_DEFAULTS = {
    "attn_mode": "torch",
    "attn_param": None,
    "concat_condition": True,
    "glyph_byT5_v2": True,
    "guidance_embed": False,
    "heads_num": 16,
    "hidden_size": 2048,
    "ideal_resolution": "480p",
    "ideal_task": "t2v",
    "in_channels": 32,
    "is_reshape_temporal_channels": False,
    "mlp_act_type": "gelu_tanh",
    "mlp_width_ratio": 4,
    "mm_double_blocks_depth": 54,
    "mm_single_blocks_depth": 0,
    "out_channels": 32,
    "patch_size": [1, 1, 1],
    "qk_norm": True,
    "qk_norm_type": "rms",
    "qkv_bias": True,
    "rope_dim_list": [16, 56, 56],
    "rope_theta": 256,
    "text_pool_type": None,
    "text_projection": "single_refiner",
    "text_states_dim": 3584,
    "text_states_dim_2": None,
    "use_attention_mask": True,
    "use_cond_type_embedding": True,
    "use_meanflow": False,
    "vision_projection": "linear",
    "vision_states_dim": 1152,
}


class HunyuanVideo15Config(PretrainedConfig):
    model_type = "HunyuanVideo15Model"
    condition_model_type = "HunyuanVideo15ConditionModel"

    def __init__(self, **kwargs):
        for key, default in NATIVE_DEFAULTS.items():
            setattr(self, key, kwargs.pop(key, deepcopy(default)))
        if self.ideal_task != "t2v" or self.ideal_resolution != "480p":
            raise ValueError("This integration targets the 480p T2V base model")
        if not self.concat_condition or self.guidance_embed or self.use_meanflow:
            raise ValueError("Distilled/meanflow/unconditional-input variants are not supported")
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
        self.architectures = ["HunyuanVideo15Model"]

    def to_native_dict(self):
        return {key: getattr(self, key) for key in NATIVE_DEFAULTS}


class HunyuanVideo15ConditionConfig(PretrainedConfig):
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
        dtype="bfloat16",
        height=480,
        width=832,
        num_frames=33,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_path = base_model_path
        self.seed = seed
        self.generator_device = generator_device
        self.train_timestep_shift = train_timestep_shift
        self.num_train_timesteps = num_train_timesteps
        self.max_text_length = max_text_length
        self.byt5_max_length = byt5_max_length
        self.dtype = dtype
        self.height, self.width, self.num_frames = height, width, num_frames
        if height % 16 or width % 16 or num_frames % 4 != 1:
            raise ValueError("VAE requires spatial multiples of 16 and 4n+1 frames")
        if train_timestep_shift <= 0 or num_train_timesteps <= 0:
            raise ValueError("Time shift and timestep scale must be positive")

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        config, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        # The native root config describes a pipeline, not a Transformers model.
        config = {"base_model_path": str(pretrained_model_name_or_path)}
        return config, kwargs
