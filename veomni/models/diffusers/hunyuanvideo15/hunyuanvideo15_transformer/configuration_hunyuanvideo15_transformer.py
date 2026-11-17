from copy import deepcopy

from ....configuration_base import VeOmniPretrainedConfig


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


class HunyuanVideo15Config(VeOmniPretrainedConfig):
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
        # Native attention always uses dense SDPA, including the checkpoint's
        # "flash" label. Do not inherit the framework's FlashAttention default.
        kwargs.setdefault("attn_implementation", "sdpa")
        super().__init__(**kwargs)
        self.architectures = ["HunyuanVideo15Model"]

    def to_native_dict(self):
        return {key: getattr(self, key) for key in NATIVE_DEFAULTS}
