import inspect

import diffusers
from transformers import PretrainedConfig


diffusers_version = diffusers.__version__


class LTXVideoTransformerModelConfig(PretrainedConfig):
    model_type = "LTXVideoTransformerModel"
    condition_model_type = "LTXVideoConditionModel"

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        caption_channels: int = 4096,
        norm_eps: float = 1e-06,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        rope_type: str = "split",
        frequencies_precision: str = "float32",
        apply_gated_attention: bool = False,
        caption_proj_before_connector: bool = False,
        cross_attention_adaln: bool = False,
        has_image_input: bool = False,
        with_audio: bool = False,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list = None,
        av_ca_timestep_scale_multiplier: int = 1,
        **kwargs,
    ):
        if positional_embedding_max_pos is None:
            positional_embedding_max_pos = [20, 2048, 2048]
        if audio_positional_embedding_max_pos is None:
            audio_positional_embedding_max_pos = [20]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_layers = num_layers
        self.cross_attention_dim = cross_attention_dim
        self.caption_channels = caption_channels
        self.norm_eps = norm_eps
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.frequencies_precision = frequencies_precision
        self.apply_gated_attention = apply_gated_attention
        self.caption_proj_before_connector = caption_proj_before_connector
        self.cross_attention_adaln = cross_attention_adaln
        self.has_image_input = has_image_input
        self.with_audio = with_audio
        self.audio_num_attention_heads = audio_num_attention_heads
        self.audio_attention_head_dim = audio_attention_head_dim
        self.audio_in_channels = audio_in_channels
        self.audio_out_channels = audio_out_channels
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
        super().__init__(**kwargs)

    def to_diffuser_dict(self):
        return {
            key: getattr(self, key) for key in _LTX_CONFIG_INIT_SIGNATURE.parameters if key not in ("self", "kwargs")
        }

    def to_dict(self):
        return_dict = super().to_dict()
        return_dict["_class_name"] = "LTXVideoTransformerModel"
        return_dict["_diffusers_version"] = diffusers_version
        del return_dict["dtype"]
        return return_dict


_LTX_CONFIG_INIT_SIGNATURE = inspect.signature(LTXVideoTransformerModelConfig.__init__)
