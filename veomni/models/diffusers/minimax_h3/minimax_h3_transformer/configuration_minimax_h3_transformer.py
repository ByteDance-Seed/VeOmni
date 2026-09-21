from transformers import PretrainedConfig


class MiniMaxH3DiTModelConfig(PretrainedConfig):
    model_type = "MiniMaxH3DiTModel"
    condition_model_type = "MiniMaxH3ConditionModel"

    def __init__(
        self,
        hidden_size: int = 5376,
        num_layers: int = 8,
        token_refiner_num_layers: int = 2,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        ffn_hidden_size: int = 14336,
        latents_dim: int = 24,
        audio_latents_dim: int = 32,
        patch_size: tuple = (1, 2, 2),
        text_dim: int = 5120,
        timestep_input_dim: int = 256,
        time_embed_hidden_size: int = 5376,
        time_embed_dim: int = 2688,
        adaln_out_features: int = 96768,
        final_adaln_out_features: int = 10752,
        rope_inv_freq_len: int = 16,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.token_refiner_num_layers = token_refiner_num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.ffn_hidden_size = ffn_hidden_size
        self.latents_dim = latents_dim
        self.audio_latents_dim = audio_latents_dim
        self.patch_size = patch_size
        self.text_dim = text_dim
        self.timestep_input_dim = timestep_input_dim
        self.time_embed_hidden_size = time_embed_hidden_size
        self.time_embed_dim = time_embed_dim
        self.adaln_out_features = adaln_out_features
        self.final_adaln_out_features = final_adaln_out_features
        self.rope_inv_freq_len = rope_inv_freq_len
        self.norm_eps = norm_eps
        self.qk_norm_eps = qk_norm_eps
        self.final_norm_eps = final_norm_eps
