"""Configuration for BAGEL's Qwen2 MoT backbone module."""

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class BagelQwen2MoTConfig(Qwen2Config):
    """Qwen2 config with BAGEL's runtime MoT defaults.

    The upstream ``llm_config.json`` is a plain Qwen2 config. BAGEL's loader
    mutates it at runtime with ``qk_norm=True`` and
    ``layer_module="Qwen2MoTDecoderLayer"``; this config bakes those defaults
    into the split module so a SeedOmni checkpoint is self-describing.
    """

    model_type = "bagel_qwen2_mot"

    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        tie_word_embeddings: bool = False,
        qk_norm: bool = True,
        layer_module: str = "Qwen2MoTDecoderLayer",
        freeze_und: bool = False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.qk_norm = qk_norm
        self.layer_module = layer_module
        self.freeze_und = freeze_und
