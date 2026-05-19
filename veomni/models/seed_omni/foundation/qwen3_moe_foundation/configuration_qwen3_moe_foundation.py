from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class Qwen3MoeFoundationConfig(Qwen3MoeConfig):
    model_type = "qwen3_moe_foundation"
    sub_configs = {}

    def __init__(
        self,
        rope_type: str = "1d_rope",
        **kwargs,
    ):
        self.rope_type = rope_type
        super().__init__(**kwargs)
