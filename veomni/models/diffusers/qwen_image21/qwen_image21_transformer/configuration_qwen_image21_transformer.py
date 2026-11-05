import inspect
from typing import Optional, Tuple

import diffusers
from diffusers import QwenImage21Transformer2DModel
from transformers import PretrainedConfig


QWEN_IMAGE21_INIT_SIGNATURE = inspect.signature(QwenImage21Transformer2DModel.__init__)
diffusers_version = diffusers.__version__


class QwenImage21Transformer2DModelConfig(PretrainedConfig):
    model_type = "QwenImage21Transformer2DModel"
    condition_model_type = "QwenImage21ConditionModel"

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
        causal_condition: bool = True,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.context_in_dim = context_in_dim
        self.mlp_ratio = mlp_ratio
        self.axes_dims_rope = axes_dims_rope
        self.eps = eps
        self.causal_condition = causal_condition
        super().__init__(**kwargs)

    def validate_build_prerequisites(self):
        from .....distributed.parallel_state import get_parallel_state, is_parallel_state_initialized

        if is_parallel_state_initialized() and get_parallel_state().sp_enabled:
            raise NotImplementedError(
                "Qwen-Image-2.1 sequence parallelism is not implemented yet; set ulysses_size=1 and cp_size=1."
            )

    def to_diffuser_dict(self):
        return {key: getattr(self, key) for key in QWEN_IMAGE21_INIT_SIGNATURE.parameters if key != "self"}

    def to_dict(self):
        return_dict = super().to_dict()
        return_dict["_class_name"] = "QwenImage21Transformer2DModel"
        return_dict["_diffusers_version"] = diffusers_version
        return_dict.pop("dtype", None)
        return return_dict
