import inspect
from typing import Optional, Tuple

import diffusers
from diffusers import QwenImageTransformer2DModel
from transformers import PretrainedConfig


QWEN_IMAGE_INIT_SIGNATURE = inspect.signature(QwenImageTransformer2DModel.__init__)
diffusers_version = diffusers.__version__


class QwenImageTransformer2DModelConfig(PretrainedConfig):
    model_type = "QwenImageTransformer2DModel"
    condition_model_type = "QwenImageConditionModel"

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.guidance_embeds = guidance_embeds
        self.axes_dims_rope = axes_dims_rope
        self.zero_cond_t = zero_cond_t
        self.use_additional_t_cond = use_additional_t_cond
        self.use_layer3d_rope = use_layer3d_rope
        super().__init__(**kwargs)

    def to_diffuser_dict(self):
        return {key: getattr(self, key) for key in QWEN_IMAGE_INIT_SIGNATURE.parameters.keys() if key != "self"}

    def to_dict(self):
        return_dict = super().to_dict()
        return_dict["_class_name"] = "QwenImageTransformer2DModel"
        return_dict["_diffusers_version"] = diffusers_version
        return_dict.pop("dtype", None)
        return return_dict
