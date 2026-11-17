"""Config for :class:`Qwen3VLLlm`."""

from typing import Any, Dict, Optional

from transformers import Qwen3VLTextConfig

from ...module_configuration_base import OmniModuleConfig


class Qwen3VLLlmConfig(OmniModuleConfig):
    """Top-level config for the Qwen3-VL AR backbone (no wte, no lm_head).

    ``spatial_merge_size`` is copied from the vision config so the backbone can
    rebuild multimodal RoPE (M-RoPE) position ids from each image item's
    ``grid_thw`` without holding a reference to the vision module.
    """

    model_type = "qwen3vl_llm"
    # Register the nested backbone config so transformers propagates
    # ``attn_implementation`` (e.g. veomni_flash_attention_2) down to it —
    # otherwise the inner ``Qwen3VLTextModel`` silently falls back to SDPA and
    # breaks under sequence parallelism (sliced q vs full-length mask).
    sub_configs = {"text_config": Qwen3VLTextConfig}

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        spatial_merge_size: int = 2,
        **kwargs,
    ):
        self.text_config = Qwen3VLTextConfig(**text_config) if text_config else Qwen3VLTextConfig()
        self.spatial_merge_size = spatial_merge_size
        super().__init__(**kwargs)
