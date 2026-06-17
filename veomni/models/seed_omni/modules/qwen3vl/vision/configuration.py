"""Config for :class:`Qwen3VLVisionEncoder`.

The vision-side hyper-parameters travel under ``vision_config`` — that mirrors
the upstream :class:`transformers.Qwen3VLVisionConfig` schema so
``convert_model.py`` can dump the field unchanged.  ``model_type`` is the lookup
key for ``OMNI_CONFIG_REGISTRY`` / ``OMNI_MODEL_REGISTRY``.
"""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig, Qwen3VLVisionConfig


class Qwen3VLVisionEncoderConfig(PretrainedConfig):
    """Top-level config for the Qwen3-VL vision tower (ViT + patch merger + deepstack).

    Three optional knobs adapt the tower to a different LLM backbone (e.g.
    bootstrapping a text-only Qwen3-0.6B, hidden 1024, onto this 2048-d ViT), all
    applied here by editing the inner ``vision_config`` — no extra modules:

    * ``out_hidden_size``: retarget the patch merger's output to the LLM hidden
      size. The merger's ``linear_fc2`` becomes the trainable projection (its
      stock-shape weights are dropped + re-initialised at load).
    * ``disable_deepstack``: zero out ``deepstack_visual_indexes`` so the ViT
      produces no DeepStack features — a plain (non-Qwen3VL) LLM backbone ignores
      them anyway.
    * ``freeze``: when ``True``, :meth:`Qwen3VLVisionEncoder.freeze_model` freezes
      the ViT and keeps only the patch merger trainable.
    """

    model_type = "qwen3vl_vision"

    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        out_hidden_size: Optional[int] = None,
        disable_deepstack: bool = False,
        freeze: bool = False,
        **kwargs,
    ):
        self.vision_config = Qwen3VLVisionConfig(**vision_config) if vision_config else Qwen3VLVisionConfig()
        if out_hidden_size is not None:
            self.vision_config.out_hidden_size = out_hidden_size
        if disable_deepstack:
            self.vision_config.deepstack_visual_indexes = []
        self.out_hidden_size = out_hidden_size
        self.disable_deepstack = disable_deepstack
        self.freeze = freeze
        super().__init__(**kwargs)
