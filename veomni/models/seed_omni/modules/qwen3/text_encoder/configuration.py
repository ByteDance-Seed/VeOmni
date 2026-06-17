"""Config for :class:`Qwen3TextEncoder`."""

from ...base.text_encoder.configuration import TextEncoderConfig


class Qwen3TextEncoderConfig(TextEncoderConfig):
    """TextEncoder config for Qwen3 ChatML tokenization.

    A single optional knob (off by default → plain text-only Qwen3) lets the same
    module bootstrap a text LLM into image understanding without a separate module
    type. It is meant to be flipped at train time via a per-module
    ``model_config:`` override in ``modules_train.yaml``:

    * ``enable_image`` — use the Qwen3-VL image-aware ChatML template (wraps image
      items in ``<|vision_start|> … <|vision_end|>``) and handle image/video parts
      in the decode hidden-state assembly. When ``False`` the original text-only
      template is used verbatim. With it on, :meth:`Qwen3TextEncoder.freeze_model`
      also trains *only* the vision special-token embedding rows — their ids are
      resolved from the module's own tokenizer, so no token-id list is configured.
    """

    model_type = "qwen3_text_encoder"

    def __init__(
        self,
        enable_image: bool = False,
        **kwargs,
    ):
        self.enable_image = enable_image
        super().__init__(**kwargs)
