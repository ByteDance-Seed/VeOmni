"""Config for :class:`Qwen3TextEncoder`."""

from ...text_encoder.configuration import TextEncoderConfig


class Qwen3TextEncoderConfig(TextEncoderConfig):
    """TextEncoder config for Qwen3 ChatML tokenization.

    A single optional knob (off by default → plain text-only Qwen3) lets the same
    module bootstrap a text LLM into image understanding without a separate module
    type. It is flipped at train time through a per-module ``model_config:``
    override in ``modules_train.yaml``:

    * ``enable_image`` — use the Qwen3-VL image/video-aware ChatML template (media
      wrapped in ``<|vision_start|> … <|vision_end|>``, one block per video frame)
      and accept image/video items on the carrier. When ``False`` the
      text-only template is used verbatim and a media item is an error. With it
      on, ``freeze_model()`` on the accelerated wrapper also trains *only* the vision
      special-token embedding rows — their ids are resolved from the module's own
      tokenizer, so no token-id list is configured — and ``configure_optimizer``
      applies the module's weight decay to those rows only.
    """

    model_type = "qwen3_text_encoder"

    def __init__(
        self,
        enable_image: bool = False,
        **kwargs,
    ):
        self.enable_image = enable_image
        super().__init__(**kwargs)
