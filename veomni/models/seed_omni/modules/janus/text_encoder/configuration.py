"""Config for :class:`JanusTextEncoder`.

Specialises :class:`TextEncoderConfig` with the two Janus-specific boundary
token ids that the model is responsible for emitting around a VQ image
span:

* ``begin_of_image_token_id``  — :code:`<begin_of_image>` (runtime, via tokenizer).
* ``end_of_image_token_id``    — :code:`<end_of_image>`   (runtime, via tokenizer).

Why a Janus-specific subclass?
------------------------------
The plain :class:`TextEncoder` is vocab-bound but model-agnostic and has
*no* notion of boundary tokens.  Emitting :code:`<begin_of_image>` /
:code:`<end_of_image>` around a VQ image span is a Janus-specific
behaviour — that knowledge lives in the model, not in the FSM
framework (see :class:`JanusTextEncoder` for the emit methods).

The framework reaches these emit methods via dedicated graph nodes
(``emit_image_start`` / ``emit_image_end``) declared in the YAML.
"""

from ...base.text_encoder.configuration import TextEncoderConfig


class JanusTextEncoderConfig(TextEncoderConfig):
    """TextEncoder config + Janus image-boundary token ids.

    ``begin_of_image_token_id`` / ``end_of_image_token_id`` are **not**
    constructor parameters — they are resolved at runtime from the module's
    own tokenizer asset (``tokenizer.json`` in the checkpoint folder).
    """

    model_type = "janus_text_encoder"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
