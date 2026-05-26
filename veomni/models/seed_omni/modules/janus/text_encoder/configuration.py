"""Config for :class:`JanusTextEncoder`.

Specialises :class:`TextEncoderConfig` with the two Janus-specific boundary
token ids that the model is responsible for emitting around a VQ image
span:

* ``begin_of_image_token_id``  — :code:`<begin_of_image>` (default ``100016``).
* ``end_of_image_token_id``    — :code:`<end_of_image>`   (default ``100593``).

Why a Janus-specific subclass?
------------------------------
The plain :class:`TextEncoder` is vocab-bound but model-agnostic and has
*no* notion of boundary tokens.  Emitting :code:`<begin_of_image>` /
:code:`<end_of_image>` around a VQ image span is a Janus-specific
behaviour — that knowledge lives in the model, not in the FSM
framework.  See the design discussion under "Boundary tokens are model
state" in ``design.md``.

The framework reaches these emit methods via dedicated graph nodes
(``emit_image_start`` / ``emit_image_end``) declared in the YAML.
"""

from ...base.text_encoder.configuration import TextEncoderConfig


class JanusTextEncoderConfig(TextEncoderConfig):
    """TextEncoder config + Janus image-boundary token ids.

    Defaults are the Janus-1.3B tokenizer values; ``scripts/split_janus.py``
    re-reads them from the actual tokenizer and writes the result into
    ``config.json`` so reloads are checkpoint-faithful.
    """

    model_type = "janus_text_encoder"

    def __init__(
        self,
        begin_of_image_token_id: int = 100016,
        end_of_image_token_id: int = 100593,
        **kwargs,
    ) -> None:
        self.begin_of_image_token_id = begin_of_image_token_id
        self.end_of_image_token_id = end_of_image_token_id
        super().__init__(**kwargs)
