"""Config for :class:`Qwen3OmniVisionEncoder`."""

from typing import Any, Dict, Optional

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeVisionEncoderConfig

from ...module_configuration_base import OmniModuleConfig


class Qwen3OmniVisionEncoderConfig(OmniModuleConfig):
    """Top-level config for the Qwen3-Omni thinker's vision tower.

    The vision hyper-parameters travel under ``vision_config``, unchanged from
    the upstream schema, so the checkpoint split can dump the field verbatim.

    ``freeze`` keeps the ViT fixed while leaving the patch merger trainable —
    the merger is the projection into the thinker's embedding space.

    Unlike the ``qwen3vl`` sibling there are no ``out_hidden_size`` /
    ``disable_deepstack`` retargeting knobs: those exist to bootstrap that ViT
    onto a *differently sized* text backbone, whereas this tower is only ever
    paired with the Qwen3-Omni thinker it shipped with.
    """

    model_type = "qwen3omni_vision"
    # Registered so transformers propagates ``attn_implementation`` (e.g.
    # veomni_flash_attention_2_with_sp) down to the inner ViT. Without it the
    # tower silently falls back to SDPA, which breaks under sequence
    # parallelism: the ViT's Ulysses all-to-all lives in the flash-attention
    # path, so SDPA would leave this rank's patch slice ungathered while
    # cu_seqlens still describe the full sequence.
    sub_configs = {"vision_config": Qwen3OmniMoeVisionEncoderConfig}

    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        freeze: bool = False,
        freeze_merger: bool = False,
        **kwargs,
    ):
        self.vision_config = (
            Qwen3OmniMoeVisionEncoderConfig(**vision_config) if vision_config else Qwen3OmniMoeVisionEncoderConfig()
        )
        self.freeze = freeze
        # ``freeze`` alone is the understanding-SFT recipe: tower frozen, patch
        # merger still adapting into the thinker's embedding space. Talker
        # training wants no visual gradient at all, and not merely to save an
        # optimizer: a trainable merger makes the thinker's ``inputs_embeds``
        # require grad (the dummy anchors are added in as ``x * 0``, which
        # preserves it), so the frozen backbone would build and retain a full
        # activation graph for a backward that never comes.
        self.freeze_merger = freeze_merger
        # This tower owns no embedding / head pair, so the shared post-load tie
        # must stay a no-op. Same guard as the other qwen3omni modules.
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
