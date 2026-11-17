"""How a Qwen3-VL checkpoint splits into SeedOmni modules.

Registered under ``OMNI_HF_LAYOUT_REGISTRY["qwen3_vl"]``. Point
``model.model_path`` at an upstream Qwen3-VL checkpoint to train / infer from it
directly, or split it offline with ``scripts/seed_omni/convert_model.py`` (which
still goes through the family converter — see
:mod:`veomni.models.seed_omni.modules.qwen3vl.convert_model`). Graphs come from
``configs/seed_omni/Qwen/qwen3vl/`` (``train/`` DAG, ``infer/`` FSM).

Modules::

    qwen3vl_vision        # model.visual.*                           (+ image / video processor)
    qwen3vl_text_encoder  # model.language_model.embed_tokens (+ lm_head if untied) + tokenizer
    qwen3vl_llm           # the rest of model.language_model.*        (decoder layers + final norm)

The three module boundaries are exactly the upstream key prefixes, so every
source tensor routes to one module and exports back under its own name. The only
transform the split applies is dropping ``embed_tokens`` from the backbone (that
module rebuilds it as :class:`torch.nn.Identity`), which the prefixes already
express: ``model.language_model.embed_tokens.`` is claimed by the text encoder
and wins over the backbone's shorter ``model.language_model.``.
"""

from __future__ import annotations

from typing import Any

from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .llm.configuration import Qwen3VLLlmConfig
from .text_encoder.configuration import Qwen3VLTextEncoderConfig
from .vision.configuration import Qwen3VLVisionEncoderConfig


def _vision_config(hf_config: Any) -> Qwen3VLVisionEncoderConfig:
    return Qwen3VLVisionEncoderConfig(vision_config=hf_config.vision_config.to_dict())


def _tie_word_embeddings(hf_config: Any) -> bool:
    """Whether the checkpoint ties its embedding to the LM head.

    ``Qwen3VLConfig`` declares ``tie_word_embeddings`` as its own field defaulting
    to ``False``, so it is always present and always wins — which is correct for
    the official checkpoints, where the value lives on the top level (2B ties, 8B
    does not). A nested ``text_config`` value is only consulted when the outer one
    is genuinely absent, which happens for a duck-typed config rather than a real
    ``Qwen3VLConfig``.
    """
    outer = getattr(hf_config, "tie_word_embeddings", None)
    if outer is not None:
        return bool(outer)
    nested = getattr(hf_config.text_config, "tie_word_embeddings", None)
    return True if nested is None else bool(nested)


def _text_encoder_config(hf_config: Any) -> Qwen3VLTextEncoderConfig:
    text_config = hf_config.text_config
    return Qwen3VLTextEncoderConfig(
        vocab_size=text_config.vocab_size,
        hidden_size=text_config.hidden_size,
        tie_word_embeddings=_tie_word_embeddings(hf_config),
        lm_head_bias=False,
    )


def _llm_config(hf_config: Any) -> Qwen3VLLlmConfig:
    # `spatial_merge_size` is copied from the vision config so the backbone can
    # rebuild M-RoPE position ids from each item's `grid_thw` without a reference
    # to the vision module.
    return Qwen3VLLlmConfig(
        text_config=hf_config.text_config.to_dict(),
        spatial_merge_size=hf_config.vision_config.spatial_merge_size,
    )


def _tokenizer(model_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    return {"tokenizer": AutoTokenizer.from_pretrained(model_path)}


def _vision_assets(model_path: str) -> dict[str, Any]:
    """Both processors the vision module binds: image patches and video frames.

    Qwen3-VL pairs a ``Qwen2VLImageProcessor`` (images) with a
    ``Qwen3VLVideoProcessor`` (temporal patchify), each from its own config file
    in the upstream root. :class:`~transformers.AutoProcessor` reads both in one
    call, exactly as the family converter does.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    return {"image_processor": processor.image_processor, "video_processor": processor.video_processor}


QWEN3VL_HF_LAYOUT = OmniHFLayout(
    modules={
        "qwen3vl_vision": OmniHFModuleLayout(
            key_prefixes=(("model.visual.", "visual."),),
            build_config=_vision_config,
            build_assets=_vision_assets,
        ),
        "qwen3vl_text_encoder": OmniHFModuleLayout(
            key_prefixes=(
                ("model.language_model.embed_tokens.", "embed_tokens."),
                ("lm_head.", "lm_head."),
            ),
            build_config=_text_encoder_config,
            build_assets=_tokenizer,
        ),
        "qwen3vl_llm": OmniHFModuleLayout(
            key_prefixes=(("model.language_model.", "language_model."),),
            build_config=_llm_config,
        ),
    },
    # Qwen3-VL-2B ties the head yet still stores the copy.
    tied_source_keys={"lm_head.weight": "model.language_model.embed_tokens.weight"},
    graph_dir="configs/seed_omni/Qwen/qwen3vl",
    training_graphs="train/graph_train.yaml",
    generation_graphs={"vision_understanding": "infer/graph_infer.yaml"},
    infer_type="vision_understanding",
)


@OMNI_HF_LAYOUT_REGISTRY.register("qwen3_vl")
def _register_qwen3vl_hf_layout(source_path: str | None = None) -> OmniHFLayout:
    return QWEN3VL_HF_LAYOUT


__all__ = ["QWEN3VL_HF_LAYOUT"]
