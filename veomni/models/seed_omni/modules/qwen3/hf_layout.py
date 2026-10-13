"""How a Qwen3 checkpoint splits into SeedOmni modules.

Registered under ``OMNI_HF_LAYOUT_REGISTRY["qwen3"]``. Point ``model.model_path``
at an upstream Qwen3 checkpoint to train / infer from it directly, or split it
offline with ``scripts/seed_omni/convert_model.py``. Graphs come from
``configs/seed_omni/Qwen/qwen3_0.6b/train/``.

Modules::

    qwen3_text_encoder   # model.embed_tokens (+ lm_head if untied) + tokenizer
    qwen3_llm            # the rest of model.* (decoder layers + final norm)
"""

from __future__ import annotations

from typing import Any

from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .llm.configuration import Qwen3LlmConfig
from .text_encoder.configuration import Qwen3TextEncoderConfig


def _text_encoder_config(hf_config: Any) -> Qwen3TextEncoderConfig:
    return Qwen3TextEncoderConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        tie_word_embeddings=hf_config.tie_word_embeddings,
        lm_head_bias=False,
    )


def _llm_config(hf_config: Any) -> Qwen3LlmConfig:
    return Qwen3LlmConfig(text_config=hf_config.to_dict())


def _tokenizer(model_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    return {"tokenizer": AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)}


QWEN3_HF_LAYOUT = OmniHFLayout(
    modules={
        "qwen3_text_encoder": OmniHFModuleLayout(
            key_prefixes=(("model.embed_tokens.", "embed_tokens."), ("lm_head.", "lm_head.")),
            build_config=_text_encoder_config,
            build_assets=_tokenizer,
        ),
        "qwen3_llm": OmniHFModuleLayout(
            key_prefixes=(("model.", "language_model."),),
            build_config=_llm_config,
        ),
    },
    # Small Qwen3 checkpoints tie the head yet still store it.
    tied_source_keys={"lm_head.weight": "model.embed_tokens.weight"},
    graph_dir="configs/seed_omni/Qwen/qwen3_0.6b/train",
    training_graphs="graph_train.yaml",
    generation_graphs={"infer_text": "graph_infer.yaml"},
    infer_type="infer_text",
)


@OMNI_HF_LAYOUT_REGISTRY.register("qwen3")
def _register_qwen3_hf_layout() -> OmniHFLayout:
    return QWEN3_HF_LAYOUT


__all__ = ["QWEN3_HF_LAYOUT"]
