"""How a Janus checkpoint splits into SeedOmni modules.

Registered under ``OMNI_HF_LAYOUT_REGISTRY["janus"]``. Point ``model.model_path``
at an upstream ``JanusForConditionalGeneration`` checkpoint to train / infer
from it directly, or split it offline with ``scripts/seed_omni/convert_model.py``
(which still goes through the family converter — see
:mod:`veomni.models.seed_omni.modules.janus.convert_model` — because a DeepSeek
``multi_modality`` source has to be rewritten to the HF layout first). Graphs
come from ``configs/seed_omni/Janus/janus/`` (``train/`` DAG, ``infer/``
FSM).

Modules::

    janus_siglip         # model.vision_model.* + model.aligner.*        (+ image processor)
    janus_vqvae          # model.vqmodel.* + the three generation tensors (+ image processor)
    janus_text_encoder   # model.language_model.embed_tokens (+ lm_head if untied) + tokenizer
    janus_llama          # the rest of model.language_model.*

The four module boundaries are exactly the upstream key prefixes, so every
source tensor routes to one module and exports back under its own name. The only
transform the split applies is dropping ``embed_tokens`` from the backbone (that
module rebuilds it as :class:`torch.nn.Identity`), which the prefixes already
express: ``model.language_model.embed_tokens.`` is claimed by the text encoder
and wins over the backbone's shorter ``model.language_model.``.
"""

from __future__ import annotations

import json
from typing import Any

from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .llama.configuration import JanusLlamaConfig
from .siglip.configuration import JanusSiglipConfig
from .text_encoder.configuration import JanusTextEncoderConfig
from .vqvae.configuration import JanusVqvaeConfig


def _siglip_config(hf_config: Any) -> JanusSiglipConfig:
    return JanusSiglipConfig(vision_config=hf_config.vision_config.to_dict())


def _vqvae_config(hf_config: Any) -> JanusVqvaeConfig:
    return JanusVqvaeConfig(vq_config=hf_config.vq_config.to_dict())


def _tie_word_embeddings(hf_config: Any) -> bool:
    """Whether the text encoder ties its head to the input embedding.

    Read straight from the config: ``tie_word_embeddings`` is a config field, not
    something to infer from the weight values. The framework acts on it —
    :func:`~veomni.models.checkpoint.weights.tie_word_embeddings_if_requested`
    points the head at the embedding when both the outer and nested flags allow
    it — so a value derived from the tensors here would make the module disagree
    with the very code that ties it.

    Only the text encoder has a wte / lm_head pair; every other Janus module
    reports ``get_output_embeddings() is None`` and the hook skips it.
    """
    outer = getattr(hf_config, "tie_word_embeddings", None)
    nested = getattr(getattr(hf_config, "text_config", None), "tie_word_embeddings", None)
    if outer is None and nested is None:
        return False
    # The same AND the framework applies to the loaded module.
    return bool(True if outer is None else outer) and bool(True if nested is None else nested)


def _text_encoder_config(hf_config: Any) -> JanusTextEncoderConfig:
    text_config = hf_config.text_config
    return JanusTextEncoderConfig(
        vocab_size=text_config.vocab_size,
        hidden_size=text_config.hidden_size,
        tie_word_embeddings=_tie_word_embeddings(hf_config),
        lm_head_bias=False,
    )


def _llm_config(hf_config: Any) -> JanusLlamaConfig:
    # `image_token_id` is the understanding-image placeholder and lives on the
    # upstream config. `gen_image_token_id` (``<image_0>``) is a tokenizer fact
    # `build_config` cannot see, and nothing reads either field at runtime today;
    # the family converter still fills both, so a split checkpoint keeps them.
    return JanusLlamaConfig(
        text_config=hf_config.text_config.to_dict(),
        image_token_id=hf_config.image_token_id,
    )


def _tokenizer(model_path: str) -> dict[str, Any]:
    """The byte-level tokenizer, rejecting a rebuilt Metaspace one.

    Janus's vocab is byte-level BPE. An HF export written before
    ``convert_janus_weight_to_hf`` bypassed the mislabelled ``LlamaTokenizer``
    class carries a rebuilt Metaspace pipeline that decodes differently
    (``"TheĠimage"``), so the same check the family converter runs applies here.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        pre_tokenizer = json.loads(backend.to_str()).get("pre_tokenizer") or {}
        if pre_tokenizer.get("type") == "Metaspace":
            raise ValueError(
                f"{model_path} holds a Metaspace tokenizer; Janus needs the byte-level one. "
                "Convert from the DeepSeek checkpoint (the directory without the -hf suffix) instead."
            )
    return {"tokenizer": tokenizer}


def _image_processor(model_path: str) -> dict[str, Any]:
    """The single ``JanusImageProcessor`` both towers bind.

    SigLIP (understanding) and the VQVAE (generation) read the same resize /
    normalise constants from the upstream root's ``preprocessor_config.json``,
    and each module wraps them in its own alias class so a split checkpoint
    keeps a per-module ``preprocessor_config.json``.

    Read the image processor on its own rather than through :class:`JanusProcessor`:
    the wrapper also demands a tokenizer exposing ``image_token``, which the
    understanding-only modules have no use for.
    """
    from transformers.models.janus.image_processing_janus import JanusImageProcessor

    return {"image_processor": JanusImageProcessor.from_pretrained(model_path)}


def _siglip_assets(model_path: str) -> dict[str, Any]:
    from .siglip.processing import JanusSiglipProcessor

    processor = _image_processor(model_path)["image_processor"]
    return {"image_processor": JanusSiglipProcessor(**processor.to_dict())}


def _vqvae_assets(model_path: str) -> dict[str, Any]:
    from .vqvae.processing import JanusVqvaeProcessor

    processor = _image_processor(model_path)["image_processor"]
    return {"image_processor": JanusVqvaeProcessor(**processor.to_dict())}


JANUS_GENERATION_KWARGS: dict[str, Any] = {
    "max_new_tokens": 2048,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "guidance_scale": 5.0,
}
"""Default decode settings for every Janus generation scenario.

``guidance_scale`` is what arms classifier-free guidance on the image span; it
belongs here rather than only in the converter, so an HF checkpoint loaded
directly (no offline convert) generates images the same way.
"""


JANUS_HF_LAYOUT = OmniHFLayout(
    modules={
        "janus_siglip": OmniHFModuleLayout(
            key_prefixes=(
                ("model.vision_model.", "vision_model."),
                ("model.aligner.", "aligner."),
            ),
            build_config=_siglip_config,
            build_assets=_siglip_assets,
        ),
        "janus_vqvae": OmniHFModuleLayout(
            key_prefixes=(
                ("model.vqmodel.", "vqmodel."),
                ("model.generation_embeddings.", "generation_embeddings."),
                ("model.generation_aligner.", "generation_aligner."),
                ("model.generation_head.", "generation_head."),
            ),
            build_config=_vqvae_config,
            build_assets=_vqvae_assets,
        ),
        "janus_text_encoder": OmniHFModuleLayout(
            key_prefixes=(
                ("model.language_model.embed_tokens.", "embed_tokens."),
                ("lm_head.", "lm_head."),
            ),
            build_config=_text_encoder_config,
            build_assets=_tokenizer,
        ),
        "janus_llama": OmniHFModuleLayout(
            key_prefixes=(("model.language_model.", "language_model."),),
            build_config=_llm_config,
        ),
    },
    # A tied Janus checkpoint still stores ``lm_head.weight``; the layout skips
    # that duplicate when the text encoder ties the head instead of holding it.
    # Janus-1.3B is untied, so there the module keeps and loads its own copy.
    tied_source_keys={"lm_head.weight": "model.language_model.embed_tokens.weight"},
    graph_dir="configs/seed_omni/Janus/janus",
    training_graphs="train/graph_train.yaml",
    generation_graphs={
        "infer_gen": "infer/graph_infer_gen.yaml",
        "infer_und": "infer/graph_infer_und.yaml",
    },
    # Text-to-image is the scenario this family exists for; understanding is
    # selected with ``--model.model_config.infer_type infer_und``.
    infer_type="infer_gen",
    generation_kwargs=dict(JANUS_GENERATION_KWARGS),
)


@OMNI_HF_LAYOUT_REGISTRY.register("janus")
def _register_janus_hf_layout(source_path: str | None = None) -> OmniHFLayout:
    return JANUS_HF_LAYOUT


__all__ = ["JANUS_HF_LAYOUT"]
