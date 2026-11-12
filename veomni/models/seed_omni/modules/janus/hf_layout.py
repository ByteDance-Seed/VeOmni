"""How a Janus checkpoint splits into SeedOmni modules.

Registered under ``OMNI_HF_LAYOUT_REGISTRY["janus"]``. Point ``model.model_path``
at an upstream ``JanusForConditionalGeneration`` checkpoint to train / infer
from it directly, or split it offline with ``scripts/seed_omni/convert_model.py``
(which still goes through the family converter — see
:mod:`veomni.models.seed_omni.modules.janus.convert_model` — because a DeepSeek
``multi_modality`` source has to be rewritten to the HF layout first). Graphs
come from ``configs/seed_omni/Janus/janus_1.3b/`` (``train/`` DAG, ``infer/``
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

from typing import Any

from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .llama.configuration import JanusLlamaConfig
from .siglip.configuration import JanusSiglipConfig
from .text_encoder.configuration import JanusTextEncoderConfig
from .vqvae.configuration import JanusVqvaeConfig


_TIED_EMBEDDING_SOURCE_KEY = "__janus_tied_embedding_source__"
"""Attribute set by :func:`_read_hf_config` recording whether the checkpoint stores ``lm_head.weight``.

``build_config`` only receives a config, so the fact has to travel on it. The
name is private to this module and never serialised.
"""


def _read_hf_config(source_path: str):
    """Read the upstream config and record what the weight files say about the head.

    ``transformers`` decides the tie from *both* the flag and the stored keys:
    when ``lm_head.weight`` sits beside ``model.language_model.embed_tokens.weight``
    it declines to tie and warns. Reading the key list here is the only place
    that sees the files, so the decision is recorded for :func:`_tie_word_embeddings`.
    """
    from transformers import AutoConfig

    from ...utils.hf_layout import source_weight_files

    hf_config = AutoConfig.from_pretrained(source_path)
    try:
        stored = set(source_weight_files(source_path))
    except FileNotFoundError:
        # No safetensors weights to inspect (e.g. a bin-only or fixture root):
        # fall back to the flags alone.
        setattr(hf_config, _TIED_EMBEDDING_SOURCE_KEY, None)
    else:
        setattr(hf_config, _TIED_EMBEDDING_SOURCE_KEY, "lm_head.weight" in stored)
    return hf_config


def _siglip_config(hf_config: Any) -> JanusSiglipConfig:
    return JanusSiglipConfig(vision_config=hf_config.vision_config.to_dict())


def _vqvae_config(hf_config: Any) -> JanusVqvaeConfig:
    return JanusVqvaeConfig(vq_config=hf_config.vq_config.to_dict())


def _tie_word_embeddings(hf_config: Any) -> bool:
    """Whether the text encoder can tie its head to the input embedding.

    ``transformers`` ties only when the config asks *and* the checkpoint stores
    no separate ``lm_head.weight``: given both keys it declines to tie and warns
    ("both are present in the checkpoints with different values, so we will NOT
    tie them"). Mirroring that matters in both directions:

    * A checkpoint with the head stored but a stale ``tie_word_embeddings: true``
      (the ``JanusConfig`` dataclass default) must still get its head — reading
      the flag alone silently substituted the embedding for it.
    * A genuinely tied export omits the key, and building a head there would
      initialise it from nothing.

    When the source key list is unavailable, both flags are ANDed, the same
    predicate :func:`~veomni.models.checkpoint.weights.post_process_after_weight_loading`
    applies to the loaded module.
    """
    stored = getattr(hf_config, _TIED_EMBEDDING_SOURCE_KEY, None)
    if stored is not None:
        return not stored

    outer = getattr(hf_config, "tie_word_embeddings", None)
    nested = getattr(getattr(hf_config, "text_config", None), "tie_word_embeddings", None)
    if outer is None and nested is None:
        return True
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
    import json

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
    graph_dir="configs/seed_omni/Janus/janus_1.3b",
    training_graphs="train/graph_train.yaml",
    generation_graphs={
        "infer_gen": "infer/graph_infer_gen.yaml",
        "infer_und": "infer/graph_infer_und.yaml",
        "infer_interleave": "infer/graph_infer_interleave.yaml",
    },
    infer_type="infer_interleave",
    # Tying depends on whether the checkpoint stores ``lm_head.weight``, which
    # only the weight files reveal; this reads them alongside the config.
    load_hf_config=_read_hf_config,
)


@OMNI_HF_LAYOUT_REGISTRY.register("janus")
def _register_janus_hf_layout(source_path: str | None = None) -> OmniHFLayout:
    return JANUS_HF_LAYOUT


__all__ = ["JANUS_HF_LAYOUT"]
