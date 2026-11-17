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

import torch

from ...utils.hf_layout import (
    OMNI_HF_LAYOUT_REGISTRY,
    OmniHFLayout,
    OmniHFModuleLayout,
    read_source_tensors,
    read_source_weight_files,
)
from .llama.configuration import JanusLlamaConfig
from .siglip.configuration import JanusSiglipConfig
from .text_encoder.configuration import JanusTextEncoderConfig
from .vqvae.configuration import JanusVqvaeConfig


def _siglip_config(hf_config: Any) -> JanusSiglipConfig:
    return JanusSiglipConfig(vision_config=hf_config.vision_config.to_dict())


def _vqvae_config(hf_config: Any) -> JanusVqvaeConfig:
    return JanusVqvaeConfig(vq_config=hf_config.vq_config.to_dict())


_TIED_HEAD_SOURCE_KEY = "lm_head.weight"
_TIED_HEAD_TARGET_KEY = "model.language_model.embed_tokens.weight"


def _stored_head_relation(source_path: str) -> str:
    """How the checkpoint's head relates to its embedding: ``absent`` / ``equal`` / ``different``.

    Mirrors what ``PreTrainedModel.tie_weights`` does during ``from_pretrained``:
    it consults the *stored* tensors, not only the config. Three cases, and all
    three want a different answer:

    * **absent** — upstream ties the pair and omitted the duplicate, so the head
      *is* the embedding. Building an independent head leaves it randomly
      initialised, because there is no weight to load into it.
    * **equal** — a tied export that wrote the duplicate out. Upstream keeps them
      tied (``torch.equal`` passes), so the module must too.
    * **different** — a genuinely separate head. Upstream logs that the config
      disagrees with the checkpoint and deliberately does *not* tie, so the head
      must load its own copy.

    Both serializations are read: ``.bin`` checkpoints store the pair as two
    keys referencing one storage just as often as safetensors writes an equal
    copy, and a config-only answer would drop a trained head there.

    Returns ``"unknown"`` when the tensors cannot be read; the caller then falls
    back to the config.
    """
    try:
        weight_files = read_source_weight_files(source_path)
    except (FileNotFoundError, OSError, ValueError):
        return "unknown"

    if _TIED_HEAD_SOURCE_KEY not in weight_files:
        return "absent"
    if _TIED_HEAD_TARGET_KEY not in weight_files:
        return "unknown"
    try:
        tensors = read_source_tensors(source_path, weight_files, (_TIED_HEAD_SOURCE_KEY, _TIED_HEAD_TARGET_KEY))
    except (OSError, ValueError, RuntimeError):
        return "unknown"
    head, embed = tensors[_TIED_HEAD_SOURCE_KEY], tensors[_TIED_HEAD_TARGET_KEY]
    if head.shape != embed.shape or head.dtype != embed.dtype:
        return "different"
    return "equal" if torch.equal(head, embed) else "different"


def _config_allows_tie(hf_config: Any) -> bool:
    """Whether the config asks for the tie, the way upstream resolves it.

    Upstream expands ``lm_head.weight -> embed_tokens.weight`` from the **outer**
    ``tie_word_embeddings`` alone; the nested ``text_config`` flag does not take
    part. Verified against transformers 5.16.1: an outer ``True`` beside a nested
    ``False`` still yields the tied mapping. Unset counts as ``True``, which is
    the ``PretrainedConfig`` default.
    """
    return bool(getattr(hf_config, "tie_word_embeddings", True))


def _tie_word_embeddings(hf_config: Any, source_path: str | None = None) -> bool:
    """Whether the text encoder ties its head to the input embedding.

    The checkpoint decides when it can, the config otherwise — the same order
    upstream applies in ``from_pretrained``:

    1. no stored head -> tie (there is nothing to load into an independent one);
    2. a stored head that differs from the embedding -> do not tie (upstream
       warns and keeps them separate, so the stored values win over the flag);
    3. a stored head equal to the embedding, or weights we cannot read -> the
       config's flag decides, and ``True`` keeps upstream's shared parameter.

    Only the text encoder has a wte / lm_head pair; every other Janus module
    reports ``get_output_embeddings() is None`` and the tie hook skips it.
    """
    if source_path is not None:
        relation = _stored_head_relation(source_path)
        if relation == "absent":
            return True
        if relation == "different":
            return False
    return _config_allows_tie(hf_config)


def _text_encoder_config(hf_config: Any, source_path: str | None = None) -> JanusTextEncoderConfig:
    text_config = hf_config.text_config
    return JanusTextEncoderConfig(
        vocab_size=text_config.vocab_size,
        hidden_size=text_config.hidden_size,
        tie_word_embeddings=_tie_word_embeddings(hf_config, source_path),
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
