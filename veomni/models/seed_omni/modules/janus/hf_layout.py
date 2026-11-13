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
import os
from collections.abc import Mapping
from typing import Any

import torch

from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .llama.configuration import JanusLlamaConfig
from .siglip.configuration import JanusSiglipConfig
from .text_encoder.configuration import JanusTextEncoderConfig
from .vqvae.configuration import JanusVqvaeConfig


_TIED_EMBEDDING_SOURCE_KEY = "__janus_tied_embedding_source__"
"""Attribute set by :meth:`JanusHFLayout.read_hf_config`: should the text encoder tie its head?

``True`` when the checkpoint stores no ``lm_head.weight`` (a tied export) or
stores it byte-identical to the embedding (a tied model that wrote the duplicate
out); ``False`` when the two differ; ``None`` when the weights could not be
inspected. ``build_config`` only receives a config, so the verdict travels on it;
the name is private to this module and never serialised.
"""


def _tie_verdict_from_weights(
    source_path: str, weight_files: Mapping[str, str], head_key: str, embed_key: str
) -> str | None:
    """How the stored tensors relate to each other: ``"absent"`` / ``"equal"`` / ``"different"``.

    The three cases want different answers, so the verdict is not a bool:

    * ``"absent"`` — no head is stored, so the head *is* the embedding. Tying is the
      only option; an independent head would be randomly initialised.
    * ``"equal"`` — a tied model that wrote the duplicate out. Tying keeps SFT from
      drifting the pair apart, but the config may still say they are separate.
    * ``"different"`` — a genuinely independent head, which must load its own copy.

    Returns ``None`` when there is nothing to read.
    """

    from safetensors import safe_open

    if head_key not in weight_files:
        return "absent"
    if embed_key not in weight_files:
        return None
    tensors = {}
    for key in (head_key, embed_key):
        with safe_open(os.path.join(source_path, weight_files[key]), framework="pt", device="cpu") as f:
            tensors[key] = f.get_tensor(key)
    head, embed = tensors[head_key], tensors[embed_key]
    if head.shape != embed.shape or head.dtype != embed.dtype:
        return "different"
    return "equal" if torch.equal(head, embed) else "different"


def _siglip_config(hf_config: Any) -> JanusSiglipConfig:
    return JanusSiglipConfig(vision_config=hf_config.vision_config.to_dict())


def _vqvae_config(hf_config: Any) -> JanusVqvaeConfig:
    return JanusVqvaeConfig(vq_config=hf_config.vq_config.to_dict())


def _config_allows_tie(hf_config: Any) -> bool:
    """Whether the config itself asks for a tie.

    Both flags are consulted — the AND that
    :func:`~veomni.models.checkpoint.weights.post_process_after_weight_loading`
    applies to the loaded module. An explicit ``False`` on either side is the
    author's decision and outranks anything the tensors look like.
    """
    outer = getattr(hf_config, "tie_word_embeddings", None)
    nested = getattr(getattr(hf_config, "text_config", None), "tie_word_embeddings", None)
    if outer is None and nested is None:
        return True
    return bool(True if outer is None else outer) and bool(True if nested is None else nested)


def _tie_word_embeddings(hf_config: Any) -> bool:
    """Whether the text encoder can tie its head to the input embedding.

    The config states intent, the weights state fact, and each stored-tensor case
    wants a different answer:

    * **No stored head** — the head *is* the embedding. Tying is the only option:
      an independent head would be **randomly initialised**, which silently
      corrupts every logit. This outranks the config, because an outer ``True``
      beside an untied ``text_config`` is exactly how a tied export looks.
    * **A stored head identical to the embedding** — a tied model that wrote the
      duplicate out. Tie unless the config explicitly asks for separate weights, so
      a deliberate ``tie_word_embeddings: false`` is not overridden but a tied pair
      cannot drift apart under SFT.
    * **A stored head with different values** — load it independently.

    With nothing to inspect, fall back to the config's word — the same AND
    :func:`~veomni.models.checkpoint.weights.post_process_after_weight_loading`
    applies to the loaded module.
    """
    verdict = getattr(hf_config, _TIED_EMBEDDING_SOURCE_KEY, None)
    if verdict == "absent":
        return True
    if verdict == "different":
        return False
    # ``"equal"`` (or nothing to inspect): the config decides.
    return _config_allows_tie(hf_config)


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


class JanusHFLayout(OmniHFLayout):
    """Janus' upstream layout, deciding the head/embedding tie from the weight files.

    ``read_hf_config`` is the only place that sees the checkpoint's tensors before
    the module configs are built, so the tie verdict is recorded on the config
    there and :func:`_tie_word_embeddings` reads it back.
    """

    def read_hf_config(self, source_path: str):
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(source_path)
        try:
            weight_files = self.source_weight_files(source_path)
        except FileNotFoundError:
            # No safetensors weights to inspect (e.g. a bin-only or fixture root):
            # fall back to the config flags alone.
            setattr(hf_config, _TIED_EMBEDDING_SOURCE_KEY, None)
        else:
            setattr(
                hf_config,
                _TIED_EMBEDDING_SOURCE_KEY,
                _tie_verdict_from_weights(
                    source_path, weight_files, "lm_head.weight", "model.language_model.embed_tokens.weight"
                ),
            )
        return hf_config


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


JANUS_HF_LAYOUT = JanusHFLayout(
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
        "infer_interleave": "infer/graph_infer_interleave.yaml",
    },
    infer_type="infer_interleave",
    generation_kwargs=dict(JANUS_GENERATION_KWARGS),
)


@OMNI_HF_LAYOUT_REGISTRY.register("janus")
def _register_janus_hf_layout(source_path: str | None = None) -> OmniHFLayout:
    return JANUS_HF_LAYOUT


__all__ = ["JANUS_HF_LAYOUT"]
