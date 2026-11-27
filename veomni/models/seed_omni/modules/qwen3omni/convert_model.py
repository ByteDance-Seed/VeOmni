"""Split a Qwen3-Omni checkpoint into SeedOmni module subfolders.

Registered under ``OMNI_CONVERT_REGISTRY["qwen3_omni_moe"]`` and dispatched by
``scripts/convert_model.py`` (via :func:`convert_checkpoint`).

Output layout::

    <output_dir>/
      qwen3omni_audio/         # audio tower + WhisperFeatureExtractor config
      qwen3omni_vision/        # ViT + patch merger + image/video processor configs
      qwen3omni_text_encoder/  # embed_tokens + lm_head (untied) + tokenizer
      qwen3omni_llm/           # thinker backbone (no embed_tokens / no lm_head)
      qwen3omni_talker/        # talker + code predictor
      qwen3omni_codec/         # Qwen's code2wav  +  kyutai/mimi encoder

Two things here are unlike the other model families:

* **The codec needs a second checkpoint.** Qwen3-Omni ships a decoder but no
  audio tokenizer, so the encoder half comes from ``kyutai/mimi``
  (https://huggingface.co/kyutai/mimi — see ``codec/modeling.py`` for how that
  was verified). The ``mimi_path`` kwarg points at it — ``--extra mimi_path=...``
  from the shared convert CLI — and the split fails loudly rather than emitting a
  codec that can only decode.
* **Weights are streamed, not loaded.** The 30B-A3B checkpoint is 28010 tensors
  over 15 shards, and instantiating the upstream container to re-save it would
  need ~60 GB of host RAM for no benefit. Each shard is opened once and its
  tensors are routed to whichever module claims the prefix.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, Tuple

import torch
from safetensors import safe_open
from transformers import AutoProcessor, AutoTokenizer
from transformers.initialization import no_init_weights

from veomni.models.checkpoint.weights import init_empty_weights
from veomni.utils import logging

from ...utils.convert_registry import OMNI_CONVERT_REGISTRY, attach_module_assets, load_family_graphs
from .hf_layout import QWEN3OMNI_HF_LAYOUT


logger = logging.get_logger(__name__)


# The layout names modules ``qwen3omni_<part>``; this converter's build steps
# read them as the short names below.
_BUCKET_FOR_MODULE = {
    "qwen3omni_audio": "audio",
    "qwen3omni_vision": "vision",
    "qwen3omni_text_encoder": "text_encoder",
    "qwen3omni_llm": "llm",
    "qwen3omni_talker": "talker",
    "qwen3omni_codec": "code2wav",
}

# A hub id, not a cluster path: the encoder half is a public checkpoint
# (https://huggingface.co/kyutai/mimi), and a default pointing into internal
# storage would fail for anyone outside it. ``_resolve_mimi_path`` downloads it
# on demand when a hub id is given.
DEFAULT_MIMI_PATH = "kyutai/mimi"


def _resolve_mimi_path(mimi_path: str) -> str:
    """Return a local directory for the Mimi half, downloading it if need be.

    ``mimi_path`` takes either a directory or a hub id, so that the default
    works off-cluster without anyone having to know where the encoder lives.
    """
    if os.path.isdir(mimi_path):
        return mimi_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(mimi_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"qwen3omni convert: could not resolve the Mimi checkpoint {mimi_path!r} as a local directory or a hub "
            f"id ({exc}). Qwen3-Omni ships no audio tokenizer, so the codec's encoder half comes from kyutai/mimi "
            f"(https://huggingface.co/kyutai/mimi) — "
            f"pass --extra mimi_path=<dir-or-hub-id>. Emitting a decode-only codec would break talker training "
            f"later, far from here."
        ) from exc


def _iter_checkpoint_tensors(model_path: str) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield every ``(key, tensor)`` in a sharded or single-file checkpoint.

    Opens each shard exactly once so the peak footprint is one shard, not the
    whole model.
    """
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as handle:
            weight_map: Dict[str, str] = json.load(handle)["weight_map"]
        by_shard: Dict[str, list] = defaultdict(list)
        for key, shard in weight_map.items():
            by_shard[shard].append(key)
        shards = sorted(by_shard.items())
    else:
        shards = [("model.safetensors", None)]

    for shard, keys in shards:
        path = os.path.join(model_path, shard)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in keys if keys is not None else handle.keys():
                yield key, handle.get_tensor(key)


def _route_thinker_and_talker(model_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """Stream the checkpoint once, bucketing tensors by destination module.

    Routed through the family's :data:`QWEN3OMNI_HF_LAYOUT` rather than a second
    prefix table: the layout is already the single statement of "which upstream
    tensor belongs to which module", and it is what a direct ``model.model_path``
    load uses. A private copy here could drift from it — the two would then split
    the same checkpoint differently depending on whether you converted offline or
    loaded in place, which is invisible until weights silently land in the wrong
    module.

    The layout names modules the way the omni checkpoint does
    (``qwen3omni_audio``); the buckets below key them the way this converter's
    build steps read them (``audio``).
    """
    buckets: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in _iter_checkpoint_tensors(model_path):
        routed = QWEN3OMNI_HF_LAYOUT.route(key)
        if routed is None:
            raise ValueError(
                f"qwen3omni convert: tensor '{key}' matches no module prefix in the family layout. The "
                f"checkpoint layout changed; routing it to the wrong module would silently produce a broken split."
            )
        module, module_key = routed
        buckets[_BUCKET_FOR_MODULE[module]][module_key] = tensor
    return buckets


def _fuse_moe_experts(state_dict: Dict[str, torch.Tensor], num_experts: int) -> Dict[str, torch.Tensor]:
    """Rewrite per-expert MoE tensors into the fused layout the modeling expects.

    On disk each expert is its own tensor (``experts.7.gate_proj.weight``); the
    transformers-v5 modeling wants one stacked ``experts.gate_up_proj`` per
    layer. Reuses the runtime loader's converter rather than restacking here,
    because the concat order and which projections merge are easy to get subtly
    wrong and produce a model that loads cleanly and generates noise.
    """
    from veomni.models.transformers.qwen3_omni_moe.checkpoint_tensor_converter import (
        Qwen3OmniMoeCheckpointTensorConverter,
    )

    converter = Qwen3OmniMoeCheckpointTensorConverter(num_experts=num_experts)
    fused: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if not converter.can_handle(key):
            fused[key] = tensor
            continue
        converted = converter.convert(key, tensor)
        if converted is not None:
            fused[converted.name] = converted.tensor
    converter.finalize()  # raises if any expert group came out incomplete
    return fused


def _build(model_cls: Callable, config: Any):
    with no_init_weights(), init_empty_weights():
        return model_cls._from_config(config)


def convert_qwen3omni_checkpoint(
    model_path: str,
    mimi_path: str = DEFAULT_MIMI_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    """Split an upstream Qwen3-Omni checkpoint into six SeedOmni modules."""
    training_graphs, generation_graphs = load_family_graphs(
        "configs/seed_omni/Qwen/qwen3omni_30b_a3b",
        training={
            "thinker": "train/thinker/graph_train.yaml",
            "talker": "train/talker/graph_train.yaml",
            "joint": "train/joint/graph_train.yaml",
        },
        generation={
            "audio_output": "infer/graph_infer_audio.yaml",
            "thinker_only": "infer/graph_infer_thinker_only.yaml",
        },
        training_graph=kwargs.pop("training_graph", None),
        generation_graph=kwargs.pop("generation_graph", None),
    )
    # The modules are instantiated on the meta device only, so no attention
    # backend is ever built; ``attn_implementation`` is still accepted because the
    # shared CLI passes it to every family converter.
    kwargs.pop("attn_implementation", None)
    del kwargs
    logger.info_rank0(f"Loading Qwen3-Omni config from: {model_path}")
    import veomni.models.seed_omni.modules  # noqa: F401
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY

    with open(os.path.join(model_path, "config.json")) as handle:
        raw = json.load(handle)
    thinker = raw["thinker_config"]
    text_config = thinker["text_config"]
    talker_config = raw["talker_config"]

    logger.info_rank0(f"Streaming {model_path} ...")
    buckets = _route_thinker_and_talker(model_path)
    logger.info_rank0("  routed: " + ", ".join(f"{name}={len(sd)}" for name, sd in sorted(buckets.items())))

    logger.info_rank0("Extracting qwen3omni_audio ...")
    audio_cfg = OMNI_CONFIG_REGISTRY["qwen3omni_audio"]()(
        audio_config=thinker["audio_config"],
        audio_token_id=thinker["audio_token_id"],
    )
    audio = _build(OMNI_MODEL_REGISTRY["qwen3omni_audio"](), audio_cfg)
    # Loaded on the *wrapper*, not `audio.audio_tower`: the layout renames onto
    # the module's own parameter names (`audio_tower.conv2d1.weight`), which is
    # where the wrapper holds the tower. Loading the inner module would expect
    # `conv2d1.weight` and reject every key.
    audio.load_state_dict(buckets["audio"], assign=True, strict=True)

    logger.info_rank0("Extracting qwen3omni_vision ...")
    vision_cfg = OMNI_CONFIG_REGISTRY["qwen3omni_vision"]()(vision_config=thinker["vision_config"])
    vision = _build(OMNI_MODEL_REGISTRY["qwen3omni_vision"](), vision_cfg)
    vision.load_state_dict(buckets["vision"], assign=True, strict=True)

    # Qwen3-Omni drives image *and* video off one Qwen2VL-style preprocessor
    # config, and the audio module reads its mel settings from the same file.
    processor = AutoProcessor.from_pretrained(model_path)
    attach_module_assets(vision, image_processor=processor.image_processor, video_processor=processor.video_processor)
    attach_module_assets(audio, processor=processor.feature_extractor)

    logger.info_rank0("Extracting qwen3omni_text_encoder ...")
    # Untied, unlike Qwen3-VL 2B: the checkpoint ships a separate lm_head.
    if bool(text_config.get("tie_word_embeddings", False)):
        raise ValueError(
            "qwen3omni convert: this checkpoint reports tied embeddings, but Qwen3-Omni's thinker ships a "
            "separate lm_head. Refusing to guess which one to keep."
        )
    te_cfg = OMNI_CONFIG_REGISTRY["qwen3omni_text_encoder"]()(
        vocab_size=text_config["vocab_size"],
        hidden_size=text_config["hidden_size"],
        tie_word_embeddings=False,
        lm_head_bias=False,
    )
    te = _build(OMNI_MODEL_REGISTRY["qwen3omni_text_encoder"](), te_cfg)
    te.load_state_dict(buckets["text_encoder"], assign=True, strict=True)
    attach_module_assets(te, tokenizer=AutoTokenizer.from_pretrained(model_path))

    logger.info_rank0("Extracting qwen3omni_llm ...")
    llm_cfg = OMNI_CONFIG_REGISTRY["qwen3omni_llm"]()(
        text_config=text_config,
        spatial_merge_size=thinker["vision_config"]["spatial_merge_size"],
        position_id_per_seconds=thinker["position_id_per_seconds"],
        # The talker reads an intermediate thinker layer rather than the last
        # one; the depth is the talker's to declare, so it is copied from there.
        accept_hidden_layer=talker_config["accept_hidden_layer"],
        # `collect_thinker_states` is deliberately left at its default (off).
        # Whether the thinker has to keep every layer's hidden states is a
        # property of the graph being run — only a talker consumes them — not of
        # the weights, so the scenario's `modules_*.yaml` opts in. Baking it in
        # here made thinker-only training and text-only inference pay for a
        # trace nothing reads.
    )
    llm = _build(OMNI_MODEL_REGISTRY["qwen3omni_llm"](), llm_cfg)
    # The layout renamed `thinker.model.*` onto this module's own names, so the
    # bucket reads `language_model.layers...`. The backbone's `embed_tokens` was
    # already claimed by the text encoder (longest prefix), so there is nothing
    # to drop here — but `nn.Identity()` replaced it, and a leftover key would
    # only surface as a strict-load failure naming the wrong module.
    backbone = _fuse_moe_experts(buckets["llm"], text_config["num_experts"])
    llm.language_model.load_state_dict(
        {k.removeprefix("language_model."): v for k, v in backbone.items()},
        assign=True,
        strict=True,
    )

    logger.info_rank0("Extracting qwen3omni_talker ...")
    talker_cfg = OMNI_CONFIG_REGISTRY["qwen3omni_talker"]()(talker_config=talker_config)
    talker = _build(OMNI_MODEL_REGISTRY["qwen3omni_talker"](), talker_cfg)
    # The talker is its own MoE tower and stores experts the same split way.
    talker_sd = _fuse_moe_experts(buckets["talker"], talker_config["text_config"]["num_experts"])
    talker.load_state_dict(talker_sd, assign=True, strict=True)

    logger.info_rank0(f"Extracting qwen3omni_codec (decoder here, encoder from {mimi_path}) ...")
    mimi_path = _resolve_mimi_path(mimi_path)
    with open(os.path.join(mimi_path, "config.json")) as handle:
        mimi_raw = json.load(handle)
    codec_cfg = OMNI_CONFIG_REGISTRY["qwen3omni_codec"]()(
        code2wav_config=raw["code2wav_config"],
        mimi_config=mimi_raw,
    )
    codec = _build(OMNI_MODEL_REGISTRY["qwen3omni_codec"](), codec_cfg)
    # Same wrapper-relative naming as the other modules: the route keeps the
    # `code2wav.` prefix because that is where the codec holds the vocoder, so
    # the bucket is loaded into `codec` — not `codec.code2wav`, whose parameter
    # names start at `decoder.0.weight`.
    codec.load_state_dict(buckets["code2wav"], assign=True, strict=False)
    mimi_sd = dict(_iter_checkpoint_tensors(mimi_path))
    # Non-strict because the published checkpoint carries decoder-side keys this
    # module does not instantiate, but a *missing* key would leave that
    # parameter on the meta device and surface much later as a corrupt encoder.
    incompatible = codec.mimi.load_state_dict(mimi_sd, assign=True, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            f"qwen3omni convert: the Mimi checkpoint at {mimi_path} is missing "
            f"{len(incompatible.missing_keys)} encoder tensor(s), e.g. {incompatible.missing_keys[:5]}."
        )

    return {
        "modules": {
            "qwen3omni_audio": audio,
            "qwen3omni_vision": vision,
            "qwen3omni_text_encoder": te,
            "qwen3omni_llm": llm,
            "qwen3omni_talker": talker,
            "qwen3omni_codec": codec,
        },
        "training_graphs": training_graphs,
        "generation_graphs": generation_graphs,
        "infer_type": "audio_output",
    }


@OMNI_CONVERT_REGISTRY.register("qwen3_omni_moe")
def _register_qwen3omni_convert():
    return convert_qwen3omni_checkpoint


__all__ = ["convert_qwen3omni_checkpoint"]
